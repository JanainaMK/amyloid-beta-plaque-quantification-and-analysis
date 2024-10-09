import argparse
import os
import sys
import time

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import OPTICS, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.categorisation.file_filter import valid_region


def sample_plaques(features, file_names, local_indices, sample_size):
    selected_indices = np.random.choice(len(features), size=sample_size, replace=False)
    features = features[selected_indices]
    file_names = file_names[selected_indices]
    local_indices = local_indices[selected_indices]
    return features, file_names, local_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="kmeans", type=str)
    parser.add_argument("-dm", "--distance_metric", default="cosine", type=str)
    parser.add_argument("-noc", "--number_of_clusters", default=20, type=int)
    parser.add_argument("-af", "--alex_features", action="store_true")
    parser.add_argument("-rf", "--res_features", action="store_true")
    parser.add_argument("-sf", "--simple_features", action="store_true")
    parser.add_argument("-vf", "--vae_features", action="store_true")
    parser.add_argument("-f", "--fit", action="store_true", default=False)
    parser.add_argument("-p", "--predict", action="store_true", default=False)
    parser.add_argument("-s0", "--start", default=0, type=int)
    parser.add_argument("-s1", "--stop", default=579, type=int)
    parser.add_argument("-r", "--root", default="", type=str)
    parser.add_argument("-tsne", "--tsne", action="store_true", default=False)
    parser.add_argument("-sd", "--standardize", action="store_true", default=False)
    parser.add_argument("-pca", "--pca", action="store_true", default=False)
    args = parser.parse_args()
    print(args)

    model_string = args.model
    distance_metric = args.distance_metric
    n_clusters = args.number_of_clusters
    alex_features = args.alex_features
    resnet_features = args.res_features
    simple_features = args.simple_features
    vae_features = args.vae_features
    root = args.root
    use_tsne = args.tsne
    standardize = args.standardize
    use_pca = args.pca
    start = time.time()

    os.makedirs("result/cluster_model/", exist_ok=True)
    os.makedirs("result/class_assignment/", exist_ok=True)

    m = 0
    features_names = []
    if alex_features:
        m += 1000
        feature_string = "alex"
        features_names.append(feature_string)
    if resnet_features:
        m += 512
        feature_string = "res"
        features_names.append(feature_string)
    if vae_features:
        m += 512
        feature_string = "vae"
        features_names.append(feature_string)
    if simple_features:
        m += 2
        feature_string = "simple"
        features_names.append(feature_string)
    if m == 0:
        raise ValueError("please select features")
    feature_string = "+".join(features_names)

    print("collecting features")
    # concatenate features
    all_features = []
    file_names = []
    local_indices = []
    all_files = os.listdir(root)
    all_files = [f for f in all_files if valid_region(f)]
    all_files = all_files[args.start : args.stop]
    for i, filename in enumerate(all_files):
        with h5py.File(os.path.join(root, filename), "r") as file:
            prev_flag = 0
            n_plaques = file.attrs["n_plaques"]
            local_features = np.empty(n_plaques)
            if alex_features:
                try:
                    alex_feats = file["alex_features"][()]
                except Exception:
                    print("alex features not found in file", filename)
                    continue
                if prev_flag:
                    local_features = np.column_stack((alex_feats, local_features))
                else:
                    local_features = alex_feats
                prev_flag = 1
            if resnet_features:
                try:
                    resnet_feats = file["res_features"][()]
                except Exception:
                    print("resnet features not found in file", filename)
                    continue
                if prev_flag:
                    local_features = np.column_stack((resnet_feats, local_features))
                else:
                    local_features = resnet_feats
                prev_flag = 1
            if vae_features:
                try:
                    vae_feats = file["res_features"][()]
                except Exception:
                    print("vae features not found in file", filename)
                    continue
                if prev_flag:
                    local_features = np.column_stack((vae_feats, local_features))
                else:
                    local_features = vae_feats
                prev_flag = 1
            if simple_features:
                area = file["area"][()]
                roundness = file["roundness"][()]
                if prev_flag:
                    local_features = np.column_stack((roundness, area, local_features))
                else:
                    local_features = np.column_stack((roundness, area))
                prev_flag = 1

            all_features.append(local_features)
            file_names.extend([filename] * n_plaques)
            local_indices.extend(list(range(n_plaques)))
            print(
                i + 1, "/", len(all_files), filename, "collected", time.time() - start
            )

    file_names = np.array(file_names)
    local_indices = np.array(local_indices)
    features = np.concatenate(all_features)
    print("features concatenated, shape:", features.shape, "\n")

    if standardize:
        sample_size = min(len(features), 100000)
        print("standard scaler sample size", sample_size)
        selected_indices = np.random.choice(
            len(features), size=sample_size, replace=False
        )
        temp_features = features[selected_indices]
        print("new features shape used for fitting", temp_features.shape, "\n")

        scaler = StandardScaler().fit(temp_features)
        features = scaler.transform(features)

        feature_string += f"-sd-s{sample_size}"
    if use_pca:
        sample_size = min(len(features), 100000)
        print("pca sample size", sample_size)
        selected_indices = np.random.choice(
            len(features), size=sample_size, replace=False
        )
        temp_features = features[selected_indices]
        print("new features shape used for fitting", temp_features.shape, "\n")

        n_components = 10
        pca = PCA(n_components=n_components).fit(temp_features)
        features = pca.transform(features)

        feature_string += f"-pca-comp{n_components}-s{sample_size}"

    print("features shape:", features.shape, "\n")

    # fit the model based on plaque features
    if model_string == "optics":
        sample_size = min(len(features), 10000)
        print("optics sample size", sample_size)
        features, file_names, local_indices = sample_plaques(
            features, file_names, local_indices, sample_size
        )
        print("new features shape", features.shape)

        min_samples = int(len(features) * 0.05)
        print("optics min samples", min_samples)
        model = OPTICS(min_samples=min_samples, n_jobs=-1)
        print(model_string, "model selected", time.time() - start)

        name_string = f"{model_string}-s{sample_size}-ms{min_samples}-{feature_string}"
        assignments = model.fit_predict(features)
        joblib.dump(model, f"result/cluster_model/{name_string}.joblib")
        print("model saved", time.time() - start)

        space = np.arange(len(model.labels_))
        reachability = model.reachability_[model.ordering_]
        labels = model.labels_[model.ordering_]

        n_clusters = len(set(labels[labels != -1]))
        print("clusters found:", n_clusters)
        np.savez(
            f"result/class_assignment/{name_string}.npz",
            file_name=file_names,
            local_idx=local_indices,
            label=assignments,
        )
        print("assignments saved", time.time() - start)

        _, ax = plt.subplots(figsize=(10, 7))

        # OPTICS reachability plot
        colors = ["g.", "r.", "b.", "y.", "c."]
        for klass, color in zip(range(0, 5), colors):
            Xk = space[labels == klass]
            Rk = reachability[labels == klass]
            ax.plot(Xk, Rk, color, alpha=0.3)
        ax.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
        ax.plot(space, np.full_like(space, 2.0, dtype=float), "k-", alpha=0.5)
        ax.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
        ax.set_ylabel("Reachability (epsilon distance)")
        ax.set_title("Reachability Plot")

        plt.tight_layout()
        os.makedirs("result/optics/", exist_ok=True)
        plt.savefig(f"result/optics/{name_string}-reachability_plot.png", dpi=1200)
    else:
        name_string = (
            f"{model_string}-dm{distance_metric}-nc{n_clusters}-{feature_string}"
        )

        if args.fit:
            if model_string == "kmeans":
                model = KMeans(n_clusters=n_clusters, n_init="auto", init="k-means++")
            elif model_string == "mbkmeans":
                model = MiniBatchKMeans(
                    n_clusters=n_clusters, n_init="auto", init="k-means++"
                )
            else:
                print(model_string, "is not a valid model")
                sys.exit()

            print(model_string, "model selected", time.time() - start)
            assignments = model.fit(features)
            joblib.dump(model, f"result/cluster_model/{name_string}.joblib")
            print("model saved", time.time() - start)
        # generate predictions based on plaque features
        if args.predict:
            model = joblib.load(f"result/cluster_model/{name_string}.joblib")
            print("model loaded")
            assignments = model.predict(features)
            n_clusters = assignments.max() + 1
            print("clusters found:", n_clusters, time.time() - start)

            np.savez(
                f"result/class_assignment/{name_string}.npz",
                file_name=file_names,
                local_idx=local_indices,
                label=assignments,
            )
            print("assignments saved", time.time() - start)

    if use_tsne:
        sample_size = min(len(features), 1000)
        selected_indices = np.random.choice(
            len(features), size=sample_size, replace=False
        )
        features = features[selected_indices]
        assignments = assignments[selected_indices]

        perplexities = [50]
        print("explored perplexities for tsne:", perplexities)
        for perplexity in perplexities:
            tsne = TSNE(n_components=2, perplexity=perplexity)
            print("fitting tsne")
            features = tsne.fit_transform(features)
            plt.figure(figsize=(14, 8))
            # Plot each cluster separately
            for cluster_label in np.unique(assignments):
                cluster_indices = np.where(assignments == cluster_label)
                plt.scatter(
                    features[cluster_indices, 0],
                    features[cluster_indices, 1],
                    label=f"{cluster_label}",
                )

            plt.title(f"t-SNE for {model_string}")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.legend().set_title("Clusters")
            plt.tight_layout()

            os.makedirs("result/tsne/", exist_ok=True)
            plt.savefig(
                f"result/tsne/tsne-perp{perplexity}-ts{sample_size}-{name_string}.png",
                dpi=1200,
            )

    print("finished program for", name_string)


if __name__ == "__main__":
    main()
