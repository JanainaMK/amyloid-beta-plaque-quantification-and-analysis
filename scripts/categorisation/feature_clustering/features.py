import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

from src.categorisation.feature_clustering.extract_features import generate_features
from src.categorisation.feature_clustering.pre_processing import alex_prep, rgb_to_dab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s0", "--start", default=0, type=int)
    parser.add_argument("-s1", "--stop", default=579, type=int)
    parser.add_argument("-r", "--root", type=str)
    parser.add_argument("-f", "--feature", default="alex", type=str)
    args = parser.parse_args()

    start_at = args.start
    stop_at = args.stop
    root = args.root
    feature_src = args.feature

    if feature_src not in ["alex", "res", "vae"]:
        raise Exception("Not accepted feature type.")

    if not torch.cuda.is_available():
        print("cuda not availible, failing")
        sys.exit()

    print("starting...")

    filenames = os.listdir(root)

    print(len(filenames), "slides found")

    for j, filename in enumerate(filenames):
        start = time.time()
        if j < start_at:
            continue
        elif j >= stop_at:
            break

        file = h5py.File(os.path.join(root, filename), "a")
        try:
            n = file["plaques"].attrs["length"]
            n_act = 0

            fail = False
            print(filename, "-", n, "plaques")
            if feature_src == "alex":
                features = np.zeros((n, 1000))
            elif feature_src == "res":
                features = np.zeros((n, 512))
            else:
                features = np.zeros((n, 512))
            for i in range(n):
                try:
                    plaque = file[f"plaques/{i}"]
                except KeyError:
                    fail = True
                    continue
                n_act += 1
                image = plaque["plaque"][()]
                pre = rgb_to_dab(image)
                tensor = alex_prep(pre)
                feature = generate_features(tensor, "res").detach().cpu().numpy()
                features[i] = feature
                if f"{feature_src}_feature" in plaque:
                    try:
                        del plaque[f"{feature_src}_feature"]
                    except KeyError as e:
                        print(e)
                        print("plaque deletion failed, exiting")
                        break
                plaque.create_dataset(f"{feature_src}_feature", data=feature)
            if fail:
                print(f"{n_act}/{n} plaques found, feature extraction failed")
                continue
            if f"{feature_src}_features" in file:
                del file[f"{feature_src}_features"]
            file.create_dataset(f"{feature_src}_features", data=features)
        except BlockingIOError:
            print("file", filename, "already open, failing feature collection")
            continue
        except KeyError:
            print(
                "file", filename, "missing length attribute, failing feature collection"
            )
            continue
        finally:
            file.close()
        print(f"{feature_src} features saved", time.time() - start)


if __name__ == "__main__":
    main()
