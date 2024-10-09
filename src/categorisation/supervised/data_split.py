import random as r

r.seed(478)


class FoldIterable:
    def __init__(self, labels, n_folds, n_class_per_fold):
        self.labels, self.num_folds, self.n_class_per_fold = (
            labels,
            n_folds,
            n_class_per_fold,
        )
        self.folds = self.generate_folds()
        self.current_fold = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_fold >= self.num_folds:
            raise StopIteration

        # get test indices of current fold
        test_indices = self.folds[self.current_fold]
        # get train indices of current fold
        train_indices = [
            idx
            for i, fold in enumerate(self.folds)
            if i != self.current_fold
            for idx in fold
        ]

        self.current_fold += 1
        return train_indices, test_indices

    # generate a list of folds containing plaque indices
    def generate_folds(self):
        labels, n_folds, n_class_per_fold = (
            self.labels,
            self.num_folds,
            self.n_class_per_fold,
        )
        label_indices = {}
        # save relevant indices for each class/label
        for index, label in enumerate(labels):
            if label not in label_indices:
                label_indices[label] = [index]
            else:
                label_indices[label].append(index)
        # shuffle order of indices of each class
        for label in label_indices.keys():
            r.shuffle(label_indices[label])

        # assigns samples to folds based on trying to achieve a balanced distribution between folds
        folds = []
        # populate each fold with indices
        for foldidx in range(n_folds):
            fold = []
            # iterate over each class and its indices
            for indices in label_indices.values():
                # checks if number of data points for a class is smaller than the amount required across all folds
                if len(indices) < n_folds * n_class_per_fold:
                    # adjusts how many fold samples for a class are expected if required amount is not reached
                    skip = round(len(indices) / n_folds)
                else:
                    skip = n_class_per_fold
                if foldidx == n_folds - 1:
                    # if last fold, add all remaining class indices to the fold
                    agg_indices = indices[
                        skip * foldidx : skip * foldidx + n_class_per_fold
                    ]
                else:
                    agg_indices = indices[skip * foldidx : skip * foldidx + skip]
                fold.extend(agg_indices)
                # if class balance for the fold is still not achieved, sample random missing indices from the existing ones
                if len(agg_indices) != n_class_per_fold:
                    missing_size = n_class_per_fold - len(agg_indices)
                    sampled_indices = r.sample(agg_indices, missing_size)
                    fold.extend(sampled_indices)
            folds.append(fold)
        return folds

    # get indices of each created fold as a list
    def get_fold_indices(self):
        all_indices = []
        for fold in self.folds:
            all_indices.extend(fold)
        return all_indices


def get_val_train_indices(labels, val_size, train_size=1000):
    # Create a dictionary to store indices for each class
    label_indices = {}

    # save indices associated with each class
    for index, label in enumerate(labels):
        if label not in label_indices:
            label_indices[label] = [index]
        else:
            label_indices[label].append(index)

    val_indices = []
    train_indices = []
    # distribute val and train indices based on specified sizes
    for _, indices in label_indices.items():
        r.shuffle(indices)
        val_indices.extend(indices[:val_size])
        train_indices.extend(indices[val_size : train_size + val_size])

    return val_indices, train_indices
