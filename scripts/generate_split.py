
from hparam_tuning_project.data.datasets import PytorchDataset
from hparam_tuning_project.utils import PATHS
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import torch
from torch.utils.data import Subset


def check_classes(dataset, train_idxs, val_idxs, targets):
    """Print out the percentage of the train and val labels belonging to each unique class in targets
    """
    train_dataset = Subset(dataset, train_idxs.astype(int))
    val_dataset = Subset(dataset, val_idxs.astype(int))

    train_labels = np.array([i[1] for i in train_dataset])
    val_labels = np.array([i[1] for i in val_dataset])

    # print("targets: ", targets)
    # print("type of targets", type(targets))
    # assert False
    for c in torch.unique(targets):
        # calculate percentage of train indices with this class
        print(f"% of train dataset with class {c}: {(train_labels[train_labels==c.item()].shape[0]/train_labels.shape[0])*100:.4f}%; "
              f"% of val dataset with class {c}: {(val_labels[val_labels==c.item()].shape[0]/val_labels.shape[0])*100:.4f}%")
    print("intersect of train / val (should be empty): ", np.intersect1d(train_idxs, val_idxs))
    print('\n\n')


def main():
    args = parse_args()

    # load dataset
    ds = PytorchDataset.dataset_registry[args.dataset_id]
    dataset = ds(root=PATHS['dataset_path'] + args.dataset_id + '/',
                 train=True,
                 download=True)

    # initialize a list of indices representing each row in the dataset
    idxes = np.arange(len(dataset))

    # do the train test split, stratifying on target classes
    targets = dataset.targets
    if isinstance(targets, list):
        targets = torch.tensor(targets)
    train_idxs, val_idxs = train_test_split(idxes, train_size=0.8, stratify=targets)

    print("=" * 80)
    print(f'checking class distributions for full {args.dataset_id} dataset, contains {len(train_idxs)} training samples and {len(val_idxs)} val samples')
    check_classes(dataset, train_idxs, val_idxs, targets)

    np.savetxt(f"./splits/{args.dataset_id}_train.txt", train_idxs)
    np.savetxt(f"./splits/{args.dataset_id}_val.txt", val_idxs)

    # generate splits for subset of original datdaset
    fracs = [0.75, 0.5, 0.25, 0.1]

    for f in fracs:
        train_subset_idx, _ = train_test_split(train_idxs, train_size=f, stratify=targets[train_idxs])
        val_subset_idx, _ = train_test_split(val_idxs, train_size=f, stratify=targets[val_idxs])

        print("=" * 80)
        print(f"checking class distributions for {f*100}% of {args.dataset_id} dataset, contains {len(train_subset_idx)} training samples and {len(val_subset_idx)} val samples")
        check_classes(dataset, train_subset_idx, val_subset_idx, targets)

        np.savetxt(f"./splits/{args.dataset_id}_{str(f).replace('.', '_')}_train.txt", train_subset_idx)
        np.savetxt(f"./splits/{args.dataset_id}_{str(f).replace('.', '_')}_val.txt", val_subset_idx)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
