from hparam_tuning_project.data.datasets import PytorchDataset
from hparam_tuning_project.utils import PATHS
import numpy as np
from torch.utils.data.dataset import random_split
import argparse


def main():
    args = parse_args()
    ds = PytorchDataset.dataset_registry[args.dataset_id]
    dataset = ds(root=PATHS['dataset_path'] + args.dataset_id + '/',
                 train=True,
                 download=True)

    idxes = np.arange(len(dataset))

    train_size = int(0.8 * len(idxes))
    val_size = len(idxes) - train_size

    # main split
    train_idxs, val_idxs = random_split(idxes, [train_size, val_size])
    np.savetxt(f"./splits/{args.dataset_id}_train.txt", train_idxs)
    np.savetxt(f"./splits/{args.dataset_id}_val.txt", val_idxs)

    # one tenth of usual size
    throwaway = len(idxes) - (train_size // 10) - (val_size // 10)
    train_idxs, val_idxs, _ = random_split(idxes, [train_size // 10, val_size // 10, throwaway])
    np.savetxt(f"./splits/{args.dataset_id}_small_train.txt", train_idxs)
    np.savetxt(f"./splits/{args.dataset_id}_small_val.txt", val_idxs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
