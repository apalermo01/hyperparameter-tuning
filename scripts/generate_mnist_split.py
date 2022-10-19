from torchvision import datasets as ds
from hparam_tuning_project.utils import PATHS
import numpy as np
from torch.utils.data.dataset import random_split


def main():
    mnist = ds.MNIST(root=PATHS['dataset_path'] + 'mnist' + '/',
                     train=True,
                     download=True)

    idxes = np.arange(len(mnist))

    train_size = int(0.8 * len(idxes))
    val_size = len(idxes) - train_size

    # main split
    train_idxs, val_idxs = random_split(idxes, [train_size, val_size])
    np.savetxt("./splits/mnist_train.txt", train_idxs)
    np.savetxt("./splits/mnist_val.txt", val_idxs)

    # one tenth of usual size
    throwaway = len(idxes) - (train_size // 10) - (val_size // 10)
    train_idxs, val_idxs, _ = random_split(idxes, [train_size // 10, val_size // 10, throwaway])
    np.savetxt("./splits/mnist_small_train.txt", train_idxs)
    np.savetxt("./splits/mnist_small_val.txt", val_idxs)


if __name__ == '__main__':
    main()
