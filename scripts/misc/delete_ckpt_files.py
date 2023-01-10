"""
Removing checkpoint files from dowloaded checkpoints to save space
"""

import os

PATH = "/home/alex/datasets/hparam_results/"


def main():
    for root, dirs, files in os.walk(PATH):
        for f in files:
            if ".ckpt" in f:
                path_to_remove = os.path.join(root, f)
                print("removing ", path_to_remove)
                os.remove(path_to_remove)


if __name__ == '__main__':
    main()
