import cv2
import numpy as np


def celeba_generator(batch_size=64, partition='0'):
    partition_list = open("./data/list_eval_partition.txt").readlines()
    img_list = []
    for line in partition_list:
        split = line.split()
        if split[1] == partition:
            img_list.append(split[0])
    size = len(img_list)
    identity_list = open("./data/identity_CelebA.txt").readlines()
    identity = {}
    for line in identity_list:
        identity[line.split()[0]] = line.split()[1]
    for i in range(0, size, batch_size):
        x = []
        y = []
        for j in range(i, np.min([i + batch_size, size])):
            img = cv2.imread(f"./data/img_align_celeba/{img_list[j]}")
            x.append(img)
            y.append(identity[img_list[j]])


def get_partition_size(partition='0'):
    partition_list = open("./data/list_eval_partition.txt").readlines()
    res = 0
    for line in partition_list:
        split = line.split()
        if split[1] == partition:
            res += 1
    return res
