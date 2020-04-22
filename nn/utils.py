import cv2
import numpy as np
import random


def _get_new_img_list(partition):
    img_list = []
    tmp_list = []
    file1 = open("./data/list_eval_partition.txt").readlines()
    partition_list = {}
    for line in file1:
        split = line.split()
        partition_list[split[0]] = split[1]
    file2 = open("./data/identity_CelebA.txt").readlines()
    identity = {}
    for line in file2:
        split = line.split()
        if partition_list[split[0]] == partition:
            if split[1] in identity:
                identity[split[1]].append(split[0])
            else:
                identity[split[1]] = [split[0]]
    for key in identity:
        person = identity[key]
        sz = len(person)
        i = 0
        while i < sz:
            ap = []
            for j in range(i, np.min([i+4, sz])):
                ap.append([person[j], key])
            tmp_list.append(ap)
            i = np.min([i+4, sz])
    random.shuffle(tmp_list)
    for i in tmp_list:
        for j in i:
            img_list.append(j)
    return img_list


def celeba_generator(batch_size=64, partition='0'):
    img_list = _get_new_img_list(partition)
    size = len(img_list)
    ind = 0
    while(1):
        x = []
        y = []
        for i in range(ind, np.min([ind + batch_size, size])):
            img = cv2.imread(f"./data/img_align_celeba/{img_list[i][0]}")/255
            x.append(img)
            y.append(int(img_list[i][1]))
        yield np.array(x), np.array(y)
        ind = np.min([ind + batch_size, size]) % size
        if ind == 0:
            img_list = _get_new_img_list(partition)


def get_partition_size(partition='0'):
    partition_list = open("./data/list_eval_partition.txt").readlines()
    res = 0
    for line in partition_list:
        split = line.split()
        if split[1] == partition:
            res += 1
    return res
