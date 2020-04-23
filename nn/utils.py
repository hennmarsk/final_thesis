import cv2
import numpy as np
import random


def _get_img_list(partition):
    file1 = open("./data/list_eval_partition.txt").readlines()
    partition_list = {}
    for line in file1:
        split = line.split()
        partition_list[split[0]] = split[1]
    file2 = open("./data/identity_CelebA.txt").readlines()
    img_list = {}
    for line in file2:
        split = line.split()
        if partition_list[split[0]] == partition:
            if split[1] in img_list:
                img_list[split[1]].append(split[0])
            else:
                img_list[split[1]] = [split[0]]
    return img_list


def celeba_generator(person_num=16, partition='0'):
    img_list = _get_img_list(partition)
    random.seed(a=None)
    while(1):
        x = []
        y = []
        person_list = random.sample(list(img_list), person_num)
        for person in person_list:
            person_img = img_list[person]
            person_img = random.sample(
                list(person_img), np.min([4, len(person_img)]))
            for img in person_img:
                x.append(cv2.imread(f'./data/img_align_celeba/{img}')/255)
                y.append(int(person))
        yield np.array(x), np.array(y)
