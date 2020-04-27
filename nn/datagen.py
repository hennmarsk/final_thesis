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
            if int(split[1]) in img_list:
                img_list[int(split[1])].append(split[0])
            else:
                img_list[int(split[1])] = [split[0]]
    return img_list


def celeba_gen(batch_size, partition='0'):
    img_list = _get_img_list(partition)
    key = list(img_list.keys())
    sz = len(img_list)
    pp = np.arange(sz)
    k = np.arange(sz)
    random.seed(a=None)
    anc = []
    pos = []
    neg = []
    ind = 0
    while(True):
        t = random.randint(1, sz - 1)
        random.shuffle(k)
        p = (pp + t) % sz
        for h in k:
            i = h
            j = p[h]
            if len(img_list[int(key[i])]) > 1:
                i_p = random.sample(img_list[int(key[i])], 2)
                i_n = random.sample(img_list[int(key[j])], 1)
                img1 = cv2.imread(f'./data/img_align_celeba/{i_p[0]}')/255
                img2 = cv2.imread(f'./data/img_align_celeba/{i_p[1]}')/255
                img3 = cv2.imread(f'./data/img_align_celeba/{i_n[0]}')/255
                anc.append(img1)
                pos.append(img2)
                neg.append(img3)
                ind = (ind + 1) % batch_size
                if ind == 0:
                    x = np.array(anc + pos + neg)
                    y = np.ndarray(shape=(batch_size * 3, 1))
                    yield x, y
                    anc.clear()
                    pos.clear()
                    neg.clear()


def celeba_gen_batch(batch_size, partition='0'):
    random.seed(a=None)
    img_list = _get_img_list(partition)
    key = list(img_list.keys())
    while(True):
        x = []
        y = []
        people = random.sample(key, np.min([batch_size, len(key)]))
        for person in people:
            imgs = random.sample(img_list[person], np.min(
                [4, len(img_list[person])]))
            for src in imgs:
                img = cv2.imread(f"./data/img_align_celeba/{src}") / 255
                x.append(img)
                y.append(int(person))
        yield np.array(x), np.array(y)
