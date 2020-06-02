import cv2
import numpy as np
import random
import mxnet as mx
import os
from mxnet import recordio


def _get_img_list():
    file2 = open("./data/identity_CelebA.txt").readlines()
    img_list = {}
    for line in file2:
        split = line.split()
        if int(split[1]) in img_list:
            img_list[int(split[1])].append(split[0])
        else:
            img_list[int(split[1])] = [split[0]]
    return img_list


def celeba_gen(batch_size):
    img_list = _get_img_list()
    key = list(img_list.keys())
    random.seed(a=None)
    while(True):
        p = random.sample(key, 2)
        while len(img_list[p[0]]) < 2:
            p = random.sample(key, 2)
        anc = []
        pos = []
        neg = []
        for i in range(batch_size):
            a = random.sample(img_list[p[0]], 2)
            b = random.sample(img_list[p[1]], 1)
            img1 = cv2.imread(
                f"./data/img_align_celeba_112/{a[0]}") / 255
            img2 = cv2.imread(
                f"./data/img_align_celeba_112/{a[1]}") / 255
            img3 = cv2.imread(
                f"./data/img_align_celeba_112/{b[0]}") / 255
            anc.append(img1)
            pos.append(img2)
            neg.append(img3)
        x = np.array(anc + pos + neg)
        y = np.ndarray(shape=(batch_size * 3, 1))
        yield x, y


def celeba_gen_batch(batch_size, sample_size):
    random.seed(a=None)
    img_list = _get_img_list()
    key = list(img_list.keys())
    while True:
        x = []
        y = []
        people = random.sample(key, np.min([batch_size, len(key)]))
        for person in people:
            imgs = random.sample(img_list[person], np.min(
                [sample_size, len(img_list[person])]))
            for src in imgs:
                img = cv2.imread(
                    f"./data/img_align_celeba_112/{src}") / 255
                x.append(img)
                y.append(int(person))
        yield np.array(x), np.array(y)


def _get_ms1m_list(imgrec):
    ms1m_list = {}
    for i in range(3804847):
        header, s = recordio.unpack(imgrec.read_idx(i + 1))
        label = header.label
        if label in ms1m_list:
            ms1m_list[label].append(i + 1)
        else:
            ms1m_list[label] = [i + 1]
    return ms1m_list


def ms1m_gen(batch_size):
    path_idx = "./data/faces_emore/train.idx"
    path_rec = "./data/faces_emore/train.rec"
    imgrec = recordio.MXIndexedRecordIO(path_idx, path_rec, 'r')
    ms1m_list = np.load("ms1m_list.npy", allow_pickle=True).item()
    keys = list(ms1m_list.keys())
    sz = len(ms1m_list)
    pp = np.arange(sz)
    k = np.arange(sz)
    random.seed(a=None)
    anc = []
    pos = []
    neg = []
    ind = 0
    while True:
        t = random.randint(1, sz - 1)
        random.shuffle(k)
        p = (pp + t) % sz
        for h in k:
            i = h
            j = p[h]
            if len(ms1m_list[keys[i]]) > 1:
                i_p = random.sample(ms1m_list[keys[i]], 2)
                i_n = random.sample(ms1m_list[keys[j]], 1)
                header, s = recordio.unpack(imgrec.read_idx(int(i_p[0])))
                img1 = mx.image.imdecode(s).asnumpy() / 255
                header, s = recordio.unpack(imgrec.read_idx(int(i_p[1])))
                img2 = mx.image.imdecode(s).asnumpy() / 255
                header, s = recordio.unpack(imgrec.read_idx(int(i_n[0])))
                img3 = mx.image.imdecode(s).asnumpy() / 255
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


def ms1m_gen_batch(batch_size, sample_size):
    path_idx = "./data/faces_emore/train.idx"
    path_rec = "./data/faces_emore/train.rec"
    imgrec = recordio.MXIndexedRecordIO(path_idx, path_rec, 'r')
    ms1m_list = np.load("ms1m_list.npy", allow_pickle=True).item()
    keys = list(ms1m_list.keys())
    random.seed(a=None)
    while True:
        x = []
        y = []
        people = random.sample(keys, batch_size)
        for person in people:
            imgs = random.sample(ms1m_list[person], np.min(
                [sample_size, len(ms1m_list[person])]))
            for src in imgs:
                header, s = recordio.unpack(imgrec.read_idx(int(src)))
                img = mx.image.imdecode(s).asnumpy() / 255
                x.append(img)
                y.append(person)
        yield np.array(x), np.array(y)


def _get_list(fl):
    fl_list = {}
    f = os.listdir(f"./data/{fl}_96")
    for folder in f:
        key = f"./data/{fl}_96/{folder}"
        imgs = os.listdir(key)
        for src in imgs:
            data = f"./data/{fl}_96/{folder}/{src}"
            if folder in fl_list:
                fl_list[folder].append(data)
            else:
                fl_list[folder] = [data]
    np.save(f'./{fl}_list.npy', fl_list)


def casia_gen_batch(batch_size, sample_size):
    casia_list = np.load("casia_list.npy", allow_pickle=True).item()
    keys = list(casia_list.keys())
    random.seed(a=None)
    while True:
        x = []
        y = []
        people = random.sample(keys, batch_size)
        for person in people:
            imgs = random.sample(casia_list[person], np.min(
                [sample_size, len(casia_list[person])]))
            for src in imgs:
                img = cv2.imread(src) / 255
                x.append(img)
                y.append(int(person))
        yield np.array(x), np.array(y)
