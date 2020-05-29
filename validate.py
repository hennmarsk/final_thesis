import cv2
import numpy as np
from tensorflow.keras import backend as K
import nn.my_model


def _euclid(x, y):
    return K.sqrt(K.sum(K.square(x - y), axis=0))


def _name(name, number):
    t = 4 - len(number)
    s = f"./data/lfw_112/{name}/{name}_"
    for i in range(t):
        s += '0'
    s += number
    s += ".jpg"
    return s


model = nn.my_model.create_model([112, 112, 3], "resnet")
model.load_weights(filepath="./weights/weight_best_euclid_resnet.hdf5")

f = open("./data/pairs.txt").readlines()
tp = 0.0
tn = 0.0
fp = 0.0
fn = 0.0
for line in f:
    split = line.split()
    if len(split) == 3:
        # print(_name(split[0], split[1]))
        img1 = cv2.imread(_name(split[0], split[1]))
        img1 = cv2.resize(img1, (112, 112)) / 255
        img2 = cv2.imread(_name(split[0], split[2]))
        img2 = cv2.resize(img2, (112, 112)) / 255
        x = model.predict(np.array([img1, img2]))
        p1 = x[0]
        p2 = x[1]
        print("positive", _euclid(p1, p2))
        if (_euclid(p1, p2) < 0.5):
            tp += 1
        else:
            fp += 1
    elif len(split) == 4:
        img1 = cv2.imread(_name(split[0], split[1]))
        img1 = cv2.resize(img1, (112, 112)) / 255
        img2 = cv2.imread(_name(split[2], split[3]))
        img2 = cv2.resize(img2, (112, 112)) / 255
        x = model.predict(np.array([img1, img2]))
        p1 = x[0]
        p2 = x[1]
        print("negative", _euclid(p1, p2))
        if (_euclid(p1, p2) > 0.5):
            tn += 1
        else:
            fn += 1
print("false positive:", fp)
print("true positive:", tp)
print("false negative:", fn)
print("true negative: ", tn)
print("accuracy:", (tp + tn) / (tp + tn + fp + fn))
