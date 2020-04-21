import os
from cv2 import cv2
import numpy as np
import keras.applications as applications
from keras.models import Model
from keras.layers import Input, Lambda
from keras import backend as K

def euclidean_distance(x, y):
    sum_square = K.sum(K.square(x - y), axis=0, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


# def cosine_distance(x, y):
#     x = K.l2_normalize(x, axis=-1)
#     y = K.l2_normalize(y, axis=-1)
#     return -K.mean(x * y, axis=-1, keepdims=True)


# def l1_distance(x, y):
#     return K.sum(K.abs(x - y), axis=1, keepdims=True)


def dist_mode(x, y, mode):
    if mode == 'euclidean':
        return euclidean_distance(x, y)
    elif mode == 'cosine':
        return cosine_distance(x, y)
    elif mode == 'l1':
        return l1_distance(x, y)

def triplet_loss(mode, margin=0.6):
    def loss(y_true, y_pred):
        anchor = y_pred[0]
        positive = y_pred[1]
        negative = y_pred[2]
        pos_dist = dist_mode(positive, anchor, mode)
        neg_dist = dist_mode(negative, anchor, mode)
        basic_loss = pos_dist - neg_dist + margin
        return K.maximum(basic_loss, 0.0)
    return loss

def random_indice(path):
    size = len(os.listdir(path))
    a = np.random.permutation(size)
    return a

def generator(input_shape, output_shape, batch_size=64, word='train'):
    t = input_shape
    t.insert(0, 3)
    t.insert(0, batch_size)
    train_batch = np.ndarray(shape=tuple(t))
    ind = 0

    t1 = output_shape
    t1.insert(0, 3)
    t1.insert(0, batch_size)
    dummy = np.ndarray(shape=tuple(t1))

    ind = 0
    a = random_indice(f"/home/tung/final-thesis/data/{word}")
    for positives in a:
        a_positive = random_indice(f"/home/tung/final-thesis/data/{word}/{positives}")
        for anchor in a_positive:
            img_a = cv2.imread(f"/home/tung/final-thesis/data/{word}/{positives}/{anchor}")
            for positive in a_positive:
                if (anchor != positive):
                    img_p = cv2.imread(f"/home/tung/final-thesis/data/{word}/{positives}/{positive}")
                    for negatives in a:
                        if (negatives != positives):
                            for negative in negatives:
                                img_n = cv2.imread(f"/home/tung/final-thesis/data/{word}/{negatives}/{negative}")
                                train_batch[ind] = [img_a, img_p, img_n]
                                ind = (ind + 1) % batch_size
                                print(ind)
                                if (ind == 0):
                                    yield train_batch, dummy
    if ind > 0:
        train_batch = train_batch[:ind]
        dummy = dummy[:ind]
        print("last batch!")
        yield train_batch, dummy


def training(input_shape, mode):
    mobile_net = applications.MobileNetV2(weights=None, input_shape=input_shape)

    anchor = Input(shape=input_shape)
    positive = Input(shape=input_shape)
    negative = Input(shape=input_shape)

    processed_a = mobile_net(anchor)
    processed_p = mobile_net(positive)
    processed_n = mobile_net(negative)

    model = Model(inputs=[anchor, positive, negative], outputs=[processed_a, processed_p, processed_n])
    model.compile(loss=triplet_loss(mode=mode), optimizer='adam')

    model.fit_generator(generator=generator(input_shape=input_shape, output_shape=[1000], batch_size=512), epochs=10, verbose=1, steps_per_epoch=50)
    mobile_net.save_weights(f"{mode}_weights.h5")

input_shape = [218, 178, 3]
training(input_shape, 'euclidean')