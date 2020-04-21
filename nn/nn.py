import os
import cv2
import numpy as np
import tensorflow
import tensorflow.keras.applications as applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, concatenate
from tensorflow.keras import backend as K

def euclidean_distance(x, y):
    sum_square = K.sum(K.square(x - y), axis=1)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def cosine_distance(x, y):
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)


def l1_distance(x, y):
    return K.sum(K.abs(x - y), axis=1, keepdims=True)


def dist_mode(x, y, mode):
    if mode == 'euclidean':
        return euclidean_distance(x, y)
    elif mode == 'cosine':
        return cosine_distance(x, y)
    elif mode == 'l1':
        return l1_distance(x, y)

def triplet_loss(mode, margin=0.6):
    def loss(y_true, y_pred):
        if (mode == 'cosine'):
            margin = 0.01
        y_len = y_pred.shape.as_list()[-1]
        anchor = y_pred[:, 0:int(y_len * 1 /3)]
        positive = y_pred[:, int(y_len * 1 /3):int(y_len * 2 /3)]
        negative = y_pred[:, int(y_len * 2 /3):y_len]
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
    a = random_indice(f"../data/{word}")
    sz = np.zeros(8192)
    for i in a:
        sz[i] = len(os.listdir(f"../data/{word}/{i}"))
    m = 0
    ind = 0
    for i in a:
        if m < int(sz[i]):
            m = int(sz[i])
    c = np.ndarray(shape=(10177, m*m, 2))
    for i in a:
        b = random_indice(f"../data/{word}/{i}")
        for j in b:
            for k in b:
                c[i][ind] = j,k
                ind += 1
        ind = 0

    t = input_shape
    t.insert(0, batch_size)
    anc = np.ndarray(shape=tuple(t))
    pos = np.ndarray(shape=tuple(t))
    neg = np.ndarray(shape=tuple(t))

    t1 = output_shape
    t1.insert(0, batch_size)
    dummy = np.ndarray(shape=tuple(t1))

    ind = 0
    d = np.zeros(10177)
    for neg_ in a:
        for pos_ in a:
            if pos_ != neg_:
                pair = c[pos_][int(d[pos_])]
                d[pos_]+=1
                img_a = cv2.imread(f"../data/{word}/{pos_}/{int(pair[0])}")
                img_p = cv2.imread(f"../data/{word}/{pos_}/{int(pair[1])}")
                l = sz[neg_]
                k = np.random.randint(0, l-1)
                img_n = cv2.imread(f"../data/{word}/{neg_}/{k}")
                anc[ind] = img_a
                pos[ind] = img_p
                neg[ind] = img_n
                ind = (ind + 1) % batch_size
                yield [anc, pos, neg], dummy
    if (ind > 0):
        yield [anc[:ind], pos[:ind], neg[:ind]], dummy[:ind]
    
def training(input_shape, mode):
    mobile_net = applications.MobileNetV2(weights=None, input_shape=input_shape, classes=128)

    anchor = Input(shape=input_shape)
    positive = Input(shape=input_shape)
    negative = Input(shape=input_shape)

    processed_a = mobile_net(anchor)
    processed_p = mobile_net(positive)
    processed_n = mobile_net(negative)

    merged_vector = concatenate([processed_a, processed_p, processed_n], axis=-1, name='merged_layer')

    model = Model(inputs=[anchor, positive, negative], outputs=merged_vector)
    model.compile(loss=triplet_loss(mode=mode), optimizer='adam')
    model.fit_generator(generator=generator(batch_size=16, input_shape=input_shape, output_shape=[128*3]), steps_per_epoch=10177, epochs=1)
    mobile_net.save_weights(f"{mode}_weights.h5")

input_shape = [218, 178, 3]
training(input_shape, 'cosine')
training(input_shape, 'euclidean')
training(input_shape, 'l1')