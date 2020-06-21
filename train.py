import nn.my_model as my_model
import nn.loss as L
import nn.datagen as dg
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import cv2
import numpy as np


def _euclid(x, y):
    return K.sqrt(K.sum(K.square(x - y), axis=0))


def _cosine(x, y):
    nx = tf.nn.l2_normalize(x, 0)
    ny = tf.nn.l2_normalize(y, 0)
    return 1 - tf.reduce_sum(tf.multiply(nx, ny))


def _distance(x, y, metric):
    if metric == 'euclid':
        return _euclid(x, y)
    elif metric == 'cosine':
        return _cosine(x, y)


def _name(name, number):
    t = 4 - len(number)
    s = f"./data/lfw_96/{name}/{name}_"
    for i in range(t):
        s += '0'
    s += number
    s += ".jpg"
    return s


class validate(Callback):
    def __init__(self, metric, batch_mode):
        super(Callback, self).__init__()
        self.metric = metric
        self.mode = batch_mode

    def on_epoch_end(self, epoch, logs=None):
        f = open("./data/pairs.txt").readlines()
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
        for i, line in enumerate(f):
            split = line.split()
            if len(split) == 3:
                # print(_name(split[0], split[1]))
                img1 = cv2.imread(_name(split[0], split[1])) / 255
                img2 = cv2.imread(_name(split[0], split[2])) / 255
                x = self.model.predict(np.array([img1, img2]))
                p1 = x[0]
                p2 = x[1]
                if (_distance(p1, p2, self.metric) < 0.5):
                    tp += 1
                else:
                    fp += 1
            elif len(split) == 4:
                img1 = cv2.imread(_name(split[0], split[1])) / 255
                img2 = cv2.imread(_name(split[2], split[3])) / 255
                x = self.model.predict(np.array([img1, img2]))
                p1 = x[0]
                p2 = x[1]
                if (_distance(p1, p2, self.metric) >= 0.5):
                    tn += 1
                else:
                    fn += 1
            print(f'{int(i / 60)} %\r', end='')
        acc = (tp + tn) / (tp + tn + fp + fn)
        print("acc:", acc)
        f = open(f'./log_{self.metric}_{self.mode}.txt', 'a')
        f.write(
            f'{epoch};{logs["p_a"]};{acc};{tp};{fp};{tn};{fn}\n')
        f.close()


class base:
    def __init__(self):
        self.input_shape = [96, 96, 3]
        self.model = my_model.create_model(self.input_shape)
        self.batch = 80
        self.step_t = 3200
        self.sample = 4
        self.epochs = 1000
        self.learning_rate = 1e-3
        self.alpha = 1 / self.sample / self.batch
        self.beta = 1 / self.sample / self.batch
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)

    def train(self, metric, batch_mode, pretrain=''):
        if (len(pretrain) > 0):
            self.model.load_weights(pretrain)
        self.model.compile(
            loss=L.batch_all(metric, alpha=self.alpha,
                             beta=self.beta, mode=batch_mode),
            optimizer=self.optimizer,
            metrics=[L.pos_all(metric)])
        val = validate(metric, batch_mode)
        checkpoint = ModelCheckpoint(
            f"./weights/best_{metric}_{batch_mode}.hdf5",
            monitor='p_a', verbose=1,
            save_best_only=True,
            mode='auto', save_freq='epoch')
        callbacks_list = [checkpoint, val]
        self.model.fit(
            x=dg.casia_gen_batch(
                self.batch, self.sample),
            epochs=self.epochs,
            steps_per_epoch=self.step_t,
            callbacks=callbacks_list)
        self.model.save_weights(
            f"./weights/final_{metric}_tl.hdf5")


l2 = base()
l2.train('cosine', 'all')
