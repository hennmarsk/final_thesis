import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
import tensorflow.keras.applications as applications
import numpy as np
import os
import cv2

class MyModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(MyModel, self).__init__(name='mymodel')
        self.mobile_net = applications.MobileNetV2(weights=None, input_shape=input_shape, classes=128, include_top=False)

    @tf.function
    def call(self, input_tensor, is_train=True):
        anchor = input_tensor[:, 0]
        positive = input_tensor[:, 1]
        negative = input_tensor[:, 2]
        processed_a = self.mobile_net(anchor)
        processed_p = self.mobile_net(positive)
        processed_n = self.mobile_net(negative)
        return processed_a, processed_p, processed_n

def random_indice(path):
    size = len(os.listdir(path))
    a = np.random.permutation(size)
    return a

def preprocess(image):
    return applications.mobilenet.preprocess_input(image)

def generator(word='train'):
    a = random_indice(f"../data/{word}")
    for positives in a:
        a_positive = random_indice(f"../data/{word}/{positives}")
        for anchor in a_positive:
            img_a = cv2.imread(f"../data/{word}/{positives}/{anchor}")
            for positive in a_positive:
                if (anchor != positive):
                    img_p = cv2.imread(f"../data/{word}/{positives}/{positive}")
                    for negatives in a:
                        if (negatives != positives):
                            a_negatives = random_indice(f"../data/{word}/{negatives}")
                            for negative in a_negatives:
                                img_n = cv2.imread(f"../data/{word}/{negatives}/{negative}")
                                yield [(img_a)/255, (img_p)/255, (img_n)/255]

def euclidean_distance(x, y):
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    # return tf.math.sqrt(tf.math.maximum(sum_square, 1e-7))
    return sum_square

def cosine_distance(x, y):
    x = tf.math.l2_normalize(x, axis=-1)
    y = tf.math.l2_normalize(y, axis=-1)
    return -tf.math.reduce_mean(x * y, axis=-1, keepdims=True)

def l1_distance(x, y):
    return tf.math.reduce_sum(tf.math.abs(x - y), axis=1, keepdims=True)

def dist_mode(x, y, mode):
    if mode == 'euclidean':
        return euclidean_distance(x, y)
    elif mode == 'cosine':
        return cosine_distance(x, y)
    elif mode == 'l1':
        return l1_distance(x, y)

def triplet_loss(y_pred, mode='euclidean', margin=0.6):
    anchor = y_pred[0]
    positive = y_pred[1]
    negative = y_pred[2]
    pos_dist = dist_mode(positive, anchor, mode)
    neg_dist = dist_mode(negative, anchor, mode)
    basic_loss = pos_dist - neg_dist + margin
    return tf.math.maximum(basic_loss, 0.0)

def training(model, input_data, optimizer, mode='euclidean'):
    # anchor, positive, negative = input_data
    with tf.GradientTape() as tape:
        y = model(input_data)
        loss = triplet_loss(y)
        gradient = tape.gradient(loss, model.trainable_variables)
        print(gradient)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    return loss

input_shape = (218, 178, 3, )
batch_size=64
epoch = 1000
model = MyModel(input_shape)
optimizer = tf.keras.optimizers.Adam()

dataset = tf.data.Dataset.from_generator(generator, (tf.float32))
dataset = dataset.batch(batch_size)
iterator = dataset.make_one_shot_iterator()

for i in range(epoch):    
    print('Epoch:', i)    
    training_input = iterator.get_next()
    loss = training(model=model, input_data=training_input, optimizer=optimizer)
    print("Train loss:", np.mean(loss))
    # break