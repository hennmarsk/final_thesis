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
        self.mobile_net = applications.MobileNetV2(weights=None, input_shape=input_shape, classes=128)

    @tf.function
    def call(self, input_tensor, is_train=True):
        anchor = input_tensor[:, 0, :, :, :]
        positive = input_tensor[:, 1, :, :, :]
        negative = input_tensor[:, 2, :, :, :]
        processed_a = self.mobile_net(anchor)
        processed_p = self.mobile_net(positive)
        processed_n = self.mobile_net(negative)
        return [processed_a, processed_p, processed_n]

input_shape = 