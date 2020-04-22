import tensorflow.keras.applications as applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


def create_model(input_shape):
    inp = Input(shape=input_shape)
    out = applications.ResNet50(
        weights=None, input_shape=input_shape, classes=128)(inp)
    model = Model(inputs=inp, outputs=out)
    return model
