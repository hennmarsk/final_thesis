import tensorflow.keras.applications as applications


def create_model(input_shape):
    model = applications.MobileNetV2(
        weights=None, input_shape=input_shape, classes=128)
    return model
