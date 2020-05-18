import tensorflow.keras.applications as applications


def create_model(input_shape):
    model = applications.ResNet50V2(
        weights=None, input_shape=input_shape, classes=512)
    return model
