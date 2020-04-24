import tensorflow.keras.applications as applications


def create_model(input_shape):
    model = applications.resnet_v2.ResNet50V2(
        weights=None, input_shape=input_shape, classes=128)
    return model
