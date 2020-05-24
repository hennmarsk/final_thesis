import tensorflow.keras.applications as applications


def create_model(input_shape, t="resnet"):
    if t == "mobilenet":
        model = applications.MobileNetV2(
            weights=None, input_shape=input_shape, classes=512)
    elif t == "resnet":
        model = applications.ResNet50V2(
            weights=None, input_shape=input_shape, classes=512)
    return model
