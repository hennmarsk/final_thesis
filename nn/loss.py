import tensorflow.keras.backend as K


def _euclid(x, y):
    return K.sqrt(K.sum(K.square(K.maximum(x - y, K.epsilon())), axis=1))


def _cosine(x, y):
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return 1 - K.mean(K.maximum(x * y, K.epsilon()), axis=-1, keepdims=True)


def _distance(x, y, metric):
    if metric == 'euclid':
        return _euclid(x, y)
    else:
        return _cosine(x, y)


def triplet_loss(metric, batch_size):
    def instance(y_true, y_pred):
        if (metric == 'euclid'):
            margin = 0.5
        else:
            margin = 0.01
        sz = batch_size * 3
        anchor = y_pred[0:int(sz*1/3)]
        positive = y_pred[int(sz*1/3):int(sz*2/3)]
        negative = y_pred[int(sz*2/3):int(sz*3/3)]
        pos_dist = _distance(anchor, positive, metric)
        neg_dist = _distance(anchor, negative, metric)
        basic_loss = pos_dist - neg_dist + margin
        loss = K.maximum(basic_loss, 0.0)
        return loss
    return instance
