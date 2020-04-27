import tensorflow as tf
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
            margin = 0.2
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


def _pairwise_distances(y_pred, squared=False):
    dot_product = tf.matmul(y_pred, tf.transpose(y_pred))
    square_norm = tf.linalg.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * \
        dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.maximum(distances, 1e-16)
    if not squared:
        distances = tf.sqrt(distances)
    return distances


def _get_anchor_positive_triplet_mask(y_true):
    indices_equal = tf.cast(tf.eye(tf.shape(y_true)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    y_true_equal = tf.equal(tf.expand_dims(y_true, 0),
                            tf.expand_dims(y_true, 1))
    mask = tf.logical_and(indices_not_equal, y_true_equal)
    return mask


def _get_anchor_negative_triplet_mask(y_true):
    y_true_equal = tf.equal(tf.expand_dims(y_true, 0),
                            tf.expand_dims(y_true, 1))
    mask = tf.logical_not(y_true_equal)
    return mask


def _get_triplet_mask(y_true):
    indices_equal = tf.cast(tf.eye(tf.shape(y_true)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
    distinct_indices = tf.logical_and(tf.logical_and(
        i_not_equal_j, i_not_equal_k), j_not_equal_k)
    label_equal = tf.equal(tf.expand_dims(y_true, 0),
                           tf.expand_dims(y_true, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)
    valid_y_true = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))
    mask = tf.logical_and(distinct_indices, valid_y_true)
    return mask


def batch_all(margin, squared=False):
    def instance(y_true, y_pred):
        pairwise_dist = _pairwise_distances(y_pred, squared=squared)
        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(
            anchor_positive_dist.shape)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(
            anchor_negative_dist.shape)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
        mask = _get_triplet_mask(y_true)
        mask = tf.cast(mask, tf.float32)
        triplet_loss = tf.multiply(mask, triplet_loss)
        triplet_loss = tf.maximum(triplet_loss, 0.0)
        valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float)
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_positive_triplets = num_positive_triplets / \
            (num_valid_triplets + 1e-16)
        triplet_loss = tf.reduce_sum(triplet_loss) / \
            (num_positive_triplets + 1e-16)
        return triplet_loss, fraction_positive_triplets
    return instance


def batch_hard(margin=0.5, squared=False):
    def instance(y_true, y_pred):
        pairwise_dist = _pairwise_distances(y_pred, squared=squared)
        y_true = tf.squeeze(y_true, axis=-1)
        mask_anchor_positive = _get_anchor_positive_triplet_mask(y_true)
        mask_anchor_positive = tf.cast(mask_anchor_positive, tf.float32)
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
        hardest_positive_dist = tf.reduce_max(
            anchor_positive_dist, axis=1, keepdims=True)
        mask_anchor_negative = _get_anchor_negative_triplet_mask(y_true)
        mask_anchor_negative = tf.cast(mask_anchor_negative, tf.float32)
        max_anchor_negative_dist = tf.reduce_max(
            pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + \
            max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist = tf.reduce_min(
            anchor_negative_dist, axis=1, keepdims=True)
        triplet_loss = tf.maximum(
            hardest_positive_dist - hardest_negative_dist + margin, 0.0)
        triplet_loss = tf.reduce_mean(triplet_loss)
        return triplet_loss
    return instance
