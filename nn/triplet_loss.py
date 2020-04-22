import tensorflow as tf


def _cosine_pairwise_distance(y_pred):
    normalize_a = tf.nn.l2_normalize(y_pred, 1)
    normalize_b = tf.nn.l2_normalize(y_pred, 1)
    distance = 1 - tf.matmul(normalize_a, tf.transpose(normalize_b))
    return distance


def _euclidean_pairwise_distance(y_pred, squared=False):
    dot_product = tf.matmul(y_pred, tf.transpose(y_pred))
    square_norm = tf.linalg.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * \
        dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.maximum(distances, 0.0)
    if not squared:
        mask = tf.compat.v1.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)
    return distances


def _distance_mode(y_pred, metric, squared=False):
    if metric == 'cosine':
        return _cosine_pairwise_distance(y_pred)
    elif metric == 'euclid':
        return _euclidean_pairwise_distance(y_pred, squared=squared)


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


def loss(metric, squared=False):
    def instance(y_true, y_pred):
        if metric == 'cosine':
            margin = 0.01
        else:
            margin = 0.5
        pairwise_dist = _distance_mode(y_pred, metric)
        mask_anchor_positive = _get_anchor_positive_triplet_mask(y_true)
        mask_anchor_positive = tf.compat.v1.to_float(mask_anchor_positive)
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
        hardest_positive_dist = tf.reduce_max(
            anchor_positive_dist, axis=1, keepdims=True)
        tf.summary.scalar("hardest_positive_dist",
                          tf.reduce_mean(hardest_positive_dist))
        mask_anchor_negative = _get_anchor_negative_triplet_mask(y_true)
        mask_anchor_negative = tf.compat.v1.to_float(mask_anchor_negative)
        max_anchor_negative_dist = tf.reduce_max(
            pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + \
            max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist = tf.reduce_min(
            anchor_negative_dist, axis=1, keepdims=True)
        tf.summary.scalar("hardest_negative_dist",
                          tf.reduce_mean(hardest_negative_dist))
        triplet_loss = tf.maximum(
            hardest_positive_dist - hardest_negative_dist + margin, 0.0)
        triplet_loss = tf.reduce_mean(triplet_loss)
        return triplet_loss
    return instance
