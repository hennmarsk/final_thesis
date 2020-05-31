import tensorflow as tf


def _pairwise_euclid(y_pred, squared=False):
    dot_product = tf.matmul(y_pred, tf.transpose(y_pred))
    square_norm = tf.linalg.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * \
        dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.maximum(distances, 1e-16)
    if not squared:
        distances = tf.sqrt(distances)
    return distances


def _pairwise_cosine(y_pred):
    normalize_a = tf.nn.l2_normalize(y_pred, 1)
    normalize_b = tf.nn.l2_normalize(tf.transpose(y_pred), 1)
    distance = 1 - tf.matmul(normalize_a, normalize_b)
    return distance


def _distance(metric, y_pred, squared=False):
    if metric == 'euclid':
        return _pairwise_euclid(y_pred, squared)
    else:
        return _pairwise_cosine(y_pred)


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


def batch_mode(metric, margin=0.5, squared=False, mode="semi"):
    def instance(y_true, y_pred):
        pairwise_dist = _distance(metric, y_pred, squared=squared)
        y_true = tf.squeeze(y_true, axis=-1)
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
        if mode == 'semi':
            mask1 = tf.logical_and(tf.greater(
                triplet_loss, 0.0), tf.less(triplet_loss, margin))
        else:
            mask1 = tf.greater(triplet_loss, 0.0)
        valid_triplets = tf.cast(mask1, tf.float32)
        triplet_loss = tf.multiply(valid_triplets, triplet_loss)
        num_valid_triplets = tf.reduce_sum(valid_triplets)
        triplet_loss = tf.reduce_sum(
            triplet_loss) / tf.maximum(num_valid_triplets, 1e-16)
        return triplet_loss
    return instance


def pos_all(metric, margin=0.5, squared=False):
    def p_a(y_true, y_pred):
        pairwise_dist = _distance(metric, y_pred, squared=squared)
        y_true = tf.squeeze(y_true, axis=-1)
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
        num_triplets = tf.reduce_sum(mask)
        mask2 = tf.cast(tf.greater(triplet_loss, 0.0), tf.float32)
        num_positive_triplets = tf.reduce_sum(mask2)
        fraction2 = num_positive_triplets / num_triplets
        return fraction2
    return p_a


def pairwise_loss(metric, margin=0.5, squared=False):
    def instance(y_true, y_pred):
        pairwise_dist = _distance(metric, y_pred, squared=squared)
        y_true = tf.squeeze(y_true, axis=-1)
        mask_anchor_positive = _get_anchor_positive_triplet_mask(y_true)
        mask_anchor_positive = tf.cast(mask_anchor_positive, tf.float32)
        mask_anchor_negative = _get_anchor_negative_triplet_mask(y_true)
        mask_anchor_negative = tf.cast(mask_anchor_negative, tf.float32)
        _negative = tf.maximum(margin - pairwise_dist, 0.0)
        _negative = _negative * mask_anchor_negative
        _mask_negative = tf.cast(tf.greater(_negative, 0.0), tf.float32)
        loss_negative = tf.reduce_sum(
            _negative) / tf.maximum(tf.reduce_sum(_mask_negative), 1e-16)
        _positive = pairwise_dist * mask_anchor_positive
        _mask_positive = tf.cast(tf.greater(_positive, margin), tf.float32)
        _positive = _positive * _mask_positive
        loss_positive = tf.reduce_sum(
            _positive) / tf.maximum(tf.reduce_sum(_mask_positive), 1e-16)
        final_loss = (loss_negative + loss_positive) / 2
        return final_loss
    return instance


def pos_neg_all(metric, margin=0.5, squared=False):
    def p_n_a(y_true, y_pred):
        pairwise_dist = _distance(metric, y_pred, squared=squared)
        y_true = tf.squeeze(y_true, axis=-1)
        mask_anchor_positive = _get_anchor_positive_triplet_mask(y_true)
        mask_anchor_positive = tf.cast(mask_anchor_positive, tf.float32)
        mask_anchor_negative = _get_anchor_negative_triplet_mask(y_true)
        mask_anchor_negative = tf.cast(mask_anchor_negative, tf.float32)
        _positive = pairwise_dist * mask_anchor_positive
        num_pair_pos = tf.reduce_sum(mask_anchor_positive)
        _mask_positive = tf.cast(
            tf.greater_equal(_positive, margin), tf.float32)
        positive_pair = tf.reduce_sum(_mask_positive)
        _negative = tf.maximum(margin - pairwise_dist, 0.0)
        _negative = _negative * mask_anchor_negative
        _mask_negative = tf.cast(tf.greater(_negative, 0.0), tf.float32)
        num_pair_neg = tf.reduce_sum(mask_anchor_negative)
        negative_pair = tf.reduce_sum(_mask_negative)
        return (positive_pair / tf.maximum(num_pair_pos, 1e-16) +
                negative_pair / tf.maximum(num_pair_neg, 1e-16)) / 2
    return p_n_a
