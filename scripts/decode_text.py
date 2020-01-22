#code adapted from https://github.com/ShenakhtPajouh/GPT-language-model-tf.keras/blob/master/utils.py#L126

def top_k_sampling(logits, k=25, temperature=0.8):
    'k must be greater than 0'

    values, _ = tf.nn.top_k(logits, k=k)
    min_value = tf.reduce_min(values)
    logits = tf.where(
        logits < min_value,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits)

    logits = logits / temperature

    sample = tf.multinomial(tf.expand_dims(logits, 0), num_samples=1, output_dtype=tf.int32)
    return tf.reduce_sum(sample)


def argmax(logits):
    return tf.argmax(logits)


def nucleus_sampling(logits, p=0.9):
    sorted_logits = tf.sort(logits, direction='DESCENDING')
    sorted_indices = tf.argsort(logits, direction='DESCENDING')
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits))
    t_sorted_indices_to_remove = cumulative_probs > p
    ''' Shift the indices to the right to keep also the first token above the threshold '''
    indices = tf.range(1, tf.shape(logits)[0], 1)
    sorted_indices_to_remove = tf.scatter_nd(tf.expand_dims(indices, 1), t_sorted_indices_to_remove[:-1], logits.shape)
    indices_to_remove = tf.boolean_mask(sorted_indices, sorted_indices_to_remove)
    t = tf.ones(tf.shape(indices_to_remove)[0], dtype=tf.bool)
    to_remove = tf.scatter_nd(tf.expand_dims(indices_to_remove, 1), t, logits.shape)
    logits = tf.where(
        to_remove,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits
    )

    sample = tf.multinomial(tf.expand_dims(logits, 0), num_samples=1, output_dtype=tf.int32)
    return tf.reduce_sum(sample)


def sampling(logits, temperature=0.8):
    logits = logits / temperature
    sample = tf.multinomial(tf.expand_dims(logits, 0), num_samples=1, output_dtype=tf.int32)
    return tf.reduce_sum(sample)