import tensorflow as tf
import tensorflow_hub as hub
from configuration import config

BERT_MODEL_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
max_seq_length = config.summ_length

input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")

bert_layer = hub.KerasLayer(BERT_MODEL_URL, trainable=False)

_, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=sequence_output)
