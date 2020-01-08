import tensorflow as tf
import tensorflow_hub as hub
# from configuration import config


BERT_MODEL_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
BERT_en_MODEL_URL = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

vocab_of_BERT = hub.KerasLayer(BERT_MODEL_URL, trainable=False)

# max_seq_length = config.doc_length

# input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
#                                        name="input_word_ids")
# input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
#                                    name="input_mask")
# segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
#                                     name="segment_ids")

# bert_layer = hub.KerasLayer(BERT_MODEL_URL, trainable=False)

# pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

# b_model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])

class BertLayer(tf.keras.layers.Layer):
    """
    Custom Keras layer, integrating BERT from tf-hub
    """
    def __init__(self, url=BERT_en_MODEL_URL, d_embedding=768, n_fine_tune_layers=0, **kwargs):
        self.url = url
        self.n_fine_tune_layers = n_fine_tune_layers
        self.d_embedding = d_embedding
        #self.vocab_of_BERT = hub.KerasLayer(BERT_MODEL_URL, trainable=False)
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):   
        
        self.bert = hub.load(
            self.url,
            tags=[]
        )
        
        trainable_vars = self.bert.variables
        
        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "cls/" in var.name]
        
        # Select how many layers to fine tune
        #trainable_vars = trainable_vars[-self.n_fine_tune_layers :]
        
        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)
        
        # Removed when loading from checkpoint
        # Add non-trainable weights
#         for var in self.bert.variables:
#             if var not in self._trainable_weights:
#                 self._non_trainable_weights.append(var)
        
        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [tf.cast(x, dtype="int32") for x in inputs]
        
        input_ids, input_mask, segment_ids = inputs
        
        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        result = self.bert.signatures["tokens"](**bert_inputs)["sequence_output"]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_embedding)