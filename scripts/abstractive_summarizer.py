import tensorflow as tf
import tensorflow_hub as hub
from create_tokenizer import BERT_MODEL_URL
from transformer import create_masks, Decoder, Pointer_Generator
from creates import log

BERT_MODEL_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"

# def _embedding_from_bert():
#     """
#     Extract the preratined word embeddings from a BERT model
#     Returns a numpy matrix with the embeddings
#     """
#     log.info("Extracting pretrained word embeddings weights from BERT")
    
#     bert_layer = hub.KerasLayer(BERT_MODEL_URL, trainable=False)
#     embedding_matrix = bert_layer.get_weights()[0]   
                        
#     log.info(f"Embedding matrix shape '{embedding_matrix.shape}'")
#     return embedding_matrix

class AbstractiveSummarization(tf.keras.Model):
    """
    Pretraining-Based Natural Language Generation for Text Summarization 
    https://arxiv.org/pdf/1902.09243.pdf
    """
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, input_seq_len, output_seq_len, rate=0.1):
        super(AbstractiveSummarization, self).__init__()
        
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        
        self.vocab_size = vocab_size
    
        self.bert = hub.KerasLayer(BERT_MODEL_URL, trainable=False)
        
        embedding_matrix = self.bert.get_weights()[0]
        
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, d_model, trainable=False,
            embeddings_initializer=Constant(embedding_matrix)
        )
        
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, rate)
        self.d_model = d_model

        if config.copy_gen:
            self.pointer_generator   = Pointer_Generator()
                
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inp, tar, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask, training):
        # (batch_size, seq_len) x3
        input_ids, input_mask, input_segment_ids = inp
        
        # (batch_size, seq_len + 1) x3
        target_ids, target_mask, target_segment_ids = tar
        target_ids = target_ids[:, :-1]

        # (batch_size, 1, 1, seq_len), (_), (batch_size, 1, 1, seq_len)
        _, combined_mask, dec_padding_mask = create_masks(input_ids, target_ids)

        # (batch_size, seq_len, d_bert)
        enc_output = self.bert((input_ids, input_mask, input_segment_ids))[1] # index 1 returns the sequence output
        
        # (batch_size, seq_len, d_bert)
        embeddings = self.embedding(target_ids) 

        # (batch_size, seq_len, d_bert), (_)            
        dec_output, attention_dist = self.decoder(embeddings, enc_output, training, combined_mask, dec_padding_mask)

        # (batch_size, seq_len, vocab_len)
        logits = self.final_layer(dec_output)

        if config.add_stage_2:
            N = tf.shape(enc_output)[0]
            T = self.output_seq_len
            # since we are using teacher forcing we do not need an autoregressice mechanism here
            # (batch_size x (seq_len - 1), seq_len) 
            dec_inp_ids = tile_and_mask_diagonal(target_ids, mask_with=MASK_ID)
            # (batch_size x (seq_len - 1), seq_len) 
            dec_inp_mask = tf.tile(target_mask[:, :-1], [T-1, 1])
            # (batch_size x (seq_len - 1), seq_len) 
            dec_inp_segment_ids = tf.tile(target_segment_ids[:, :-1], [T-1, 1])
            # (batch_size x (seq_len - 1), seq_len, d_bert) 
            enc_output = tf.tile(enc_output, [T-1, 1, 1])
            # (batch_size x (seq_len - 1), 1, 1, seq_len) 
            padding_mask = tf.tile(dec_padding_mask, [T-1, 1, 1, 1])
            # (batch_size x (seq_len - 1), seq_len, d_bert)
            context_vectors = self.bert((dec_inp_ids, dec_inp_mask, dec_inp_segment_ids))
            # (batch_size x (seq_len - 1), seq_len, d_bert), (_)
            dec_outputs, attention_dists = self.decoder(
                                                        context_vectors,
                                                        enc_output,
                                                        training,
                                                        look_ahead_mask=None,
                                                        padding_mask=padding_mask
                                                    )
            # (batch_size x (seq_len - 1), seq_len - 1, d_bert)
            dec_outputs = dec_outputs[:, 1:, :]
            # (batch_size x (seq_len - 1), (seq_len - 1))
            diag = tf.linalg.set_diag(tf.zeros([T-1, T-1]), tf.ones([T-1]))
            diag = tf.tile(diag, [N, 1])
            
            where = tf.not_equal(diag, 0)
            indices = tf.where(where)
            
            # (batch_size x (seq_len - 1), d_bert)
            dec_outputs = tf.gather_nd(dec_outputs, indices)
            
            # (batch_size, seq_len - 1, d_bert)
            dec_outputs = tf.reshape(dec_outputs, [N, T-1, -1])
            # (batch_size, seq_len, d_bert)
            dec_outputs = tf.concat(
                               [tf.tile(tf.expand_dims(tf.one_hot([CLS_ID], self.d_model), axis=0), [N, 1, 1]), dec_outputs],
                               axis=1
                               )


            # (batch_size, seq_len - 1, vocab_len)
            logits = self.final_layer(dec_outputs)

            # logits = tf.concat(
            #                    [tf.tile(tf.expand_dims(tf.one_hot([CLS_ID], self.vocab_size), axis=0), [N, 1, 1]), logits],
            #                    axis=1
            #                    )

            if config.copy_gen: 
                logits = self.pointer_generator(
                                                dec_output, 
                                                logits, 
                                                attention_dist, 
                                                input_ids, 
                                                tf.shape(input_ids)[1], 
                                                tf.shape(embeddings)[1], 
                                                training=training
                                                )
        return logits, attention_dist, dec_outputs