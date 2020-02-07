import tensorflow as tf
from transformers import TFBertModel
from tensorflow.keras.initializers import Constant
from transformer import create_masks, Decoder
from creates import log
from configuration import config


# Special Tokens
UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103

def tile_and_mask_diagonal(x, mask_with):
    """    
    Masks each word in the summary draft one by one with the [MASK] token
    At t-th time step the t-th word of input summary is
    masked, and the decoder predicts the refined word given other
    words of the summary.
    
    x :: (N, T)
    returrn :: (N, T-1, T)
    
    We do not mask the first and last postition (corresponding to [CLS]
    """

    N, T = tf.shape(x)[0], tf.shape(x)[1]

    first = tf.reshape(tf.tile(x[:, 0], [T-1]), [N, T-1, 1])
    
    x = x[:, 1:]
    T = T - 1
    
    masked = tf.reshape(tf.tile(x, [1, T]), [N, T, T])
    
    diag = tf.ones([N, T], dtype=masked.dtype) * mask_with
    masked = tf.linalg.set_diag(masked, diag)
    
    masked = tf.concat([first, masked], axis=2)
    
    masked = tf.reshape(masked, [N*T, T+1])
    
    return masked

def _embedding_from_bert():

  log.info("Extracting pretrained word embeddings weights from BERT")  
  vocab_of_BERT = TFBertModel.from_pretrained(config.pretrained_bert_model, trainable=False)
  embedding_matrix = vocab_of_BERT.get_weights()[0]
  log.info(f"Embedding matrix shape '{embedding_matrix.shape}'")
  return (embedding_matrix, vocab_of_BERT)

class AbstractiveSummarization(tf.keras.Model):
    """
    Pretraining-Based Natural Language Generation for Text Summarization 
    https://arxiv.org/pdf/1902.09243.pdf
    """
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, output_seq_len, rate=0.1):
        super(AbstractiveSummarization, self).__init__()
        
        self.output_seq_len = output_seq_len
        self.vocab_size = vocab_size
        embedding_matrix, self.bert_model = _embedding_from_bert()
        self.embedding = tf.keras.layers.Embedding(
                                                    vocab_size, d_model, trainable=False,
                                                    embeddings_initializer=Constant(embedding_matrix)
                                                   )
        
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, rate)
        self.d_model = d_model

    def draft_summary(self,
                      input_ids,
                      enc_output,
                      look_ahead_mask,
                      padding_mask,
                      target_ids,
                      training):
        # (batch_size, seq_len, d_bert)
        embeddings = self.embedding(target_ids) 
        # (batch_size, seq_len, vocab_len), (_)            
        draft_logits, draft_attention_dist = self.decoder(
                                                          input_ids,
                                                          embeddings, 
                                                          enc_output, 
                                                          training, 
                                                          look_ahead_mask, 
                                                          padding_mask
                                                          )
        # (batch_size, seq_len, vocab_len)
        return draft_logits, draft_attention_dist

    def refine_summary(self,
                       input_ids, 
                       enc_output, 
                       target, 
                       padding_mask, 
                       training):

        N = tf.shape(enc_output)[0]
        T = self.output_seq_len
        # since we are using teacher forcing we do not need an autoregressice mechanism here
        # (batch_size x (seq_len - 1), seq_len) 
        dec_inp_ids = tile_and_mask_diagonal(target, mask_with=MASK_ID)
        # (batch_size x (seq_len - 1), seq_len, d_bert) 
        enc_output = tf.tile(enc_output, [T-1, 1, 1])
        # (batch_size x (seq_len - 1), 1, 1, seq_len) 
        padding_mask = tf.tile(padding_mask, [T-1, 1, 1, 1])
        # (batch_size x (seq_len - 1), seq_len, d_bert)
        context_vectors = self.bert_model(dec_inp_ids)[0]

        # (batch_size x (seq_len - 1), seq_len, vocab_len), (_)
        dec_outputs, refine_attention_dist = self.decoder(
                                                           tf.tile(input_ids, [T-1, 1]),
                                                           context_vectors,
                                                           enc_output,
                                                           training,
                                                           look_ahead_mask=None,
                                                           padding_mask=padding_mask
                                                         )
        # (batch_size x (seq_len - 1), seq_len - 1, vocab_len)
        dec_outputs = dec_outputs[:, 1:, :]
        # (batch_size x (seq_len - 1), (seq_len - 1))
        diag = tf.linalg.set_diag(tf.zeros([T-1, T-1]), tf.ones([T-1]))
        diag = tf.tile(diag, [N, 1])
        
        where = tf.not_equal(diag, 0)
        indices = tf.where(where)
        
        # (batch_size x (seq_len - 1), vocab_len)
        dec_outputs = tf.gather_nd(dec_outputs, indices)
        
        # (batch_size, seq_len - 1, vocab_len)
        dec_outputs = tf.reshape(dec_outputs, [N, T-1, -1])
        # (batch_size, seq_len, vocab_len)
        refine_logits = tf.concat(
                           [tf.tile(tf.expand_dims(tf.one_hot([CLS_ID], self.vocab_size), axis=0), [N, 1, 1]), dec_outputs],
                           axis=1
                           )


        # (batch_size, seq_len, vocab_len)
        return refine_logits, refine_attention_dist

    def call(self, 
             input_ids, 
             target_ids, 
             training):

           # (batch_size, 1, 1, seq_len), (batch_size, 1, 1, seq_len)
        _, combined_mask, dec_padding_mask = create_masks(input_ids, target_ids[:, :-1])

        # (batch_size, seq_len, d_bert)
        enc_output = self.bert_model(input_ids)[0]

        # (batch_size, seq_len, vocab_len), _
        draft_logits, draft_attention_dist = self.draft_summary(
                                                                input_ids,
                                                                enc_output=enc_output,
                                                                look_ahead_mask=combined_mask,
                                                                padding_mask=dec_padding_mask,
                                                                target_ids=target_ids[:, :-1],
                                                                training=True
                                                               )

        # (batch_size, seq_len, vocab_len), _
        refine_logits, refine_attention_dist = self.refine_summary(
                                                                  input_ids,
                                                                  enc_output=enc_output,
                                                                  target=target_ids[:, :-1],            
                                                                  padding_mask=dec_padding_mask,
                                                                  training=True
                                                                  )
              
        return draft_logits, draft_attention_dist, refine_logits, refine_attention_dist
