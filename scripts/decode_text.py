#code adapted from a) https://github.com/ShenakhtPajouh/GPT-language-model-tf.keras/blob/master/utils.py
#                  b)https://github.com/raufer/bert-summarization/tree/master/models
import tensorflow as tf
tf.random.set_seed(100)

import numpy as np
tf.random.set_seed(100)
import time
from hyper_parameters import h_parms
from configuration import config
from metrics import optimizer, loss_function, label_smoothing, get_loss_and_accuracy, tf_write_summary, monitor_run
from input_path import file_path
from creates import log, train_summary_writer, valid_summary_writer
from create_tokenizer import tokenizer, model
from local_tf_ops import *
from beam_search import beam_search
from transformer import create_masks
from tqdm import tqdm

UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103

def with_column(x, i, column):
    """
    Given a tensor `x`, change its i-th column with `column`
    x :: (N, T)
    return :: (N, T)
    """

    N, T = tf.shape(x)[0], tf.shape(x)[1]

    left = x[:, :i]
    right = x[:, i+1:]
        
    return tf.concat([left, column, right], axis=1)

def mask_timestamp(x, i, mask_with):
    """
    Masks each word in the summary draft one by one with the [MASK] token
    At t-th time step the t-th word of input summary is
    masked, and the decoder predicts the refined word given other
    words of the summary.
    
    x :: (N, T)
    return :: (N, T)
    """

    N, T = tf.shape(x)[0], tf.shape(x)[1]

    left = x[:, :i]
    right = x[:, i+1:]
    
    mask = tf.ones([N, 1], dtype=x.dtype) * mask_with
    
    masked = tf.concat([left, mask, right], axis=1)

    return masked

def create_padding_mask(seq):
    """
    Mask all the pad tokens in the batch of sequence.
    It ensures that the model does not treat padding as the input.
    The mask indicates where pad value 0 is present:
    it outputs a 1 at those locations, and a 0 otherwise.    
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def top_k_sampling(logits, k=25, temperature=0.9):
    'k must be greater than 0'
    logits = tf.squeeze(logits, 0)
    values, _ = tf.nn.top_k(logits, k=k)
    min_value = tf.reduce_min(values)
    logits = tf.where(
        logits < min_value,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits)

    logits = logits / temperature
    sample = tf.random.categorical(logits, num_samples=1, dtype=tf.int32, seed=1)
    return sample

def argmax(logits):
    return tf.argmax(logits, axis=-1)

def nucleus_sampling(logits, p=0.9):
    logits = tf.squeeze(logits, 0)
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
    sample = tf.random.categorical(logits, num_samples=1, dtype=tf.int32, seed=1)
    return sample

def sampling(logits, temperature=0.8):
    logits = tf.squeeze(logits, 0)
    logits = logits / temperature
    sample = tf.random.categorical(logits, num_samples=1, dtype=tf.int32, seed=1)
    return sample
	
def draft_summary_greedy(inp, enc_output, look_ahead_mask, padding_mask, training=False):
    """
    Inference call, builds a draft summary auto-regressively
    """

    N = tf.shape(enc_output)[0]
    T = tf.shape(enc_output)[1]

    # (batch_size, 1)
    dec_input = tf.ones([N, 1], dtype=tf.int32) * CLS_ID
    summary, dec_outputs, dec_logits, attention_dists = [], [], [], []
    summary += [dec_input]
    for i in tqdm(range(0, config.summ_length)):
        _, _, dec_padding_mask = create_masks(inp, dec_input)
        # (batch_size, i+1, d_bert)
        embeddings = model.embedding(dec_input)    

        # (batch_size, i+1, vocab), (_)            
        dec_output, dec_logits_i, attention_dist = model.decoder(
                                                    inp, 
                                                    embeddings, 
                                                    enc_output, 
                                                    training, 
                                                    look_ahead_mask, 
                                                    padding_mask
                                                   )
        

        # (batch_size, 1, vocab)
        dec_output_i = dec_output[:, -1: ,:]
        preds = tf.cast(tf.argmax(dec_output_i, axis=-1), tf.int32)
        dec_outputs += [dec_output_i]
        dec_logits_i = dec_logits_i[:, -1:, :]
        dec_logits += [dec_logits_i]
        summary += [preds]
        dec_input = with_column(dec_input, i+1, preds)
        attention_dist = tf.concat(attention_dist, axis=2)
    dec_outputs = tf.concat(dec_outputs, axis=1)
    dec_logits = tf.concat(dec_logits, axis=1)
    summary = tf.concat(summary, axis=1)  
    if config.copy_gen: 
      predictions = model.decoder.pointer_generator(
                                        dec_logits,
                                        dec_outputs, 
                                        attention_dist, 
                                        inp, 
                                        tf.shape(inp)[-1], 
                                        tf.shape(dec_outputs)[1], 
                                        training=training
                                        )
      summary = tf.cast(tf.argmax(predictions, axis=-1), dtype=tf.int32)
      print(f' pointer_gen_summary_shape {tf.shape(summary)}')
    # (batch_size, seq_len, vocab_len), (batch_size, seq_len), (_)
    return tf.squeeze(summary,axis=0) , attention_dist


def refined_summary_greedy(inp, enc_output, draft_summary, padding_mask, training=False):
        """
        Inference call, builds a refined summary
        
        It first masks each word in the summary draft one by one,
        then feeds the draft to BERT to generate context vectors.
        """
        
        #logging.info("Building: 'Greedy Refined Summary'")
        refined_summary = tf.expand_dims(draft_summary,0)
        
        dec_outputs = []
        dec_logits = []
        for i in tqdm(range(1, config.summ_length)):
            
            # (batch_size, seq_len)
            refined_summary_ = mask_timestamp(refined_summary, i, MASK_ID)
            
            # (batch_size, seq_len, d_bert)
            context_vectors = model.bert_model(refined_summary_)[0]
            
            # (batch_size, seq_len, d_bert), (_)
            dec_output, dec_logits_i, attention_dist = model.decoder(
                                                        inp,
                                                        context_vectors,
                                                        enc_output,
                                                        training=training,
                                                        look_ahead_mask=None,
                                                        padding_mask=padding_mask
                                                      )
            
            # (batch_size, 1, vocab_len)
            dec_output_i = dec_output[:, i:i+1 ,:]
            preds = tf.cast(tf.argmax(dec_output_i, axis=-1), tf.int32)
            dec_outputs += [dec_output_i]
            dec_logits_i = dec_logits_i[:, i:i+1, :]
            dec_logits += [dec_logits_i]
            #dec_logits_i = tf.concat(dec_logits_i, axis=1)
            #dec_logits_i = tf.concat(dec_logits_i, axis=1)
            refined_summary = with_column(refined_summary, i, preds)
            attention_dist = tf.concat(attention_dist, axis=2)
        dec_outputs = tf.concat(dec_outputs, axis=1)
        dec_logits = tf.concat(dec_logits, axis=1)
        print(f'dec_logits {tf.shape(dec_logits)}')
        print(f'dec_outputs {tf.shape(dec_outputs)}')
        print(f'embeddings {context_vectors.shape}')
        print(f'attention_dist {tf.shape(attention_dist)}')
        print(f' decoder_refined_summary_shape {tf.shape(refined_summary)}')
        if config.copy_gen: 
          predictions = model.decoder.pointer_generator(
                                            dec_logits,
                                            dec_outputs, 
                                            attention_dist[:, :, :-1, :], 
                                            inp, 
                                            tf.shape(inp)[-1], 
                                            tf.shape(dec_outputs)[1], 
                                            training=training
                                            )
          refined_summary = tf.cast(tf.argmax(predictions, axis=-1), dtype=tf.int32)
        # (batch_size, seq_len, vocab_len), (batch_size, seq_len), (_)        
        return tf.squeeze(refined_summary, axis=0), attention_dist

def predict_greedy(inp):
  
  dec_padding_mask = create_padding_mask(inp)
  
  # (batch_size, seq_len, d_bert)
  enc_output = model.bert_model(inp)[0]
  # (batch_size, seq_len, vocab_len), (_)
  preds_draft_summary, draft_attention_dist = draft_summary_greedy( 
                                                                    inp,
                                                                    enc_output=enc_output,
                                                                    look_ahead_mask=None,
                                                                    padding_mask=dec_padding_mask
                                                                  )
  # (batch_size, seq_len, vocab_len), ()
  preds_refined_summary, refined_attention_dist = refined_summary_greedy(
                                                                        inp,
                                                                        enc_output=enc_output,
                                                                        padding_mask=dec_padding_mask,
                                                                        draft_summary=preds_draft_summary
                                                                        )

  return preds_draft_summary, draft_attention_dist, preds_refined_summary, refined_attention_dist


ckpt = tf.train.Checkpoint(
                           model=model,
                           optimizer=optimizer
                          )

''' 
Set the latest checkpoint and run the below piece of code for inference. 

ckpt.restore('/content/drive/My Drive/Text_summarization/BERT_text_summarisation/cnn_checkpoints/ckpt-35')

ip_ids = tokenizer.encode('Your summary sentence')
preds_draft_summary, draft_attention_dist, preds_refined_summary, _ = predict(ip_ids)
preds_refined_summary = (tokenizer.decode([i for i in preds_refined_summary if i not in [CLS_ID, SEP_ID, 0]]))
print(f'the predicted_refined_greedy auto regressive --> {preds_refined_summary if preds_refined_summary else "EMPTY"}')

'''
