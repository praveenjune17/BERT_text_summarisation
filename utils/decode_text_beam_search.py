#code adapted from a) https://github.com/ShenakhtPajouh/GPT-language-model-tf.keras/blob/master/utils.py
#                  b)https://github.com/raufer/bert-summarization/tree/master/models
import tensorflow as tf
tf.random.set_seed(100)
import numpy as np
import time
import tensorflow_addons as tfa
from hyper_parameters import h_parms
from configuration import config
from input_path import file_path
from creates import log, train_summary_writer, valid_summary_writer
from create_tokenizer_inference import tokenizer, model
from local_tf_ops import *
from beam_search import beam_search
from transformer import create_masks
#from tqdm import tqdm

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

def sampling(logits):
    logits = tf.squeeze(logits, 0)
    sample = tf.random.categorical(logits, num_samples=1, dtype=tf.int32, seed=1)
    return sample

def top_k_sampling(logits, k=25):
    'k must be greater than 0'
    values, _ = tf.nn.top_k(logits, k=k)
    min_value = tf.reduce_min(values)
    logits = tf.where(
        logits < min_value,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits)
    logits = tf.reshape(logits, (1, -1))
    sample = tf.random.categorical(logits, num_samples=1, dtype=tf.int32, seed=1)
    return sample
  
def nucleus_sampling(logits, p=0.9):
    sorted_logits = tf.sort(logits, direction='DESCENDING')
    sorted_indices = tf.argsort(logits, direction='DESCENDING')
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits))
    t_sorted_indices_to_remove = cumulative_probs > p
    ''' Shift the indices to the right to keep also the first token above the threshold '''
    indices = tf.range(1, tf.shape(logits)[0], 1)
    sorted_indices_to_remove = tf.scatter_nd(
                                             tf.expand_dims(indices, 1), 
                                             t_sorted_indices_to_remove[:-1], 
                                             logits.shape
                                             )
    indices_to_remove = tf.boolean_mask(sorted_indices, sorted_indices_to_remove)
    t = tf.ones(tf.shape(indices_to_remove)[0], dtype=tf.bool)
    to_remove = tf.scatter_nd(tf.expand_dims(indices_to_remove, 1), t, logits.shape)
    logits = tf.where(
        to_remove,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits
    )
    logits = tf.reshape(logits, (1, -1))
    sample = tf.random.categorical(logits, num_samples=1, dtype=tf.int32, seed=1)
    return sample
    
def draft_summary_sampling(
                           inp, 
                           enc_output, 
                           look_ahead_mask, 
                           padding_mask, 
                           sampling_type='greedy', 
                           temperature=0.9, 
                           p=0.9, 
                           k=25, 
                           training=False
                           ):
    """
    Inference call, builds a draft summary auto-regressively
    """
    log.info(f"Building: 'Draft {sampling_type} decoder'")
    N = tf.shape(enc_output)[0]
    T = tf.shape(enc_output)[1]

    # (batch_size, 1)
    dec_input = tf.ones([N, 1], dtype=tf.int32) * CLS_ID
    summary, dec_outputs, dec_logits, attention_dists = [], [], [], []
    summary += [dec_input]
    for i in (range(0, config.summ_length)):
        _, _, dec_padding_mask = create_masks(inp, dec_input)
        # (batch_size, i+1, d_bert)
        embeddings = model.embedding(dec_input)    

        # (batch_size, i+1, vocab), (_)            
        dec_output, dec_logits_i, attention_dist = model.decoder(
                                                                embeddings, 
                                                                enc_output, 
                                                                training, 
                                                                look_ahead_mask, 
                                                                padding_mask
                                                               )

        if config.copy_gen:
          dec_output = model.decoder.pointer_generator(
                                                        dec_logits_i, 
                                                        dec_output,
                                                        attention_dist,
                                                        inp,
                                                        tf.shape(inp)[1], 
                                                        tf.shape(dec_output)[1], 
                                                        training=False,
                                                       )
        

        # (batch_size, 1, vocab)
        dec_output_i = dec_output[:, -1: ,:]
        if sampling_type == 'nucleus':
          preds = tf.cast(nucleus_sampling((tf.squeeze(dec_output_i)/ temperature), p=p), tf.int32)
        elif sampling_type == 'topk':
          preds = tf.cast(top_k_sampling((tf.squeeze(dec_output_i)/ temperature), k=k), tf.int32)
        elif sampling_type == 'random_sampling':
          preds = tf.cast(sampling(tf.squeeze(dec_output_i)/ temperature), tf.int32)
        else:
          preds = tf.cast(tf.argmax(dec_output_i, axis=-1), tf.int32)
        dec_outputs += [dec_output_i]
        dec_logits_i = dec_logits_i[:, -1:, :]
        dec_logits += [dec_logits_i]
        summary += [preds]
        dec_input = with_column(dec_input, i+1, preds)
        #attention_dist = tf.concat(attention_dist, axis=2)
    summary = tf.concat(summary, axis=1)  
    # (batch_size, seq_len, vocab_len), (batch_size, seq_len), (_)
    return summary, attention_dist

def draft_summary_beam_search(input_ids, enc_output, dec_padding_mask, beam_size):

    log.info(f"Building: 'Draft beam search decoder'")
    input_ids = tfa.seq2seq.tile_batch(input_ids, multiplier=beam_size)
    enc_output = tfa.seq2seq.tile_batch(enc_output, multiplier=beam_size)
    dec_padding_mask = tfa.seq2seq.tile_batch(dec_padding_mask, multiplier=beam_size)
    #print(f'output_before {tf.shape(output)}')
    def beam_search_decoder(output):
      # (batch_size, seq_len, d_bert)    
      embeddings = model.embedding(output)
      predictions, dec_op, attention_weights = model.decoder(
                                                            embeddings, 
                                                            enc_output, 
                                                            False, 
                                                            None, 
                                                            dec_padding_mask
                                                            )
      if config.copy_gen:
        predictions = model.decoder.pointer_generator(
                                                      dec_op[:, -1:, :], 
                                                      predictions[:, -1:, :],
                                                      attention_weights[:, :, -1:, :],
                                                      input_ids,
                                                      tf.shape(input_ids)[1], 
                                                      tf.shape(predictions[:, -1:, :])[1], 
                                                      training=False,
                                                     )
      # (batch_size, 1, target_vocab_size)
      return (predictions[:,-1:,:])
    return beam_search(
                        beam_search_decoder, 
                        [CLS_ID] * h_parms.batch_size, 
                        beam_size, 
                        config.summ_length, 
                        config.input_vocab_size, 
                        h_parms.length_penalty, 
                        stop_early=False, 
                        eos_id=[[SEP_ID]]
                        )
            

def refined_summary_sampling(inp, 
                           enc_output, 
                           draft_summary, 
                           padding_mask, 
                           sampling_type='greedy', 
                           temperature=0.9, 
                           p=0.9, 
                           k=25,
                           beam_search=False,
                           training=False):
        """
        Inference call, builds a refined summary
        
        It first masks each word in the summary draft one by one,
        then feeds the draft to BERT to generate context vectors.
        """
        
        log.info(f"Building: 'Refined {sampling_type} decoder'")
        N = tf.shape(enc_output)[0]
        refined_summary = draft_summary
        batch = tf.shape(draft_summary)[0]
        print(f'draft_summary {tf.shape(draft_summary)}')
        dec_outputs = []
        dec_logits = []
        attention_dists = []
        for i in (range(1, config.summ_length)):
            
            # (batch_size, seq_len)
            refined_summary_ = mask_timestamp(refined_summary, i, MASK_ID)
            
            # (batch_size, seq_len, d_bert)
            context_vectors = model.bert_model(refined_summary_)[0]
            
            # (batch_size, seq_len, d_bert), (_)
            dec_output, dec_logits_i, attention_dist = model.decoder(
                                                                    context_vectors,
                                                                    enc_output,
                                                                    training=training,
                                                                    look_ahead_mask=None,
                                                                    padding_mask=padding_mask
                                                                  )
            
            # (batch_size, 1, vocab_len)
            dec_output_i = dec_output[:, i:i+1 ,:]
            if sampling_type == 'nucleus':
              preds = tf.cast(nucleus_sampling((tf.squeeze(dec_output_i)/ temperature), p=p), tf.int32)
            elif sampling_type == 'topk':
              preds = tf.cast(top_k_sampling((tf.squeeze(dec_output_i)/ temperature), k=k), tf.int32)
            elif sampling_type == 'random_sampling':
              preds = tf.cast(sampling(tf.squeeze(dec_output_i)/ temperature), tf.int32)
            else:
              preds = tf.cast(tf.argmax(dec_output_i, axis=-1), tf.int32)
            refined_summary = with_column(refined_summary, i, preds)
        # (batch_size, seq_len, vocab_len), (batch_size, seq_len), (_)        
        return refined_summary, attention_dist

def predict_using_sampling(
                           inp, 
                           draft_decoder_sampling_type='topk',
                           refine_decoder_sampling_type='topk', 
                           temperature=0.9, 
                           p=0.9, 
                           k=25):
  
  dec_padding_mask = create_padding_mask(inp)
  
  # (batch_size, seq_len, d_bert)
  enc_output = model.bert_model(inp)[0]
  # (batch_size, seq_len, vocab_len), (_)
  preds_draft_summary, draft_attention_dist = draft_summary_sampling( 
                                                                      inp,
                                                                      enc_output=enc_output,
                                                                      look_ahead_mask=None,
                                                                      padding_mask=dec_padding_mask,
                                                                      sampling_type=draft_decoder_sampling_type,
                                                                      temperature=temperature,
                                                                      p=p, 
                                                                      k=k,
                                                                    )
  # (batch_size, seq_len, vocab_len), ()
  preds_refined_summary, refined_attention_dist = refined_summary_sampling(
                                                                            inp,
                                                                            enc_output=enc_output,
                                                                            padding_mask=dec_padding_mask,
                                                                            draft_summary=preds_draft_summary,
                                                                            sampling_type=refine_decoder_sampling_type, 
                                                                            temperature=temperature, 
                                                                            p=p, 
                                                                            k=k,
                                                                            beam_search=False
                                                                            )


  return preds_draft_summary, draft_attention_dist, preds_refined_summary[:, 1:], refined_attention_dist

def predict_using_beam_search(
                              inp, 
                              beam_size=3, 
                              refine_decoder_sampling_type='nucleus', 
                              temperature=0.9, 
                              p=0.9, 
                              k=25):
  
  dec_padding_mask = create_padding_mask(inp)
  enc_output = model.bert_model(inp)[0]
  # (batch_size, seq_len, d_bert)
  #[batch_size*beam_size, input_Seq_len, d_bert]
  translated_output_temp = draft_summary_beam_search(inp, enc_output, dec_padding_mask, beam_size)
  # Take the sequence with high score (the last one)
  preds_draft_summary = translated_output_temp[0][:,0,:] 
  
  #print(f'preds_draft_summary {preds_draft_summary}')
  preds_refined_summary, refined_attention_dist = refined_summary_sampling(
                                                                        inp,
                                                                        enc_output=enc_output,
                                                                        padding_mask=dec_padding_mask,
                                                                        draft_summary=preds_draft_summary, 
                                                                        sampling_type=refine_decoder_sampling_type, 
                                                                        temperature=temperature, 
                                                                        p=p, 
                                                                        k=k,
                                                                        beam_search=True
                                                                        )
  print(f'preds_refined_summary shape {tf.shape(preds_refined_summary[:, 1:])}')
  return preds_draft_summary, preds_refined_summary[:, 1:], refined_attention_dist

''' 
Set the latest checkpoint and run the below piece of code for inference. 
ckpt = tf.train.Checkpoint(
                           model=model,
                           optimizer=optimizer
                          )
ckpt.restore('/content/drive/My Drive/Text_summarization/BERT_text_summarisation/cnn_checkpoints/ckpt-35')
ip_ids = tokenizer.encode('Your summary sentence')
preds_draft_summary, draft_attention_dist, preds_refined_summary, _ = predict(ip_ids)
preds_refined_summary = (tokenizer.decode([i for i in preds_refined_summary if i not in [CLS_ID, SEP_ID, 0]]))
print(f'the predicted_refined_greedy auto regressive --> {preds_refined_summary if preds_refined_summary else "EMPTY"}')
'''
