# -*- coding: utf-8 -*-

from beam_search import beam_search
from preprocess import create_dataset,filter_token_size
import tensorflow as tf
from transformer import Transformer, Generator, create_masks
from hyper_parameters import config
from metrics import optimizer
from input_path import file_path
from preprocess import tokenizer_en
import time
import os



transformer = Transformer(
        num_layers=config.num_layers, 
        d_model=config.d_model, 
        num_heads=config.num_heads, 
        dff=config.dff, 
        input_vocab_size=config.input_vocab_size, 
        target_vocab_size=config.target_vocab_size, 
        rate=config.dropout_rate)

generator   = Generator()

def encode(doc, summary):
    lang1 = tokenizer_en.encode(
    doc.numpy()) 
    lang2 = tokenizer_en.encode(
    summary.numpy()) 
    return lang1, lang2
    
def tf_encode(doc, summary):
    return tf.py_function(encode, [doc, summary], [tf.int64, tf.int64])


def restore_chkpt(checkpoint_path):
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer,
                               generator=generator)

    #ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=20)
    
    # if a checkpoint exists, restore the latest checkpoint.
    assert tf.train.latest_checkpoint(checkpoint_path), 'checkpoint not available'
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
    print (tf.train.latest_checkpoint(checkpoint_path), 'checkpoint restored!!')

def beam_search_eval(inp_sentences, beam_size):
  
  start = [tokenizer_en.vocab_size] * len(inp_sentences)
  end = [tokenizer_en.vocab_size+1]
  encoder_input = tf.tile(inp_sentences, multiples=[beam_size, 1])
  batch, inp_shape = encoder_input.shape
  
  def transformer_query(output):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
          encoder_input, output)
    predictions, attention_weights, dec_output = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask
                                                   )

    if config.copy_gen:
      predictions = generator(dec_output, predictions, attention_weights, 
                              encoder_input, inp_shape, output.shape[-1], 
                              batch, False)

    # select the last sequence
    return (predictions[:,-1:,:])  # (batch_size, 1, target_vocab_size)
  return beam_search(
          transformer_query, 
          start, 
          beam_size, 
          config.summ_length, 
          config.target_vocab_size, 
          0.6, 
          stop_early=False, 
          eos_id=[end])


def infer(documents_path, beam_size, checkpoint_path):
  doc, summ = create_dataset(documents_path, num_examples=config.num_examples)
  train_examples = tf.data.Dataset.from_tensor_slices((doc, summ))
  batch = len(doc)//2
  train_dataset = train_examples.map(tf_encode)
  train_dataset = train_dataset.filter(filter_token_size)
  train_dataset = train_dataset.cache()
  train_dataset = train_dataset.padded_batch(
      batch, padded_shapes=([-1], [-1]))
  #print(f'Number of records before filtering was {len(doc)}')
  #print(f'Number of records to be inferenced is {sum(1 for l in train_dataset) * batch} approx')
  restore_chkpt(checkpoint_path)
  start_time = time.time()
  for (_, (inp, tar)) in enumerate(train_dataset):
    translated_output_temp = beam_search_eval(inp, beam_size)
    #print(translated_output_temp)
    for true_summary, top_sentence_ids in zip(tar, translated_output_temp[0][:,0,:]):
      print()
      print('Original summary: {}'.format(tokenizer_en.decode([j for j in true_summary if j < tokenizer_en.vocab_size])))
      print('Predicted summary: {}'.format(tokenizer_en.decode([j for j in top_sentence_ids if j < tokenizer_en.vocab_size if j > 0 ])))
      print()
  print('time to process {}'.format(time.time()-start_time))    

if __name__ == "__main__":
    infer(file_path.infer_csv_path, beam_size=config.beam_size, checkpoint_path=file_path.old_checkpoint_path)