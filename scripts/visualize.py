# -*- coding: utf-8 -*-

import tempfile
import tensorflow as tf
import matplotlib
import tensorflow_datasets as tfds
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
import numpy as np
import pandas as pd
import time
from hyper_parameters import h_parms
from create_tokenizer import tokenizer_en
from configuration import config
from preprocess import tf_encode
  
def create_temp_file( text):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with tf.io.gfile.GFile(temp_file.name, "w") as w:
      w.write(text)
    return temp_file.name

# histogram of tokens per batch_size
# arg1 :- must be a padded_batch dataset
def hist_tokens_per_batch(tf_dataset, num_of_examples, samples_to_try=0.1, split='valid'):
    x=[]
    samples_per_batch = int((samples_to_try*(num_of_examples))//h_parms.batch_size)
    tf_dataset = tf_dataset.padded_batch(h_parms.batch_size, padded_shapes=([-1], [-1]))
    tf_dataset = tf_dataset.take(samples_per_batch).cache()
    tf_dataset = tf_dataset.prefetch(buffer_size=samples_per_batch)
    for (i, j) in (tf_dataset):
        x.append((tf.size(i) + tf.size(j)).numpy())
    print(f'Descriptive statistics on tokens per batch for {split}')
    print(pd.Series(x).describe())
    if config.create_hist:
      print(f'creating histogram for {samples_per_batch*h_parms.batch_size} samples')
      plt.hist(x, bins=20)
      plt.xlabel('Total tokens per batch')
      plt.ylabel('No of times')
      plt.savefig('#_of_tokens per batch in '+split+' set.png')
      plt.close() 

# histogram of Summary_lengths
# arg1 :- must be a padded_batch dataset
def hist_summary_length(tf_dataset, num_of_examples, samples_to_try=0.1, split='valid'):
    summary=[]
    document=[]
    samples = int((samples_to_try*(num_of_examples)))
    tf_dataset = tf_dataset.take(samples).cache()
    tf_dataset = tf_dataset.prefetch(buffer_size=samples)
    for (doc, summ) in (tf_dataset):
        summary.append(summ.shape[0])
        document.append(doc.shape[0])
    combined = [i+j for i,j in zip(summary, document)]
    print(f'Descriptive statistics on Summary length based for {split} set')
    print(pd.Series(summary).describe(percentiles=[0.25, 0.5, 0.8, 0.9, 0.95, 0.97] ))
    print(f'Descriptive statistics on Document length based for {split} set')
    print(pd.Series(document).describe(percentiles=[0.25, 0.5, 0.8, 0.9, 0.95, 0.97] ))
    print(f'Descriptive statistics for the combined length of docs and summs based for {split} set')
    print(pd.Series(combined).describe(percentiles=[0.25, 0.5, 0.8, 0.9, 0.95, 0.97] ))
    if config.create_hist:
      print(f'creating histogram for {samples} samples')
      plt.hist([summary, document, combined], alpha=0.5, bins=20, label=['summary', 'document', 'combined'])
      plt.xlabel('lengths of document and summary')
      plt.ylabel('Counts')
      plt.legend(loc='upper right')
      plt.savefig(split+'_lengths of document, summary and combined.png')
      plt.close() 

def beam_search_train(inp_sentences, beam_size):
  
  start = [tokenizer_en.vocab_size] * inp_sentences.shape[0]
  end = [tokenizer_en.vocab_size+1]
  encoder_input = tf.tile(inp_sentences, multiples=[beam_size, 1])
  def decoder_query(output):

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
          encoder_input, output)
    predictions, attention_weights, dec_output = model(
                                                      encoder_input, 
                                                      output,
                                                      False,
                                                      enc_padding_mask,
                                                      combined_mask,
                                                      dec_padding_mask
                                                      )
    if predictions:
      predictions = generator(dec_output, predictions, attention_weights, encoder_input, 
                                    inp_sentences.shape[1], output.shape[1], inp_sentences.shape[0], beam_size, False)

    # select the last sequence
    # (batch_size, 1, target_vocab_size)
    return (predictions[:,-1:,:]) 
  return (beam_search(decoder_query, start, beam_size, summ_length, 
                      target_vocab_size, 0.6, stop_early=True, eos_id=[end]))
 
if __name__== '__main__':
  examples, metadata = tfds.load(config.tfds_name, with_info=True, as_supervised=True)   
  splits = examples.keys()
  percentage_of_samples = 0.1
  tf_datasets = {}
  buffer_size = {}
  for split in splits:
    tf_datasets[split] = examples[split].map(tf_encode, num_parallel_calls=-1)
    buffer_size[split] = metadata.splits[split].num_examples
  for split in tf_datasets:
    #create histogram for summary_lengths and token
    hist_summary_length(tf_datasets[split], buffer_size[split], percentage_of_samples, split)  
    #hist_tokens_per_batch(tf_datasets[split], buffer_size[split], percentage_of_samples, split)

  if config.show_detokenized_samples:
    inp, tar = next(iter(examples['train']))
    for ip,ta in zip(inp.numpy(), tar.numpy()):
      print(tokenizer_en.decode([i for i in ta if i < tokenizer_en.vocab_size]))
      print(tokenizer_en.decode([i for i in ip if i < tokenizer_en.vocab_size]))
      break
    
