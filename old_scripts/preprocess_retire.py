# -*- coding: utf-8 -*-


import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
from hyper_parameters import config
from input_path import file_path
from creates import log
from create_tokenizer import tokenizer_en

''' 
Use this when the input is tensorflow datasets 
'''

AUTOTUNE = tf.data.experimental.AUTOTUNE

def create_dataset(path, num_examples):
    df = pd.read_csv(path)
    df = df[:num_examples]
    assert not df.isnull().any().any(), f'examples contains {df.isnull().sum().sum()} nans'
    return (df[file_path.document].values, df[file_path.summary].values)

def encode(doc, summary):
    lang1 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
    doc.numpy()) + [tokenizer_en.vocab_size+1]
    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
    summary.numpy()) + [tokenizer_en.vocab_size+1]
    return lang1, lang2

    # Set threshold for document and  summary length
def filter_max_length(x, y):
    return tf.logical_and(tf.size(x) <= config.doc_length,
                        tf.size(y) <= config.summ_length)
    
def filter_token_size(x, y):
    return tf.math.less_equal(config.batch_size*(tf.size(x) + tf.size(y)), config.max_tokens_per_batch)


def tf_encode(doc, summary):
    return tf.py_function(encode, [doc, summary], [tf.int64, tf.int64])
    
def batch_shuffle(dataset, buffer_size, split, batch_size=config.batch_size):
    tf_dataset = dataset.map(tf_encode, num_parallel_calls=AUTOTUNE)                           
    tf_dataset = tf_dataset.filter(filter_token_size)
    sum_of_records = sum(1 for l in tf_dataset)                                                       #optimize
    if sum_of_records > 2,000:
        tf_dataset = tf_dataset.cache()
    if buffer_size:
        tf_dataset = tf_dataset.shuffle(buffer_size, seed = 100)
    tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=([-1], [-1]))
    tf_dataset = tf_dataset.prefetch(buffer_size=AUTOTUNE)
    log.info(f'Number of records {split} filtered {buffer_size - sum_of_records}')
    log.info(f'Number of records to be {split}ed {sum_of_records}')
    return tf_dataset
  

def train_data_from_tfds():
    examples, metadata = tfds.load('gigaword', with_info=True, as_supervised=True)
    train_buffer_size = metadata.splits['train'].num_examples
    valid_buffer_size = metadata.splits['test'].num_examples
    train_dataset = batch_shuffle(
                                  examples['train'], 
                                  train_buffer_size, 
                                  split = 'train',
                                  batch_size=config.batch_size
                                  )
    valid_dataset = batch_shuffle(
                                  examples['test'], 
                                  valid_buffer_size, 
                                  split='test',
                                  batch_size=config.batch_size
                                  )
    log.info('Training and Test set created')
    return train_dataset, valid_dataset
