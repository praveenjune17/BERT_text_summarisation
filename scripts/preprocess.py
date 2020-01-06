# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
from hyper_parameters import h_parms
from configuration import config
from input_path import file_path
from create_tokenizer import tokenizer_en, create_dataframe
from creates import log

AUTOTUNE = tf.data.experimental.AUTOTUNE

def encode(doc, summary):
    lang1 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
    doc.numpy()) + [tokenizer_en.vocab_size+1]
    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
    summary.numpy()) + [tokenizer_en.vocab_size+1]
    return lang1, lang2

# Set threshold for document and  summary length
def filter_max_length(x, y):
    return tf.logical_and(
                          tf.size(x) <= config.doc_length,
                          tf.size(y) <= config.summ_length
                         )

def filter_combined_length(x, y):
    return tf.math.less_equal(
                              (tf.size(x) + tf.size(y)), 
                              config.max_tokens_per_line
                             )
                        
# this function should be added after padded batch step
def filter_batch_token_size(x, y):
    return tf.math.less_equal(
                              (tf.size(x) + tf.size(y)), 
                              config.max_tokens_per_line*h_parms.batch_size
                             )
    
def tf_encode(doc, summary):
    return tf.py_function(encode, [doc, summary], [tf.int64, tf.int64])

def map_batch_shuffle(dataset, buffer_size, split, 
                      shuffle=True, batch_size=h_parms.batch_size,
                      filter_off=False):
    tf_dataset = dataset.map(tf_encode, num_parallel_calls=AUTOTUNE)
    if not filter_off:
        tf_dataset = tf_dataset.filter(filter_combined_length)
    tf_dataset = tf_dataset.cache()
    if split == 'train' and shuffle and (not config.use_tfds):
       tf_dataset = tf_dataset.shuffle(buffer_size, seed = 100)
    tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=([-1], [-1]))
    tf_dataset = tf_dataset.prefetch(buffer_size=AUTOTUNE)
    return tf_dataset
    
def create_train_data(num_samples_to_train = config.num_examples_to_train, shuffle=True, filter_off=False):

    if config.use_tfds:
        examples, metadata = tfds.load(config.tfds_name, with_info=True, as_supervised=True)
        other_ds = 'validation' if 'validation' in examples else 'test'
        train_examples = examples['train']
        valid_examples = examples[other_ds]
        train_buffer_size = metadata.splits['train'].num_examples
        valid_buffer_size = metadata.splits[other_ds].num_examples
    else:
        doc, summ = create_dataframe(file_path.train_csv_path, num_samples_to_train)
        X_train, X_test, y_train, y_test = train_test_split(
                                                            doc, 
                                                            summ, 
                                                            test_size=config.test_size, 
                                                            random_state=42
                                                            )
        train_examples = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        valid_examples = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        train_buffer_size = len(X_train) 
        valid_buffer_size = len(X_test)
    train_dataset = map_batch_shuffle(
                                     train_examples, 
                                     train_buffer_size, 
                                     split = 'train',
                                     shuffle = shuffle,
                                     batch_size=h_parms.batch_size,
                                     filter_off=filter_off
                                     )
    valid_dataset = map_batch_shuffle(
                                     valid_examples, 
                                     valid_buffer_size, 
                                     split='valid',
                                     batch_size=h_parms.batch_size,
                                     filter_off=filter_off
                                     )
    log.info('Train and Test tf_datasets created')
    return (train_dataset, valid_dataset, train_buffer_size, valid_buffer_size)
    
def infer_data_from_df(num_of_infer_examples=config.num_examples_to_infer):
    doc, summ = create_dataframe(file_path.infer_csv_path, num_of_infer_examples)
    infer_examples = tf.data.Dataset.from_tensor_slices((doc, summ))
    infer_buffer_size = len(doc)
    infer_dataset = map_batch_shuffle(
                                      infer_examples, 
                                      infer_buffer_size, 
                                      split = 'infer',
                                      batch_size=1             #TODO  if > 1 then (ip and op) might be shuffled during beam search 
                                      )
    log.info('infer tf_dataset created')
    return infer_dataset
