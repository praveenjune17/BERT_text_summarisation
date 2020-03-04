# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
from functools import partial
from hyper_parameters import h_parms
from configuration import config
from input_path import file_path
from create_tokenizer import tokenizer, create_dataframe
from creates import log

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Special Tokens
UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103


def pad(l, n, pad=0):
    """
    Pad the list 'l' to have size 'n' using 'padding_element'
    """
    pad_with = (0, max(0, n - len(l)))
    return np.pad(l, pad_with, mode='constant', constant_values=pad)


def encode(sent_1, sent_2, tokenizer, input_seq_len, output_seq_len):
    
    input_ids_1 = tokenizer.encode(sent_1.numpy().decode('utf-8'))
    input_ids_2 = tokenizer.encode(sent_2.numpy().decode('utf-8'))
    
    # Account for [CLS] and [SEP] with "- 2"
    if len(input_ids_1) > input_seq_len - 2:
        input_ids_1 = input_ids_1[0:(input_seq_len - 2)]
    if len(input_ids_2) > (output_seq_len + 1) - 2:
        input_ids_2 = input_ids_2[0:((output_seq_len + 1) - 2)]
    input_ids_1 = pad(input_ids_1, input_seq_len, 0)
    input_ids_2 = pad(input_ids_2, output_seq_len + 1, 0)    
    return input_ids_1, input_ids_2


def tf_encode(tokenizer, input_seq_len, output_seq_len):
    """
    Operations inside `.map()` run in graph mode and receive a graph
    tensor that do not have a `numpy` attribute.
    The tokenizer expects a string or Unicode symbol to encode it into integers.
    Hence, you need to run the encoding inside a `tf.py_function`,
    which receives an eager tensor having a numpy attribute that contains the string value.
    """    
    def f(s1, s2):
        encode_ = partial(encode, tokenizer=tokenizer, input_seq_len=input_seq_len, output_seq_len=output_seq_len)
        return tf.py_function(encode_, [s1, s2], [tf.int32, tf.int32])
    
    return f

# Set threshold for document and  summary length
def filter_max_length(x, y):
    return tf.logical_and(
                          tf.size(x[0]) <= config.doc_length,
                          tf.size(y[0]) <= config.summ_length
                         )

def filter_combined_length(x, y):
    return tf.math.less_equal(
                              (tf.math.count_nonzero(x) + tf.math.count_nonzero(y)), 
                              config.max_tokens_per_line
                             )
                        
# this function should be added after padded batch step
def filter_batch_token_size(x, y):
    return tf.math.less_equal(
                              (tf.size(x[0]) + tf.size(y[0])), 
                              config.max_tokens_per_line*h_parms.batch_size
                             )
    
def map_batch_shuffle(dataset, 
                      buffer_size, 
                      split, 
                      shuffle=True, 
                      batch_size=h_parms.batch_size,
                      filter_off=False):
    tf_dataset = dataset.map(
                            tf_encode(
                                tokenizer, 
                                config.doc_length, 
                                config.summ_length
                                ), num_parallel_calls=tf.data.experimental.AUTOTUNE
                            )
    if not filter_off:
        tf_dataset = tf_dataset.filter(filter_combined_length)
    tf_dataset = tf_dataset.cache()
    if split == 'train' and shuffle and (not config.use_tfds):
       tf_dataset = tf_dataset.shuffle(buffer_size, seed = 100)
    tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=([-1], [-1]))
    tf_dataset = tf_dataset.prefetch(buffer_size=AUTOTUNE)
    return tf_dataset
    
def create_train_data(num_samples_to_train=config.num_examples_to_train, 
                      shuffle=True, 
                      filter_off=False):

    if config.use_tfds:
        train_examples, _ = tfds.load(
                                       config.tfds_name, 
                                       with_info=True,
                                       as_supervised=True, 
                                       data_dir='/content/drive/My Drive/Text_summarization/cnn_dataset',
                                       builder_kwargs={"version": "2.0.0"},
                                       split=tfds.core.ReadInstruction('train', from_=90, to=100, unit='%')
                                      )
        valid_examples, _ = tfds.load(
                                       config.tfds_name, 
                                       with_info=True,
                                       as_supervised=True, 
                                       data_dir='/content/drive/My Drive/Text_summarization/cnn_dataset',
                                       builder_kwargs={"version": "2.0.0"},
                                       split='validation'
                                      )
        train_buffer_size = 287113
        valid_buffer_size = 13368
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
                                     batch_size=h_parms.validation_batch_size,
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
                                      batch_size=h_parms.validation_batch_size              
                                      )
    log.info('infer tf_dataset created')
    return infer_dataset
