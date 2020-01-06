# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.random.set_seed(100)
import time
import os
import shutil
import tensorflow_datasets as tfds
from preprocess import create_train_data
from transformer import Transformer, Pointer_Generator, create_masks
from hyper_parameters import h_parms
from configuration import config
from metrics import optimizer, loss_function, get_loss_and_accuracy, tf_write_summary, monitor_run
from input_path import file_path
from creates import log, train_summary_writer, valid_summary_writer
from create_tokenizer import tokenizer_en
from local_tf_ops import *

train_dataset, val_dataset, num_of_train_examples, num_of_valid_examples = create_train_data()
train_loss, train_accuracy = get_loss_and_accuracy()
validation_loss, validation_accuracy = get_loss_and_accuracy()

transformer = Transformer(
                          num_layers=config.num_layers, 
                          d_model=config.d_model, 
                          num_heads=config.num_heads, 
                          dff=config.dff, 
                          input_vocab_size=config.input_vocab_size, 
                          target_vocab_size=config.target_vocab_size, 
                          rate=h_parms.dropout_rate
                          )
pointer_generator   = Pointer_Generator()

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar, inp_shape, tar_shape, batch):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  with tf.GradientTape() as tape:
    predictions, attention_weights, dec_output = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    train_variables = transformer.trainable_variables
    tf.debugging.check_numerics(
                                predictions,
                                "Nan's in the transformer predictions"
                                )
    if config.copy_gen:
      predictions = pointer_generator(dec_output, predictions, attention_weights, inp, 
                            inp_shape, tar_shape, batch, training=True)
      tf.debugging.check_numerics(
                                predictions,
                                "Nan's in the pointer_generator predictions"
                                )
    train_variables = train_variables + pointer_generator.trainable_variables
    loss = loss_function(tar_real, predictions)
  gradients = tape.gradient(loss, train_variables)    
  optimizer.apply_gradients(zip(gradients, train_variables))
  train_loss(loss)
  train_accuracy(tar_real, predictions)  

@tf.function(input_signature=val_step_signature)
def val_step(inp, tar, inp_shape, tar_shape, batch):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  predictions, attention_weights, dec_output = transformer(
                                                           inp, 
                                                           tar_inp, 
                                                           False, 
                                                           enc_padding_mask, 
                                                           combined_mask, 
                                                           dec_padding_mask
                                                           )
  if config.copy_gen:
    predictions = pointer_generator(
                            dec_output, 
                            predictions, 
                            attention_weights, 
                            inp, 
                            inp_shape, 
                            tar_shape, 
                            batch, 
                            training=False
                            )
  loss = loss_function(tar_real, predictions)
  validation_loss(loss)
  validation_accuracy(tar_real, predictions)
  
@tf.function(input_signature=val_step_with_summary_signature)
def val_step_with_summary(inp, tar, epoch, inp_shape, tar_shape, batch):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  predictions, attention_weights, dec_output = transformer(
                                                           inp, 
                                                           tar_inp, 
                                                           False, 
                                                           enc_padding_mask, 
                                                           combined_mask, 
                                                           dec_padding_mask
                                                           )
  if config.copy_gen:
    predictions = pointer_generator(
                            dec_output, 
                            predictions, 
                            attention_weights, 
                            inp, 
                            inp_shape, 
                            tar_shape, 
                            batch, 
                            training=False
                            )
  loss = loss_function(tar_real, predictions)
  
  validation_loss(loss)
  validation_accuracy(tar_real, predictions)
  return tf_write_summary(tar_real, predictions, inp[:, 1:], epoch)
  
# calculate rouge for only the first batch
def calc_validation_loss(validation_dataset, epoch):
  validation_loss.reset_states()
  validation_accuracy.reset_states()
  val_acc = 0
  val_loss = 0
  
  for (batch, (inp, tar)) in enumerate(validation_dataset):
    if batch == 0:
        rouge_score, bert_score = val_step_with_summary(inp, tar, epoch, inp.shape[1], tar.shape[1]-1, inp.shape[0])
    else:
        val_step(inp, tar, inp.shape[1], tar.shape[1]-1, inp.shape[0])
    val_loss += validation_loss.result()
    val_acc += validation_accuracy.result()
  return (val_acc.numpy()/(batch+1), val_loss.numpy()/(batch+1), rouge_score, bert_score)


def check_ckpt(checkpoint_path):
    ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer,
                           pointer_generator=pointer_generator)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=20)
    if tf.train.latest_checkpoint(checkpoint_path):
      ckpt.restore(ckpt_manager.latest_checkpoint)
      log.info(ckpt_manager.latest_checkpoint +' restored')
      latest_ckpt = int(ckpt_manager.latest_checkpoint[-2:])
    else:
        latest_ckpt=0
        log.info('Training from scratch')
    return (ckpt_manager, latest_ckpt)

# if a checkpoint exists, restore the latest checkpoint.
ck_pt_mgr, latest_ckpt = check_ckpt(file_path.checkpoint_path)

for epoch in range(h_parms.epochs):
  start = time.time()  
  train_loss.reset_states()
  train_accuracy.reset_states()
  validation_loss.reset_states()
  validation_accuracy.reset_states()
  # inp -> document, tar -> summary
  for (batch, (inp, tar)) in enumerate(train_dataset):
  # the target is shifted right during training hence its shape is subtracted by 1
  # not able to do this inside tf.function since it doesn't allow this operation
    train_step(inp, tar, inp.shape[1], tar.shape[1]-1, inp.shape[0])        
    if batch==0 and epoch ==0:
      log.info(transformer.summary())
      if config.copy_gen:
        log.info(pointer_generator.summary())
      log.info(batch_zero.format(time.time()-start))
    if batch % config.print_chks == 0:
      log.info(batch_run_details.format(
        epoch + 1, batch, train_loss.result(), train_accuracy.result()))
  if epoch == 0:
    try:
      if batch > 0:
        num_of_recs_post_filter_atmost = ((batch)*h_parms.batch_size)/num_of_train_examples
        num_of_recs_post_filter_atleast = ((batch-1)*h_parms.batch_size)/num_of_train_examples
        log.info(f'Number of records used for training should be in between {num_of_recs_post_filter_atleast*100} - \
                {num_of_recs_post_filter_atmost*100}% of training data')
    except NameError:
      assert False, 'Training dataset is empty'
    else:
      log.info(f'Number of records used for training is {sum(1 for l in train_dataset.unbatch())}')
  (val_acc, val_loss, rouge_score, bert_score) = calc_validation_loss(val_dataset, epoch+1)
  ckpt_save_path = ck_pt_mgr.save()
  latest_ckpt+=1
  log.info(
           model_metrics.format(epoch+1, train_loss.result(), 
           train_accuracy.result(),
           val_loss, val_acc,
           rouge_score, bert_score)
          )
  log.info(epoch_timing.format(epoch + 1, time.time() - start))
  log.info(checkpoint_details.format(epoch+1, ckpt_save_path))
  if not monitor_run(latest_ckpt, ckpt_save_path, val_loss, val_acc, bert_score, rouge_score, valid_summary_writer, epoch):
    break
