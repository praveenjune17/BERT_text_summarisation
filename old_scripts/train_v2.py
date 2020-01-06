# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.random.set_seed(100)
import time
import os
import shutil
import tensorflow_datasets as tfds
from preprocess import create_train_data
from transformer import Transformer, Generator, create_masks
from hyper_parameters import h_parms
from configuration import config
from metrics import optimizer, loss_function, get_loss_and_accuracy, tf_write_summary
from input_path import file_path
from creates import log, train_summary_writer, valid_summary_writer
from create_tokenizer import tokenizer_en
from local_tf_ops import *

assert(str(input('set the last_validation_loss parameter,reply "ok" if set ')) == 'ok'), \
            'Please change the hyper prameters and proceed with training model'
            
if input('Remove summaries dir and tensorboard_logs ? reply "yes or no" ') == 'yes':
  try:
    shutil.rmtree(file_path.summary_write_path)
    shutil.rmtree(file_path.tensorboard_log)
  except FileNotFoundError:
    pass

train_dataset, val_dataset, num_of_train_examples = create_train_data()
train_loss, train_accuracy = get_loss_and_accuracy()
validation_loss, validation_accuracy = get_loss_and_accuracy()

if config.show_detokenized_samples:
  inp, tar = next(iter(train_dataset))
  for ip,ta in zip(inp.numpy(), tar.numpy()):
    log.info(tokenizer_en.decode([i for i in ta if i < tokenizer_en.vocab_size]))
    log.info(tokenizer_en.decode([i for i in ip if i < tokenizer_en.vocab_size]))
    break

transformer = Transformer(
        num_layers=config.num_layers, 
        d_model=config.d_model, 
        num_heads=config.num_heads, 
        dff=config.dff, 
        input_vocab_size=config.input_vocab_size, 
        target_vocab_size=config.target_vocab_size, 
        rate=h_parms.dropout_rate)
generator   = Generator()

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
      predictions = generator(dec_output, predictions, attention_weights, inp, 
                            inp_shape, tar_shape, batch, training=True)
      tf.debugging.check_numerics(
                                predictions,
                                "Nan's in the generator predictions"
                                )
      
    train_variables = train_variables + generator.trainable_variables
    
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
  
  
  predictions, attention_weights, dec_output = transformer(inp, tar_inp, 
                               False, 
                               enc_padding_mask, 
                               combined_mask, 
                               dec_padding_mask)
  if config.copy_gen:
    predictions = generator(dec_output, predictions, attention_weights, 
                            inp, inp_shape, tar_shape, batch, training=False)
  loss = loss_function(tar_real, predictions)
  
  validation_loss(loss)
  validation_accuracy(tar_real, predictions)
  
@tf.function(input_signature=val_step_with_summary_signature)
def val_step_with_summary(inp, tar, epoch, inp_shape, tar_shape, batch):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
  
  predictions, attention_weights, dec_output = transformer(inp, tar_inp, 
                               False, 
                               enc_padding_mask, 
                               combined_mask, 
                               dec_padding_mask)
  if config.copy_gen:
    predictions = generator(dec_output, predictions, attention_weights, 
                            inp, inp_shape, tar_shape, batch, training=False)
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
                           generator=generator)

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
        log.info(generator.summary())
      log.info(batch_zero.format(time.time()-start))
    if batch % config.print_chks == 0:
      log.info(batch_run_details.format(
        epoch + 1, batch, train_loss.result(), train_accuracy.result()))
  data_after_filter = ((batch-1)*h_parms.batch_size)/num_of_train_examples
  log.info(f'Atleast {data_after_filter*100}% of training data is used')
  (val_acc, val_loss, rouge_score, bert_score) = calc_validation_loss(val_dataset, epoch+1)
  ckpt_save_path = ck_pt_mgr.save()
  ckpt_fold, ckpt_string = os.path.split(ckpt_save_path)
  latest_ckpt+=1
  if config.run_tensorboard:
    with train_summary_writer.as_default():
      tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
      tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch)

    with valid_summary_writer.as_default():
      tf.summary.scalar('validation_loss', validation_loss.result(), step=epoch)
      tf.summary.scalar('validation_accuracy', validation_accuracy.result(), step=epoch)
      tf.summary.scalar('validation_total_loss', val_acc, step=epoch)
      tf.summary.scalar('validation_total_accuracy', val_loss, step=epoch)
      tf.summary.scalar('ROUGE_score', rouge_score, step=epoch)
      tf.summary.scalar('BERT_score', bert_score, step=epoch)

  if config.verbose:

    model_metrics = 'Epoch {}, Train Loss: {:.4f}, Train_Accuracy: {:.4f}, \
                     Valid Loss: {:.4f},                                   \
                     Valid Accuracy: {:4f},                                \
                     ROUGE_score {},                                       \
                     BERT_SCORE {}'
    epoch_timing  = 'Time taken for {} epoch : {} secs' 
    checkpoint_details = 'Saving checkpoint for epoch {} at {}'

    log.info(model_metrics.format(epoch+1,
                         train_loss.result(), 
                         train_accuracy.result(),
                         val_loss, 
                         val_acc,
                         rouge_score,
                         bert_score))
    log.info(epoch_timing.format(epoch + 1, time.time() - start))
    log.info(checkpoint_details.format(epoch+1, ckpt_save_path))

  if (latest_ckpt > config.monitor_only_after) and (config.last_validation_loss > val_loss):
    
    # reset tolerance to zero if the validation loss decreases before the tolerance threshold
    config.init_tolerance=0
    config.last_validation_loss =  val_loss
    ckpt_files_tocopy = [files for files in os.listdir(os.path.split(ckpt_save_path)[0]) \
                         if ckpt_string in files]
    log.info(f'Validation loss is {val_loss} so checkpoint files {ckpt_string}           \
             will be copied to best checkpoint directory')
    shutil.copy2(os.path.join(ckpt_fold, 'checkpoint'), file_path.best_ckpt_path)
    for files in ckpt_files_tocopy:
        shutil.copy2(os.path.join(ckpt_fold, files), file_path.best_ckpt_path)
  else:
    config.init_tolerance+=1

  if config.init_tolerance > config.tolerance_threshold:
    log.warning('Tolerance exceeded')
  if config.early_stop and config.init_tolerance > config.tolerance_threshold:
    log.info(f'Early stopping since the validation loss exceeded the tolerance threshold')
    break
  if train_loss.result() == 0:
    log.info('Train loss reached zero so stopping training')
    break
