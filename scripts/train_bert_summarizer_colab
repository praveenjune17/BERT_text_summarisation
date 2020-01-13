# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.random.set_seed(100)
#tf.config.optimizer.set_jit(True)
import time
#from tensorflow.keras.mixed_precision import experimental as mixed_precision
from preprocess import create_train_data
from hyper_parameters import h_parms
from configuration import config
from metrics import optimizer, loss_function, label_smoothing, get_loss_and_accuracy, tf_write_summary, monitor_run
from input_path import file_path
from creates import log, train_summary_writer, valid_summary_writer
from create_tokenizer_v2 import tokenizer, model
from local_tf_ops_v2 import *

#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)
#optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

train_dataset, val_dataset, num_of_train_examples, _ = create_train_data()
train_loss, train_accuracy = get_loss_and_accuracy()
validation_loss, validation_accuracy = get_loss_and_accuracy()
accumulators = []
@tf.function(input_signature=train_step_signature)
def train_step(input_ids, 
               input_mask, 
               input_segment_ids, 
               target_ids_, 
               target_mask, 
               target_segment_ids, 
               target_ids, 
               mask, 
               grad_accum_flag):
  with tf.GradientTape() as tape:
    (draft_predictions, draft_attention_weights, 
      refine_predictions, refine_attention_weights) = model(
                                                           input_ids, input_mask, input_segment_ids, 
                                                           target_ids_, target_mask, target_segment_ids, 
                                                           True
                                                           )
    train_variables = model.trainable_variables
    draft_summary_loss = loss_function(target_ids[:, 1:, :], draft_predictions, mask)
    refine_summary_loss = loss_function(target_ids[:, :-1, :], refine_predictions, mask)
    loss = draft_summary_loss + refine_summary_loss
    #loss = optimizer.get_scaled_loss(loss)
  gradients  = tape.gradient(loss, train_variables)
  #gradients = optimizer.get_unscaled_gradients(gradients)
  # Initialize the shadow variables with same type as the gradients 
  if not accumulators:
    for tv in gradients:
      accumulators.append(tf.Variable(tf.zeros_like(tv), trainable=False))
  # accmulate the gradients to the shadow variables
  for (accumulator, grad) in zip(accumulators, gradients):
    accumulator.assign_add(grad)
  # apply the gradients and reset them to zero if the flag is true
  if grad_accum_flag:
    for accumlator in accumulators:
      accumulator.assign(tf.math.divide(accumulator,h_parms.accumulation_steps))
    optimizer.apply_gradients(zip(accumulators, train_variables))
    for accumulator in (accumulators):
        accumulator.assign(tf.zeros_like(accumulator))
  train_loss(loss)
  train_accuracy(target_ids_[:, :-1], refine_predictions)  
  
@tf.function(input_signature=val_step_signature)
def val_step(input_ids, 
             input_mask, 
             input_segment_ids, 
             target_ids_, 
             target_mask, 
             target_segment_ids, 
             target_ids, 
             mask, 
             epoch, 
             create_summ):
  (draft_predictions, draft_attention_weights, 
   refine_predictions, refine_attention_weights) = model(
                                                         input_ids, input_mask, input_segment_ids, 
                                                         target_ids_, target_mask, target_segment_ids, 
                                                         False
                                                         )
  draft_summary_loss = loss_function(target_ids[:, 1:, :], draft_predictions, mask)
  refine_summary_loss = loss_function(target_ids[:, :-1, :], refine_predictions, mask)
  loss = draft_summary_loss + refine_summary_loss
  validation_loss(loss)
  validation_accuracy(target_ids_[:, :-1], refine_predictions)  
  if create_summ: 
    rouge, bert = tf_write_summary(target_ids_[:, :-1], refine_predictions, epoch)  
  else: 
    rouge, bert = (1.0, 1.0)  
  return (rouge, bert)
  
def check_ckpt(checkpoint_path):
    ckpt = tf.train.Checkpoint(
                               model=model,
                               optimizer=optimizer
                              )
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
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
  for (batch, (input_ids, input_mask, input_segment_ids, target_ids_, target_mask, target_segment_ids)) in enumerate(train_dataset):
  # the target is shifted right during training hence its shape is subtracted by 1
  # not able to do this inside tf.function since it doesn't allow this operation
    if batch > config.start_from_batch:
      mask = tf.math.logical_not(tf.math.equal(target_ids_[:, 1:], 0))
      target_ids = label_smoothing(tf.one_hot(target_ids_, depth=config.input_vocab_size))
      grad_accum_flag = True if (batch+1)%h_parms.accumulation_steps == 0 else False
      train_step(
                 input_ids, 
                 input_mask, 
                 input_segment_ids, 
                 target_ids_, 
                 target_mask, 
                 target_segment_ids, 
                 target_ids, 
                 mask, 
                 grad_accum_flag
                 )
      batch_run_check(
                      batch, 
                      epoch, 
                      start, 
                      train_summary_writer, 
                      train_loss.result(), 
                      train_accuracy.result(), 
                      model
                      )
      if ((batch+1) * h_parms.batch_size) % config.eval_after == 0:
        ckpt_save_path = ck_pt_mgr.save()
    else:
      continue
      #count_recs(batch, epoch, num_of_train_examples)
#       (val_acc, val_loss, rouge_score, bert_score) = calc_validation_loss(
#                                                                           val_dataset, 
#                                                                           epoch+1, 
#                                                                           val_step, 
#                                                                           valid_summary_writer, 
#                                                                           validation_loss, 
#                                                                           validation_accuracy
#                                                                           )
  
#       latest_ckpt+=epoch
#       log.info(
#                model_metrics.format(
#                                     epoch+1, 
#                                     train_loss.result(), 
#                                     train_accuracy.result(),
#                                     val_loss, 
#                                     val_acc,
#                                     rouge_score, 
#                                     bert_score
#                                    )
#               )
#       log.info(epoch_timing.format(epoch + 1, time.time() - start))
#       log.info(checkpoint_details.format(epoch+1, ckpt_save_path))
#       if not monitor_run(
#                          latest_ckpt, 
#                          ckpt_save_path, 
#                          val_loss, 
#                          val_acc, 
#                          bert_score, 
#                          rouge_score, 
#                          valid_summary_writer, 
#                          epoch):
#         break
