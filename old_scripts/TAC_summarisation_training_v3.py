# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:08:22 2019

@author: pravech3
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:18:23 2019

@author: pravech3
"""

#!/usr/bin/env python
# coding: utf-8
##########################################################################################
#                       import the below packages
#!pip install tensorflow
#!pip install tensorflow-datasets
#!pip install tensorflow-gan
#!pip install tensorflow-probability
#!pip install tensor2tensor
#!pip install rouge==0.3.2
#!pip install bunch
#!tf_upgrade_v2 --infile c:/Users/pravech3/Summarization/beam_search.py --outfile c:/Users/pravech3/Summarization/beam_search.py
###########################################################################################
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
sys.path.insert(0, 'C:\\TAC\\Scripts\\Training\\')


import tensorflow as tf
tf.random.set_seed(100)
import time
import os
import shutil
#import tensorflow_datasets as tfds
from preprocess import create_train_data
from transformer import Transformer, Generator, create_masks
from hyper_parameters import config
from metrics import optimizer, loss_function, get_loss_and_accuracy, write_summary
from input_path import file_path


if config.run_tensorboard:
    from input_path  import train_summary_writer, valid_summary_writer
else:
    train_summary_writer = None
    valid_summary_writer = None
    

#tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(file_path.subword_vocab_path)

train_dataset, val_dataset = create_train_data()
train_loss, train_accuracy = get_loss_and_accuracy()
validation_loss, validation_accuracy = get_loss_and_accuracy()

transformer = Transformer(
        num_layers=config.num_layers, 
        d_model=config.d_model, 
        num_heads=config.num_heads, 
        dff=config.dff, 
        input_vocab_size=config.input_vocab_size, 
        target_vocab_size=config.target_vocab_size, 
        rate=config.dropout_rate)
generator   = Generator()


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
]

val_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
]

val_step_with_summary_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
]

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
    if config.copy_gen:
      predictions = generator(dec_output, predictions, attention_weights, inp, 
                            inp_shape, tar_shape, batch, training=True)
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

def tf_write_summary(tar_real, predictions, inp, epoch):
  return tf.py_function(write_summary, [tar_real, predictions, inp, epoch], Tout=tf.float32)


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
  

def calc_validation_loss(validation_dataset, epoch):
  validation_loss.reset_states()
  validation_accuracy.reset_states()
  val_acc = 0
  val_loss = 0
  
  for (batch, (inp, tar)) in enumerate(validation_dataset):
    # calculate rouge for only the first batch
    if batch == 0:
        rouge = val_step_with_summary(inp, tar, epoch, inp.shape[1], tar.shape[1]-1, inp.shape[0])
    else:
        #rouge = 'Calculated only for first batch'
        val_step(inp, tar, inp.shape[1], tar.shape[1]-1, inp.shape[0])
    val_loss += validation_loss.result()
    val_acc += validation_accuracy.result()
  return (val_acc.numpy()/(batch+1), val_loss.numpy()/(batch+1), rouge)


def check_ckpt(checkpoint_path):
    ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer,
                           generator=generator)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=20)
    
    # if a checkpoint exists, restore the latest checkpoint.
    if tf.train.latest_checkpoint(checkpoint_path) and not config.from_scratch:
      ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
      print (ckpt_manager.latest_checkpoint, 'checkpoint restored!!')
    else:
        ckpt_manager = tf.train.CheckpointManager(ckpt, file_path.new_checkpoint_path, max_to_keep=20)
        print('Training from scratch')
    return ckpt_manager

def train(checkpoint_path, epochs, batch_size):
    
    ckpt_manager = check_ckpt(checkpoint_path)
    tolerance = config.tolerance
    # get the latest checkpoint from the save directory
    latest_ckpt = int(tf.train.latest_checkpoint(checkpoint_path)[-2:]) 
    for epoch in range(epochs):
      
      start = time.time()  
      train_loss.reset_states()
      train_accuracy.reset_states()
      validation_loss.reset_states()
      validation_accuracy.reset_states()
      
      # inp -> document, tar -> summary
      for (batch, (inp, tar)) in enumerate(train_dataset):
        # the target is shifted right during training hence its shape is subtracted by 1
            #not able to do this inside tf.function 
        train_step(inp, tar, inp.shape[1], tar.shape[1]-1, batch_size)        
        if batch==0 and epoch ==0:
          print('Time taken to feed the input data to the model {} seconds'.format(time.time()-start))
        if batch % config.print_chks == 0:
          print ('Epoch {} Batch {} Train_Loss {:.4f} Train_Accuracy {:.4f}'.format(
              epoch + 1, batch, train_loss.result(), train_accuracy.result()))
      
      (val_acc, val_loss, rouge) = calc_validation_loss(val_dataset, epoch+1)
      ckpt_save_path = ckpt_manager.save()
      ckpt_fold, ckpt_string = os.path.split(ckpt_save_path)
      if config.run_tensorboard:
          with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
            tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch)
          
          with valid_summary_writer.as_default():
            tf.summary.scalar('validation_loss', validation_loss.result(), step=epoch)
            tf.summary.scalar('validation_accuracy', validation_accuracy.result(), step=epoch)
            tf.summary.scalar('validation_total_loss', val_acc, step=epoch)
            tf.summary.scalar('validation_total_accuracy', val_loss, step=epoch)
            tf.summary.scalar('ROUGE', rouge, step=epoch)
    
      if config.verbose:
    
          model_metrics = 'Epoch {}, Train Loss: {:.4f}, Train_Accuracy: {:.4f}, \
                          Valid Loss: {:.4f},                   \
                          Valid Accuracy: {:4f}, ROUGE {}'
          epoch_timing  = 'Time taken for {} epoch : {} secs' 
          checkpoint_details = 'Saving checkpoint for epoch {} at {}'
          
          print(model_metrics.format(epoch+1,
                                 train_loss.result(), 
                                 train_accuracy.result(),
                                 val_loss, 
                                 val_acc,
                                 rouge))
          print(epoch_timing.format(epoch + 1, time.time() - start))
          print(checkpoint_details.format(epoch+1, ckpt_save_path))
      
      if (latest_ckpt > config.look_only_after) and (config.optimum_validation_loss > val_loss):
          config.optimum_validation_loss =  val_loss
          ckpt_files_tocopy = [files for files in os.listdir(os.path.split(ckpt_save_path)[0]) if ckpt_string in files]
          for files in ckpt_files_tocopy:
              print(f'Checkpoint {ckpt_string} copied to the best checkpoint directory')
              print(f'Validation loss is {val_loss}')
              shutil.copy2(os.path.join(ckpt_fold, files), file_path.best_ckpt_path)
      else:
          tolerance+=1
    
      if tolerance > config.tolerance_threshold:
          print('Early stopping since the validation loss is lowering after the \
              specified tolerance threshold')
          break
      
      

if __name__ == "__main__":
    train(checkpoint_path=file_path.old_checkpoint_path, epochs=config.epochs,
          batch_size=config.batch_size)
    
    
'''
!pip install bert_score==0.2.0
from bert_score import score
import time
start = time.time()
with open("bert_score/example/hyps.txt") as f:
    cands = [line.strip() for line in f]

with open("bert_score/example/refs.txt") as f:
    refs = [line.strip() for line in f]
p, r, f1 = score(cands, refs, lang='en', model_type='xlnet-base-cased')
print(time.time()-start)
score(cands, refs, lang='en-sci')# to use SCI BERT

#######################After exam ####################################################################
https://github.com/Tiiiger/bert_score/tree/master/example
create hyper parameter training script
train with beam size output
set hyper parameters based on the two runs
  Create two scripts 
    first  a) overfit and confirm the model is expressive enough to consume the data
    second b) stop training when the model reaches a minimum validation loss
        

gradient accumulation
    https://stackoverflow.com/questions/55268762/how-to-accumulate-gradients-for-large-batch-sizes-in-keras
    https://pypi.org/project/keras-gradient-accumulation/
    
Try :-
    https://www.tensorflow.org/neural_structured_learning
    https://www.tensorflow.org/model_optimization
    
To add:-
    https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
    create test cases to check the correctness of the algorithm.
    train the algorithm to run till minimum train loss and then find out
    code to find out which checkpoint has the max validation accuracy, ROUGE, BERT score
    and min val loss, train loss, move those to the important checkpoints folder
    
    speed test 
    run the model multiple times to check the seed
    print model summary .
    print validation batch and few training batch before training the model
    gradient accumulation
    BERT score (abstractive metric) final test set metric
    create a new metric which is a combination of BERT score, validation accuracy, ROUGE use
        this in the early stopping criterion    
    heuristic to remove the duplicate lines from text
'''