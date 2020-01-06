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
import tensorflow_datasets as tfds
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
    

tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(file_path.subword_vocab_path)

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



def tf_write_summary(tar_real, predictions, inp, epoch):
  return tf.py_function(write_summary, [tar_real, predictions, inp, epoch], Tout=tf.float32)



val_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
]

@tf.function(input_signature=val_step_signature)
def val_step(inp, tar, epoch, inp_shape, tar_shape, batch):
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
    rouge = val_step(inp, tar, epoch, inp.shape[1], tar.shape[1]-1, inp.shape[0])
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

def train(checkpoint_path, epochs):
    
    ckpt_manager = check_ckpt(checkpoint_path)

    for epoch in range(epochs):
      
      start = time.time()  
      train_loss.reset_states()
      train_accuracy.reset_states()
      validation_loss.reset_states()
      validation_accuracy.reset_states()
      
      # inp -> document, tar -> summary
      for (batch, (inp, tar)) in enumerate(train_dataset):
        # the target is shifted right during training hence its shape is subtracted by 1
        train_step(inp, tar, inp.shape[1], tar.shape[1]-1, inp.shape[0])
        
        if batch==0 and epoch ==0:
          print('Time taken to feed the input data to the model {} seconds'.format(time.time()-start))
        if batch % config.print_chks == 0:
          print ('Epoch {} Batch {} Train_Loss {:.4f} Train_Accuracy {:.4f}'.format(
              epoch + 1, batch, train_loss.result(), train_accuracy.result()))
      
      (val_acc, val_loss, rouge) = calc_validation_loss(val_dataset, epoch+1)
      ckpt_save_path = ckpt_manager.save()
      
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
      
      model_metrics = 'Epoch {}, Train Loss: {:.4f}, Train_Accuracy: {:.4f}, \
                      Valid Loss: {:.4f},                   \
                      Valid Accuracy: {:4f}, ROUGE {:.4f}'
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
      
      

if __name__ == "__main__":
    train(checkpoint_path=file_path.old_checkpoint_path, epochs=config.epochs)
    
    
'''    
deconstruct generator
calculate tokens_size per batch
'''