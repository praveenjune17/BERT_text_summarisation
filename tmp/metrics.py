# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import shutil
import os
from configuration import config
from hyper_parameters import h_parms
from rouge import Rouge
from input_path import file_path
from create_tokenizer import tokenizer
from bert_score import score as b_score
from creates import log, monitor_metrics

log.info('Loading Pre-trained BERT model for BERT SCORE calculation')
_, _, _ = b_score(["I'm Batman"], ["I'm Spiderman"], lang='en', model_type=config.pretrained_bert_model)
rouge_all = Rouge()



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def label_smoothing(inputs, epsilon=h_parms.epsilon_ls):
    # number of channels
    V = inputs.get_shape().as_list()[-1] 
    epsilon = tf.cast(epsilon, dtype=inputs.dtype)
    V = tf.cast(V, dtype=inputs.dtype)
    return ((1-epsilon) * inputs) + (epsilon / V)

def convert_wordpiece_to_words(w_piece):
  new=[]
  for i in w_piece:
    if '##' in i:
      m = i.replace('##', '')
    else:
      if w_piece.index(i) == 0:
        m = i
      else:
        m = ' '+i
    new.append(m)
  return (''.join(new))

def loss_function(real, pred, mask):
  # pred shape == real shape = (batch_size, tar_seq_len, target_vocab_size)
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return loss_

def get_loss_and_accuracy():
    loss = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    return(loss, accuracy)
    
def write_summary(tar_real, predictions, step, write=config.write_summary_op):
  ref_sents=[]
  hyp_sents=[]
  for tar, ref_hyp in zip(tar_real, predictions):
      sum_ref = tokenizer.convert_ids_to_tokens([i for i in tf.squeeze(tar) if i not in [0, 101, 102]])
      sum_hyp = tokenizer.convert_ids_to_tokens([i for i in tf.squeeze(ref_hyp) if i not in [0, 101, 102]])
      sum_ref = convert_wordpiece_to_words(sum_ref)
      sum_hyp = convert_wordpiece_to_words(sum_hyp)
      ref_sents.append(sum_ref)
      hyp_sents.append(sum_hyp)
  try:
    rouges = rouge_all.get_scores(ref_sents , hyp_sents)
    avg_rouge_f1 = np.mean([np.mean([rouge_scores['rouge-1']["f"], 
                                    rouge_scores['rouge-2']["f"], 
                                    rouge_scores['rouge-l']["f"]]) for rouge_scores in rouges])
    _, _, bert_f1 = b_score(ref_sents, hyp_sents, lang='en', model_type=config.pretrained_bert_model)
    avg_bert_f1 = np.mean(bert_f1.numpy())
  except:
    avg_rouge_f1 = 0
    avg_bert_f1 = 0
  
  if write and (step)%config.write_per_step == 0:
    with tf.io.gfile.GFile(file_path.summary_write_path+str(step.numpy()), 'w') as f:
      for ref, hyp in zip(ref_sents, hyp_sents):
        f.write(ref+'\t'+hyp+'\n')
  return (avg_rouge_f1, avg_bert_f1)
  
  
def tf_write_summary(tar_real, predictions, step):
  return tf.py_function(write_summary, [tar_real, predictions, step], Tout=[tf.float32, tf.float32])
    

def monitor_run(latest_ckpt, 
                ckpt_save_path, 
                val_acc,
                bert_score, 
                rouge_score, 
                valid_summary_writer,
                step,
                to_monitor=config.monitor_metric):
  
  ckpt_fold, ckpt_string = os.path.split(ckpt_save_path)
  if config.run_tensorboard:
    with valid_summary_writer.as_default():
      tf.summary.scalar('validation_total_accuracy', val_acc, step=step)
      tf.summary.scalar('ROUGE_f1', rouge_score, step=step)
      tf.summary.scalar('BERT_f1', bert_score, step=step)
  monitor_metrics = dict()
  monitor_metrics['validation_accuracy'] = val_acc
  monitor_metrics['BERT_f1'] = bert_score
  monitor_metrics['ROUGE_f1'] = rouge_score
  monitor_metrics['combined_metric'] = (
                                        monitor_metrics['BERT_f1'], 
                                        monitor_metrics['ROUGE_f1'], 
                                        monitor_metrics['validation_accuracy']
                                        )
  # multiply with the weights                                    
  monitor_metrics['combined_metric'] = round(tf.reduce_sum([(i*j) for i,j in zip(monitor_metrics['combined_metric'],  
                                                                                 h_parms.combined_metric_weights)]).numpy(), 2)
  log.info(f"combined_metric {monitor_metrics['combined_metric']:4f}")
  cond = (config.last_recorded_value < monitor_metrics[to_monitor])
  if (latest_ckpt > config.monitor_only_after) and cond:
    # reset tolerance to zero if the monitor_metric decreases before the tolerance threshold
    config.init_tolerance=0
    config.last_recorded_value =  monitor_metrics[to_monitor]
    ckpt_files_tocopy = [files for files in os.listdir(os.path.split(ckpt_save_path)[0]) \
                         if ckpt_string in files]
    log.info(f'{to_monitor} is {monitor_metrics[to_monitor]:4f} so checkpoint files {ckpt_string}           \
             will be copied to best checkpoint directory')
    # copy the best checkpoints
    shutil.copy2(os.path.join(ckpt_fold, 'checkpoint'), file_path.best_ckpt_path)
    for files in ckpt_files_tocopy:
        shutil.copy2(os.path.join(ckpt_fold, files), file_path.best_ckpt_path)
  else:
    config.init_tolerance+=1
  # Warn and early stop
  if config.init_tolerance > config.tolerance_threshold:
    log.warning('Tolerance exceeded')
  if config.early_stop and config.init_tolerance > config.tolerance_threshold:
    log.info(f'Early stopping since the {to_monitor} reached the tolerance threshold')
    return False
  else:
    return True

lr = h_parms.learning_rate if h_parms.learning_rate else CustomSchedule(config.d_model)    

if h_parms.grad_clipnorm:
  optimizer = tf.keras.optimizers.Adam(
                             learning_rate=lr, 
                             beta_1=0.9, 
                             beta_2=0.98, 
                             clipnorm=h_parms.grad_clipnorm,
                             epsilon=1e-9
                             )
else:
  optimizer = tf.keras.optimizers.Adam(
                             learning_rate=lr, 
                             beta_1=0.9, 
                             beta_2=0.98, 
                             epsilon=1e-9
                             )

loss_object = tf.keras.losses.CategoricalCrossentropy(
                                                      from_logits=True, 
                                                      reduction='none'
                                                      )
