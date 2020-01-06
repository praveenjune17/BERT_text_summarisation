# -*- coding: utf-8 -*-
import datetime
import tensorflow as tf
import os
import logging
from hyper_parameters import h_parms
from configuration import config
from input_path import file_path


def check_and_create_dir(path):
    if not os.path.exists(path):
      os.makedirs(path)
      print(f'directory {path} created ')
        
# create folder in input_path if they don't exist
if not config.use_tfds:
  assert (os.path.exists(file_path['train_csv_path'])), 'Training dataset not available'
for key in file_path.keys():
  if key in ['infer_ckpt_path' , 'G_drive_vocab_path', 'subword_vocab_path']:
    pass
  else:
      if os.path.splitext(file_path[key])[1] == '':
        check_and_create_dir(file_path[key])
              
# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler(file_path.log_path)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)
log.propagate = False

if not tf.config.experimental.list_physical_devices('GPU'):
    log.warning("GPU Not available so Running in CPU")

if config.run_tensorboard:
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = file_path.tensorboard_log + current_time + '/train'
    validation_log_dir = file_path.tensorboard_log + current_time + '/validation'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(validation_log_dir)
else:
    train_summary_writer = None
    valid_summary_writer = None
        
# create metrics dict
monitor_metrics = dict()
monitor_metrics['validation_loss'] = None
monitor_metrics['BERT_f1'] = None
monitor_metrics['ROUGE_f1'] = None
monitor_metrics['validation_accuracy'] = None
monitor_metrics['combined_metric'] = (
                                      monitor_metrics['BERT_f1'], 
                                      monitor_metrics['ROUGE_f1'], 
                                      monitor_metrics['validation_accuracy']
                                      )
assert (config.monitor_metric in monitor_metrics.keys()), f'Available metrics to monitor are {monitor_metrics.keys()}'
assert (tf.reduce_sum(h_parms.combined_metric_weights) == 1), 'weights should sum to 1'
