# -*- coding: utf-8 -*-
from bunch import Bunch
from input_path import file_path

hyp = {
     'copy_gen':True,
     'create_hist':False,             # create histogram of summary length and # of tokens per batch
     'doc_length': 1268,
     'd_model': 512,                  # the projected word vector dimension
     'dff': 512,                      # feed forward network hidden parameters
     'early_stop' : True,
     'init_tolerance':0,
     'input_vocab_size': 8263,        # total vocab size + start and end token
     'last_recorded_value': None,
     'monitor_metric' : 'combined_metric',
     'monitor_only_after': 1,        # monitor the monitor_metric only after this epoch                                           
     'max_tokens_per_line' : 400,   # 1763 = 90 percentile  
     'num_examples_to_train': None,   #If None then all the examples in the dataset will be used to train
     'num_examples_to_infer': None,
     'num_heads': 8,                  # the number of heads in the multi-headed attention unit
     'num_layers': 3,                 # number of transformer blocks
     'print_chks': 50,                # print training progress per number of batches specified
     'run_tensorboard': True,
     'show_detokenized_samples' : False,
     'summ_length': 91,
     'target_vocab_size': 8263,       # total vocab size + start and end token
     'test_size': 0.05,               # test set split size
     'tfds_name' : 'cnn_dailymail',   # tfds dataset to be used
     'tolerance_threshold': 5,        # tolerance counter used for early stopping
     'use_tfds' : True,               # use tfds datasets as input to the model 
     'write_per_epoch': 1,            # write summary for every specified epoch
     'write_summary_op': True         # write validation summary to hardisk
     }                                    

config = Bunch(hyp)

#Parse log and get last_recorded_value of monitor_metric
try:
  with open(file_path.log_path) as f:
    for line in reversed(f.readlines()):
        if ('- tensorflow - INFO - '+ config.monitor_metric in line) and \
          (line[[i for i,char in enumerate((line)) if char.isdigit()][-1]+1] == '\n'):          
          config['last_recorded_value'] = float(line.split(config.monitor_metric)[1].split('\n')[0].strip())
          print(f"last_recorded_value of {config.monitor_metric} retained from last run {config['last_recorded_value']}")
          break
        else:
          continue
    if not config['last_recorded_value']:
      print('setting default value to last_recorded_value')
      config['last_recorded_value'] = 0 if config.monitor_metric != 'validation_loss' else float('inf')
except FileNotFoundError:
  print('setting default value to last_recorded_value')
  config['last_recorded_value'] = 0 if config.monitor_metric != 'validation_loss' else float('inf')
