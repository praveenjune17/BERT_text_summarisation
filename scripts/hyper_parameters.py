# -*- coding: utf-8 -*-
from bunch import Bunch
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

hyp = {
	 'accumulation_steps': 36,                                                                                   
	 'batch_size': 1,
	 'beam_sizes': [2, 3, 4],        	     # Used only during inference                                                 
	 'combined_metric_weights': [0.4, 0.3, 0.3], #(bert_score, rouge, validation accuracy)
	 'dropout_rate': 0.0,
	 'epochs': 4,
	 'epsilon_ls': 0.0,              	     # label_smoothing hyper parameter
	 'grad_clipnorm':None,
	 'l2_norm':0,
	 'learning_rate': None,          	     # change to None to set learning rate decay
	 'length_penalty' : 1,                       # Beam search hyps . Used only during inference                                                 
	 'mean_attention_heads':True,                # if False then the attention parameters of the last head will be used
	 'mean_parameters_of_layers':True,           # if False then the attention parameters of the last layer will be used
	 'validation_batch_size' : 8
	 }                                    

h_parms = Bunch(hyp)
