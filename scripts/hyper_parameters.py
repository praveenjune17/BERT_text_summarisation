# -*- coding: utf-8 -*-
from bunch import Bunch
import tensorflow as tf

hyp = {
	 'accumulation_steps': 2,                                                                                   # TODO
	 'batch_size': 8,
	 'beam_sizes': [2, 3, 4],        # Used only during inference                                                 #TODO for training
	 'combined_metric_weights': [0.4, 0.3, 0.3], #(bert_score, rouge, validation accuracy)
	 'dropout_rate': 0.0,
	 'epochs': 20,
	 'epsilon_ls': 0.0,              # label_smoothing hyper parameter
	 'grad_clipnorm':None,
	 'l2_norm':0,
	 'learning_rate': 3e-4,          # set learning rate decay
	 'length_penalty' : 1,
	 'mean_attention_heads':True,    # if False then the attention weight of the last head will be used
	 }                                    

h_parms = Bunch(hyp)
