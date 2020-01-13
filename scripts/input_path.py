# -*- coding: utf-8 -*-
from bunch import Bunch
import os

#g_drive_path = '/content/drive/My Drive/Client_demo/'  #G_drive
core_path = os.getcwd() 
dataset_name = 'cnn'
g_drive_vocab_path = '/content/drive/My Drive/Client_demo/input_files/Vocab_files'
file_path = {
        'best_ckpt_path' : "/content/drive/My Drive/Text_summarization/BERT_text_summarisation/created_files/training_summarization_model_ckpts/cnn/best_checkpoints",  
        'checkpoint_path' : "/content/drive/My Drive/Text_summarization/BERT_text_summarisation/cnn_checkpoints",
        'G_drive_vocab_path' : g_drive_vocab_path,
        'infer_csv_path' : os.path.join(core_path, "BERT_text_summarisation/input_files/Azure_dataset/Test.csv"),
        'infer_ckpt_path' : "/content/drive/My Drive/Text_summarization/BERT_text_summarisation/cnn_checkpoints/ckpt-1",
        'log_path' : os.path.join(core_path, "created_files/tensorflow.log"),
        'subword_vocab_path' : os.path.join(core_path, "input_files/vocab_file_summarization_"+dataset_name),
        'summary_write_path' : os.path.join(core_path, "created_files/summaries/"+dataset_name+"/"),
        'tensorboard_log' : os.path.join(core_path, "created_files/tensorboard_logs/"+dataset_name+"/"),
        'train_csv_path' : os.path.join(core_path, "input_files/Azure_dataset/Train.csv"),
        
}
file_path = Bunch(file_path)

