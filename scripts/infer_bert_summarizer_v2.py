# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/content/BERT_text_summarisation/scripts')
import tensorflow as tf
tf.random.set_seed(100)
import tensorflow_datasets as tfds
import numpy as np
import os
from create_tokenizer import tokenizer
from preprocess import map_batch_shuffle
from configuration import config
from input_path import file_path
from preprocess import infer_data_from_df
from metrics import convert_wordpiece_to_words
from rouge import Rouge
from bert_score import score as b_score
from decode_text import *

UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103
rouge_all = Rouge()
infer_template = '''Draft_decoder_type <--- {}\nRefine_decoder_type <--- {}\nROUGE-f1  <--- {}\nBERT-f1   <--- {}'''



def restore_chkpt(checkpoint_path):
    ckpt = tf.train.Checkpoint(
                               model=model
                               )
    assert tf.train.latest_checkpoint(os.path.split(checkpoint_path)[0]), 'Incorrect checkpoint directory'
    ckpt.restore(checkpoint_path).expect_partial()
    print(f'{checkpoint_path} restored')

draft_and_refine_decoder_combinations = [
                                         ('greedy', 'greedy'), 
                                         ('greedy', 'topk'), 
                                         ('topk', 'greedy'), 
                                         ('topk', 'topk') ,
                                         ('greedy', 'nucleus') ,
                                         ('nucleus', 'greedy') ,
                                         ('topk', 'nucleus') ,
                                         ('nucleus', 'topk') ,
                                         ('nucleus', 'nucleus') ,
                                         ('beam_search', 'greedy') ,
                                         ('beam_search', 'topk') ,
                                         ('beam_search', 'nucleus') ,
                                          ]

# Beam size is set to 3 by default
# Other hyperparameters include temperature, p (nucleus sampling) and k (top k sampling)
# Please refer decode_text script
def run_inference(dataset, print_output=False):

  for draft_type, refine_type in draft_and_refine_decoder_combinations:
    ref_sents = []
    hyp_sents = []
    for (doc_id, (input_ids, _, _, target_ids, _, _)) in enumerate(dataset, 1):
      start_time = time.time()
      if draft_type != 'beam_search':
          _, _, refined_summary, _ = predict_using_sampling(input_ids, draft_type, refine_type, k=10)
      else:
          _, refined_summary, _ = predict_using_beam_search(input_ids, refine_decoder_sampling_type=refine_type)
      sum_ref = tokenizer.convert_ids_to_tokens([i for i in tf.squeeze(target_ids) if i not in [0, 101, 102]])
      sum_hyp = tokenizer.convert_ids_to_tokens([i for i in tf.squeeze(refined_summary) if i not in [0, 101, 102]])
      sum_ref = convert_wordpiece_to_words(sum_ref)
      sum_hyp = convert_wordpiece_to_words(sum_hyp)
      if print_output:
        print('Original summary: {}'.format(sum_ref))
        print('Predicted summary: {}'.format(sum_hyp))
      ref_sents.append(sum_ref)
      hyp_sents.append(sum_hyp)
    print(f'Calculating scores for {len(ref_sents)} golden summaries and {len(hyp_sents)} predicted summaries')
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
    print(infer_template.format(draft_type, refine_type, avg_rouge_f1, avg_bert_f1))
    print(f'time to process document {doc_id} : {time.time()-start_time}') 

if __name__ == '__main__':
  #Restore the model's checkpoints
  #restore_chkpt('/content/drive/My Drive/Text_summarization/BERT_text_summarisation/cnn_checkpoints/ckpt-43')
  restore_chkpt(file_path.infer_ckpt_path)
  if config.use_tfds:
    examples, metadata = tfds.load(
                                   config.tfds_name, 
                                   with_info=True, 
                                   as_supervised=True, 
                                   data_dir='/content/drive/My Drive/Text_summarization/cnn_dataset'
                                   )
    test_examples = examples['test']
    test_buffer_size = metadata.splits['test'].num_examples
    test_dataset = map_batch_shuffle(
                                     test_examples, 
                                     test_buffer_size, 
                                     split='test',
                                     batch_size=h_parms.batch_size
                                     )
    log.info('Test TF_dataset created')
    # Number of samples to use
    test_dataset = test_dataset.take(50)
  else:
    test_dataset = infer_data_from_df()
  run_inference(test_dataset)
