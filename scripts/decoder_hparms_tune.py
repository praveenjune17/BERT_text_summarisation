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
from tqdm import tqdm

UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103
rouge_all = Rouge()
infer_template = '''Batch size <--- 20\nDraft_decoder_type <--- {}\nRefine_decoder_type <--- {}\nROUGE-f1  <--- {}\nBERT-f1   <--- {}\nNucleus's p   <--- {}\nTopk's K  <--- {}\ntemperature   <--- {}\nbeam-size   <--- {}'''
h_parms.batch_size = 40


def restore_chkpt(checkpoint_path):
    ckpt = tf.train.Checkpoint(
                               model=model
                               )
    assert tf.train.latest_checkpoint(os.path.split(checkpoint_path)[0]), 'Incorrect checkpoint directory'
    ckpt.restore(checkpoint_path).expect_partial()
    print(f'{checkpoint_path} restored')
restore_chkpt('/content/drive/My Drive/Text_summarization/BERT_text_summarisation/created_files/training_summarization_model_ckpts/cnn/best_checkpoints/ckpt-43')
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
  test_dataset = test_dataset.take(1)
else:
  test_dataset = infer_data_from_df()
ref_sents=[]
hyp_sents=[]

for draft_dec_type in ['beam_search', 'topktopp', 'topk', 'nucleus' 'random_sampling', 'greedy']:
  for refine_dec_type in ['topktopp', 'topk', 'nucleus' 'random_sampling', 'greedy']:
    for k in [3, 4, 5, 6, 7, 10, 13, 18, 25]:
      for p in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for beam_size in [2, 3, 4, 5]:
          for temperature in [0.6, 0.7, 0.8, 0.9, 1]:
            for (doc_id, (input_ids, _, _, target_ids, _, _)) in tqdm(enumerate(test_dataset, 1)):
              start_time = time.time()
              if draft_dec_type == 'beam_search':
                draft, refined_summary, att = predict_using_beam_search(
                                                                        input_ids, 
                                                                        beam_size=beam_size, 
                                                                        refine_decoder_sampling_type=refine_dec_type, 
                                                                        k=k, 
                                                                        p=p,
                                                                        temperature=temperature
                                                                        )
              else:
                _, _, refined_summary, _ = predict_using_sampling(
                                                                  input_ids,
                                                                  draft_decoder_sampling_type=draft_dec_type, 
                                                                  refine_decoder_sampling_type=refine_dec_type, 
                                                                  k=k,
                                                                  p=p,
                                                                  temperature=temperature
                                                                  )
              for tar, ref_hyp in zip(target_ids, refined_summary):
                sum_ref = tokenizer.convert_ids_to_tokens([i for i in tf.squeeze(tar) if i not in [0, 101, 102]])
                sum_hyp = tokenizer.convert_ids_to_tokens([i for i in tf.squeeze(ref_hyp) if i not in [0, 101, 102]])
                sum_ref = convert_wordpiece_to_words(sum_ref)
                sum_hyp = convert_wordpiece_to_words(sum_hyp)
                #print('Original summary: {}'.format(sum_ref))
                #print('Predicted summary: {}'.format(sum_hyp))
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
            print(infer_template.format(draft_dec_type, refine_dec_type, avg_rouge_f1, avg_bert_f1, p, k, temperature, beam_size))
            print(f'time to process document {doc_id} : {time.time()-start_time}')
            print(f'Calculating scores for {len(ref_sents)} golden summaries and {len(hyp_sents)} predicted summaries')
