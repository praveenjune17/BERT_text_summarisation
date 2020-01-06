# -*- coding: utf-8 -*-
import tensorflow as tf
tf.random.set_seed(100)
import time
import os
import numpy as np
from create_tokenizer import tokenizer_en
from transformer import Transformer, create_masks
from hyper_parameters import h_parms
from configuration import config
from input_path import file_path
from beam_search import beam_search
from preprocess import infer_data_from_df
from rouge import Rouge
from bert_score import score as b_score

rouge_all = Rouge()
infer_template = '''Beam size <--- {}\
                    ROUGE-f1  <--- {}\
                    BERT-f1   <--- {}'''
model = Transformer(
                    num_layers=config.num_layers, 
                    d_model=config.d_model, 
                    num_heads=config.num_heads, 
                    dff=config.dff, 
                    input_vocab_size=config.input_vocab_size, 
                    target_vocab_size=config.target_vocab_size, 
                    rate=h_parms.dropout_rate
                    )

def restore_chkpt(checkpoint_path):
    ckpt = tf.train.Checkpoint(
                               model=model
                               )
    assert tf.train.latest_checkpoint(os.path.split(checkpoint_path)[0]), 'Incorrect checkpoint direcotry'
    ckpt.restore(checkpoint_path).expect_partial()
    print(f'{checkpoint_path} restored')

def beam_search_eval(document, beam_size):
  
  start = [tokenizer_en.vocab_size] 
  end = [tokenizer_en.vocab_size+1]
  encoder_input = tf.tile(document, multiples=[beam_size, 1])
  batch, inp_shape = encoder_input.shape
  def decoder_query(output):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                                                                     encoder_input, 
                                                                     output
                                                                     )
    predictions, attention_weights, dec_output = model(
                                                       encoder_input, 
                                                       output,
                                                       enc_padding_mask,
                                                       combined_mask,
                                                       dec_padding_mask,
                                                       False
                                                       )

    # (batch_size, 1, target_vocab_size)
    return (predictions[:,-1:,:])  
  return beam_search(
                     decoder_query, 
                     start, 
                     beam_size, 
                     config.summ_length, 
                     config.input_vocab_size, 
                     h_parms.length_penalty, 
                     stop_early=True, 
                     eos_id=[end]
                    )

def run_inference(dataset, beam_sizes_to_try=h_parms.beam_sizes):
    for beam_size in beam_sizes_to_try:
      total_summary = []
      for (doc_id, (document, summary)) in enumerate(dataset, 1):
        start_time = time.time()
        # translated_output_temp[0] (batch, beam_size, summ_length+1)
        translated_output_temp = beam_search_eval(document, beam_size)
        sum_ref = tokenizer_en.decode([j for j in tf.squeeze(summary) if j < tokenizer_en.vocab_size])
        sum_hyp = tokenizer_en.decode([j for j in tf.squeeze(translated_output_temp[0][:,0,:]) if j < tokenizer_en.vocab_size])
        total_summary.append((sum_ref, sum_hyp))
        print('Original summary: {}'.format(tokenizer_en.decode([j for j in tf.squeeze(summary) if j < tokenizer_en.vocab_size])))
        print('Predicted summary: {}'.format(tokenizer_en.decode([j for j in tf.squeeze(translated_output_temp[0][:,0,:]) if j < tokenizer_en.vocab_size])))
      ref_sents = [ref for ref, _ in total_summary]
      hyp_sents = [hyp for _, hyp in total_summary]
      rouges = rouge_all.get_scores(ref_sents , hyp_sents)
      avg_rouge_f1 = np.mean([np.mean([rouge_scores['rouge-1']["f"], rouge_scores['rouge-2']["f"], rouge_scores['rouge-l']["f"]]) for rouge_scores in rouges])
      _, _, bert_f1 = b_score(ref_sents, hyp_sents, lang='en', model_type='bert-base-uncased')
      print(infer_template.format(beam_size, avg_rouge_f1, np.mean(bert_f1.numpy())))
      print(f'time to process document {doc_id} : {time.time()-start_time}') 

if __name__ == '__main__':
  #Restore the model's checkpoints
  restore_chkpt(file_path.infer_ckpt_path)
  infer_dataset = infer_data_from_df()
  run_inference(infer_dataset)
