from metrics import convert_wordpiece_to_words
from decode_text import *

def summarize():
  ckpt = tf.train.Checkpoint(
                             model=model
                            )
  ckpt.restore('/content/drive/My Drive/Text_summarization/BERT_text_summarisation/cnn_checkpoints/ckpt-69')
  start = time.time()
  ip_ids = tokenizer.encode(input('Enter the article to be summarized '))
  preds_draft_summary, preds_refined_summary, refine_attention_dist = predict_using_beam_search(tf.convert_to_tensor([ip_ids]),
                                                                                               refine_decoder_sampling_type='topk')
  sum_hyp = tokenizer.convert_ids_to_tokens([i for i in tf.squeeze(preds_refined_summary) if i not in [CLS_ID, SEP_ID, 0]])
  sum_hyp = convert_wordpiece_to_words(sum_hyp)
  print(f'the summarized output is --> {sum_hyp if sum_hyp else "EMPTY"}')
  print(f'Time to process {round(start-time.time())} seconds')

if __name__ == '__main__':
  summarize()
