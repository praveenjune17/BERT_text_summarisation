%tensorflow_version 2.x
import sys
sys.path.insert(0, '/content/BERT_text_summarisation/scripts')
from metrics import convert_wordpiece_to_words
from decode_text import *
from create_tokenizer import model

h_parms.validation_batch_size=1
def summarize():
  ckpt = tf.train.Checkpoint(
                             model=model
                            )
  ckpt.restore('/content/drive/My Drive/Text_summarization/BERT_text_summarisation/created_files/training_summarization_model_ckpts/cnn/best_checkpoints/ckpt-113').expect_partial()
  start = time.time()
  ip_ids = tokenizer.encode(input('Enter the article to be summarized '))
  preds_draft_summary, preds_refined_summary, refine_attention_dist = predict_using_beam_search(model,
                                                                                               tf.convert_to_tensor([ip_ids]),
                                                                                               refine_decoder_sampling_type='topp-topk')
  sum_hyp = tokenizer.convert_ids_to_tokens([i for i in tf.squeeze(preds_refined_summary) if i not in [CLS_ID, SEP_ID, 0]])
  sum_hyp = convert_wordpiece_to_words(sum_hyp)
  print(f'the summarized output is --> {sum_hyp if sum_hyp else "EMPTY"}')
  print(f'Time to process {round(start-time.time())} seconds')

if __name__ == '__main__':
  summarize()
