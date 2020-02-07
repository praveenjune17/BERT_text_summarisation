import tensorflow as tf
import time
from hyper_parameters import h_parms
from configuration import config
from creates import log
from metrics import label_smoothing

train_step_signature = [
                      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                      tf.TensorSpec(shape=(None, None), dtype=tf.bool),
                      tf.TensorSpec(shape=(None, None), dtype=tf.bool),
                      tf.TensorSpec(shape=(None), dtype=tf.bool)
                      ]

val_step_signature = [
                      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None), dtype=tf.bool)
                     ]
  
model_metrics = 'Step {}\n,\
                 Train Loss {:.4f}\n,\
                 Train_Accuracy {:.4f}\n,\
                 validation_accuracy {:4f}\n,\
                 ROUGE_f1 {:4f}\n,\
                 BERT_f1 {:4f}\n'
evaluation_step  = 'Time taken for {} step : {} secs' 
checkpoint_details = 'Saving checkpoint at step {} on {}'
batch_zero = 'Time taken to feed the input data to the model {} seconds'
batch_run_details = 'Step {} Train_Loss {:.4f} Train_Accuracy {:.4f}'

# run every batch
def batch_run_check(batch, start, train_summary_writer, train_loss, train_accuracy, model):
  if config.run_tensorboard:
    with train_summary_writer.as_default():
      tf.summary.scalar('train_loss', train_loss, step=batch)
      tf.summary.scalar('train_accuracy', train_accuracy, step=batch)
  if batch==0:
    log.info(model.summary())
    log.info(batch_zero.format(time.time()-start))
  log.info(
           batch_run_details.format(
                                   batch, 
                                   train_loss, 
                                   train_accuracy
                                   )
          )

# run after each epoch
def count_recs(batch, epoch, num_of_train_examples):
  if epoch == 0:
    try:
      if batch > 0:
        num_of_recs_post_filter_atmost = ((batch)*h_parms.batch_size)/num_of_train_examples
        log.info(f'Percentage of records used for training should be close to {num_of_recs_post_filter_atmost*100 :.2f}')
    except NameError:
      log.info('End of epoch')

def calc_validation_loss(validation_dataset, 
                         step, 
                         val_step, 
                         valid_summary_writer, 
                         validation_accuracy):
  total_val_acc_avg = tf.keras.metrics.Mean()
  for (batch, (input_ids, target_ids_)) in enumerate(validation_dataset.take(config.valid_samples_to_eval//h_parms.validation_batch_size)):
    # calculate rouge for only the first batch
    if batch == 0:
      draft_mask = tf.math.logical_not(tf.math.equal(target_ids_[:, 1:], 0))
      refine_mask = tf.math.logical_not(tf.math.equal(target_ids_[:, :-1], 0))
      target_ids = label_smoothing(tf.one_hot(target_ids_, depth=config.input_vocab_size))
      rouge_score, bert_score = val_step(input_ids,
                                         target_ids_,  
                                         step, 
                                         config.write_summary_op
                                         )
    else:
      _  =  val_step(input_ids,
                     target_ids_, 
                     step, 
                     False
                     )
    if config.run_tensorboard:
      with valid_summary_writer.as_default():
        tf.summary.scalar('validation_accuracy', validation_accuracy.result(), step=batch)
  return (total_val_acc_avg(validation_accuracy.result()), 
          rouge_score, 
          bert_score)
