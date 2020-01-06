import tensorflow as tf
import time
from configuration import config
from creates import log

train_step_signature = [
                      tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                      tf.TensorSpec(shape=(None, None), dtype=tf.int64)
                      ]

val_step_signature = [
                      tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                      tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                      tf.TensorSpec(shape=(None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None), dtype=tf.bool)
                     ]
  
model_metrics = 'Epoch {}\n,\
                 Train Loss {:.4f}\n,\
                 Train_Accuracy {:.4f}\n,\
                 validation_loss {:.4f}\n,\
                 validation_accuracy {:4f}\n,\
                 ROUGE_f1 {:4f}\n,\
                 BERT_f1 {:4f}\n'
epoch_timing  = 'Time taken for {} epoch : {} secs' 
checkpoint_details = 'Saving checkpoint for epoch {} at {}'
batch_zero = 'Time taken to feed the input data to the model {} seconds'
batch_run_details = 'Epoch {} Batch {} Train_Loss {:.4f} Train_Accuracy {:.4f}'


# run every batch
def batch_run_check(batch, epoch, start, train_summary_writer, train_loss, train_accuracy, transformer):
  if config.run_tensorboard:
    with train_summary_writer.as_default():
      tf.summary.scalar('train_loss', train_loss, step=batch)
      tf.summary.scalar('train_accuracy', train_accuracy, step=batch)
  if batch==0 and epoch ==0:
    log.info(transformer.summary())
    log.info(batch_zero.format(time.time()-start))
  if batch % config.print_chks == 0:
    log.info(
             batch_run_details.format(
                                     epoch + 1, 
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
                         epoch, 
                         val_step, 
                         valid_summary_writer, 
                         validation_loss, 
                         validation_accuracy):
  total_val_acc_avg = tf.keras.metrics.Mean()
  total_val_loss_avg = tf.keras.metrics.Mean()
  for (batch, (inp, tar)) in enumerate(validation_dataset):
    # calculate rouge for only the first batch
    if batch == 0:
      rouge_score, bert_score = val_step(inp, tar, epoch, config.write_summary_op)
    else:
      _ = val_step(inp, tar, epoch, False)
    if config.run_tensorboard:
      with valid_summary_writer.as_default():
        tf.summary.scalar('validation_loss', validation_loss.result(), step=batch)
        tf.summary.scalar('validation_accuracy', validation_accuracy.result(), step=batch)
  return (total_val_acc_avg(validation_accuracy.result()), 
          total_val_loss_avg(validation_loss.result()), 
          rouge_score, 
          bert_score)
