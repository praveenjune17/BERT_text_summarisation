import tensorflow as tf

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
]

val_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
]

val_step_with_summary_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
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
