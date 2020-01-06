# -*- coding: utf-8 -*-
import pandas as pd
import os
import tensorflow_datasets as tfds
from input_path import file_path 
from configuration import config
from creates import log

def create_dataframe(path, num_examples):
    df = pd.read_csv(path)
    df.columns = [i.capitalize() for i in df.columns if i.lower() in ['document', 'summary']]
    assert len(df.columns) == 2, 'column names should be document and summary'
    df = df[:num_examples]
    assert not df.isnull().any().any(), 'dataset contains  nans'
    return (df["Document"].values, df["Summary"].values)

if os.path.exists(file_path.subword_vocab_path+'.subwords'):
  tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(file_path.subword_vocab_path)
else:
  try:
    if os.path.split(file_path.subword_vocab_path)[1] + '.subwords' in os.listdir(file_path.G_drive_vocab_path):
      new_vocab_path = os.path.join(file_path.G_drive_vocab_path, os.path.split(file_path.subword_vocab_path)[1])
      tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(new_vocab_path)
  except FileNotFoundError:
    log.warning('Vocab file not available in G-drive, Did you mount G-drive and specify the correct G-drive path? ')
    try:
      os.makedirs(os.path.split(file_path.subword_vocab_path)[0])
    except FileExistsError:
      pass
    log.info('Building vocab file from the training set')
    if config.use_tfds:
      examples, metadata = tfds.load(config.tfds_name, with_info=True, as_supervised=True)
      train_examples = examples['train']
      tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
              (doc.numpy() for doc, _ in train_examples), target_vocab_size=2**13)
    else:
      doc, summ = create_dataframe(file_path.train_csv_path, None)
      tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (doc for doc, _ in zip(doc, summ)), target_vocab_size=2**13)
    tokenizer_en.save_to_file(file_path.subword_vocab_path)
             
log.info('subword vocab file loaded')
assert(tokenizer_en.vocab_size+2 == config.input_vocab_size== config.target_vocab_size),f' *vocab size in configuration script should be {tokenizer_en.vocab_size+2}'
