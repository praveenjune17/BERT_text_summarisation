# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:50:32 2019

@author: pravech3
"""
import tensorflow_datasets as tfds
from input_path import file_path 
from preprocess import create_dataset



# def create_dataset(path, num_examples):
#     df = pd.read_csv(path)
#     df = df[:num_examples]
#     return (df['cisco_technical_team'].values, df['Actions_taken'].values)

doc, summ = create_dataset(file_path.csv_path, None)


def tokenizer(doc, summ):
    try:
        tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(file_path.subword_vocab_path)
    except:
        print(f'creating the subword vocab . This may take some time depending on the training data size')
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
             (doc for doc, _ in zip(doc, summ)), target_vocab_size=2**13)
        tokenizer_en = tokenizer_en.save_to_file(file_path.subword_vocab_path)
    print("Subword Tokenizer created")
if __name__ == "__main__":
    tokenizer(doc, summ)