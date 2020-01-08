import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from configuration import config
from hyper_parameters import h_parms
from bert_tokenization import FullTokenizer
from bert_model import vocab_of_BERT
from abstractive_summarizer import AbstractiveSummarization



draft_summary_model = AbstractiveSummarization(
                                num_layers=config.num_layers, 
                                d_model=config.d_model, 
                                num_heads=config.num_heads, 
                                dff=config.dff, 
                                vocab_size=config.input_vocab_size,
                                input_seq_len=config.doc_length, 
                                output_seq_len=config.summ_length, 
                                add_stage_1=True,
                                add_stage_2=False,
                                rate=h_parms.dropout_rate
                                )

refine_summary_model = AbstractiveSummarization(
                                num_layers=config.num_layers, 
                                d_model=config.d_model, 
                                num_heads=config.num_heads, 
                                dff=config.dff, 
                                vocab_size=config.input_vocab_size,
                                input_seq_len=config.doc_length, 
                                output_seq_len=config.summ_length, 
                                add_stage_1=False,
                                add_stage_2=True,
                                rate=h_parms.dropout_rate
                                )

#bert_layer = draft_summary_model.bert
vocab_file = vocab_of_BERT.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = vocab_of_BERT.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)
del vocab_of_BERT

def create_dataframe(path, num_examples):
    df = pd.read_csv(path)
    df.columns = [i.capitalize() for i in df.columns if i.lower() in ['document', 'summary']]
    assert len(df.columns) == 2, 'column names should be document and summary'
    df = df[:num_examples]
    assert not df.isnull().any().any(), 'dataset contains  nans'
    return (df["Document"].values, df["Summary"].values)
