
import pandas as pd
from bert_tokenization import FullTokenizer
from abstractive_summarizer_v3 import AbstractiveSummarization
from hyper_parameters import h_parms
from configuration import config


model = AbstractiveSummarization(
                                num_layers=config.num_layers, 
                                d_model=config.d_model, 
                                num_heads=config.num_heads, 
                                dff=config.dff, 
                                vocab_size=config.input_vocab_size,
                                output_seq_len=config.summ_length, 
                                rate=h_parms.dropout_rate
                                )

vocab_file = model.vocab_of_BERT.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = model.vocab_of_BERT.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

def create_dataframe(path, num_examples):
    df = pd.read_csv(path)
    df.columns = [i.capitalize() for i in df.columns if i.lower() in ['document', 'summary']]
    assert len(df.columns) == 2, 'column names should be document and summary'
    df = df[:num_examples]
    assert not df.isnull().any().any(), 'dataset contains  nans'
    return (df["Document"].values, df["Summary"].values)
