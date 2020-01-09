import pandas as pd
from bert_tokenization import FullTokenizer
from train_bert_v3 import model





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
