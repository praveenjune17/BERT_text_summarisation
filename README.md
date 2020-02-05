# Text_summarisation using BERT

This Project is inspired from https://arxiv.org/pdf/1902.09243v2.pdf.  
Part of the code adapted from https://github.com/raufer/bert-summarization.  
Created using TensorFlow 2  

Apart from the existing functionalities in the adapted code added few features like   
  *)Added Beam-search mechanism during inference  
  *)Added Copy mechanism to the decoder  
  *)Added topk, nucleus decoders  
  *)Used Huggingface Transformers library to extract BERT embeddings  
  *)Mixed precision policy enabled training
  *)Migrated the adapted code from Tensorflow 1 to 2 
  

Instructions to Train the model  
  a) Run train_bert_summarizer_mixed_precision.py if you have GPU with compute compatibility >= 7.5 else use train_bert_summarizer.py.
  
*) Google colab Demo available [here](https://github.com/praveenjune17/BERT_text_summarisation/blob/master/Text_summarization_demo_using_BERT.ipynb)
