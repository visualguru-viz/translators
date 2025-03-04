import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
import pickle

model_en_fr = torch.load('model_en_fr.pkl')
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

st.title("Translator: English - French")
text_to_translate = st.text_input("English")

trans_text_encode = model_en_fr.generate(**tokenizer.prepare_seq2seq_batch([text_to_translate],return_tensors='pt'))
translated_txt = tokenizer.batch_decode(trans_text_encode)

st.subheader("French:")
fr_translated = translated_txt[0]
fr_translated = fr_translated.replace("<pad>", "").replace("</s>", "")
st.write(fr_translated)



model_en_de = torch.load('model_en_de.pkl')
de_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")


de_trans_text_encode = model_en_de.generate(**de_tokenizer.prepare_seq2seq_batch([text_to_translate],return_tensors='pt'))
de_translated_txt = de_tokenizer.batch_decode(de_trans_text_encode)

st.subheader("German:")
de_translated = de_translated_txt[0]
de_translated = de_translated.replace("<pad>", "").replace("</s>", "")
st.write(de_translated)


model_en_es = torch.load('model_en_es.pkl')
es_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")


es_trans_text_encode = model_en_es.generate(**es_tokenizer.prepare_seq2seq_batch([text_to_translate],return_tensors='pt'))
es_translated_txt = es_tokenizer.batch_decode(es_trans_text_encode)

st.subheader("Spanish:")
es_translated = es_translated_txt[0]
es_translated = es_translated.replace("<pad>", "").replace("</s>", "")
st.write(es_translated)
