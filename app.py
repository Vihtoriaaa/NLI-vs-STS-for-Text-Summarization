import streamlit as st
import tensorflow as tf
import torch
import numpy as np
import scipy.sparse
import pickle
from keras.models import load_model
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

from sentence_transformers import SentenceTransformer

def perform_nli(model, tokenizer, text):
    sentences = text.split('.')
    resulting_summary = sentences[0].strip() if sentences else ''
    result = []
    for i in range(1, len(sentences)):
        sentence = sentences[i].strip()
        # Skip empty sentences
        if not sentence:
            continue
        is_entailed = any(classify_relationship(model, tokenizer, prev_sentence, sentence)[0] == 'entailment' for prev_sentence in resulting_summary.split('.'))
        if not is_entailed:
            if resulting_summary.endswith('.'):
                resulting_summary += ' ' + sentence
            else:
                resulting_summary += '. ' + sentence
            result.append((sentence, 'added', 1.0))
        else:
            result.append((sentence, 'entailed', 1.0))

    return resulting_summary, result

def classify_relationship(model, tokenizer, sentence1, sentence2):
    premise_id = tokenizer.encode(sentence1, add_special_tokens=False)
    hypothesis_id = tokenizer.encode(sentence2, add_special_tokens=False)
    pair_token_ids = [tokenizer.cls_token_id] + premise_id + [tokenizer.sep_token_id] + hypothesis_id + [tokenizer.sep_token_id]

    segment_ids = torch.tensor([0] * (len(premise_id) + 2) + [1] * (len(hypothesis_id) + 1))
    attention_mask_ids = torch.tensor([1] * (len(premise_id) + len(hypothesis_id) + 3))

    token_ids = torch.tensor(pair_token_ids).unsqueeze(0)
    seg_ids = segment_ids.unsqueeze(0)
    mask_ids = attention_mask_ids.unsqueeze(0)

    # Generate the model's output logits
    with torch.no_grad():
        outputs = model(token_ids, attention_mask=mask_ids, token_type_ids=seg_ids)
        logits = outputs.logits

    # Predict the relationship using softmax probabilities
    probabilities = torch.softmax(logits, dim=1).tolist()[0]

    # Get the predicted label index and corresponding class
    predicted_label_index = probabilities.index(max(probabilities))
    label_map = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
    predicted_label = label_map[predicted_label_index]

    return predicted_label, max(probabilities)


def classify_relationship_sts(model, tokenizer, sentence1, sentence2):
    premise_id = tokenizer.encode(sentence1, add_special_tokens=False)
    hypothesis_id = tokenizer.encode(sentence2, add_special_tokens=False)
    pair_token_ids = [tokenizer.cls_token_id] + premise_id + [tokenizer.sep_token_id] + hypothesis_id + [tokenizer.sep_token_id]
    segment_ids = torch.tensor([0] * (len(premise_id) + 2) + [1] * (len(hypothesis_id) + 1))
    attention_mask_ids = torch.tensor([1] * (len(premise_id) + len(hypothesis_id) + 3))
    token_ids = torch.tensor(pair_token_ids).unsqueeze(0)
    seg_ids = segment_ids.unsqueeze(0)
    mask_ids = attention_mask_ids.unsqueeze(0)

    with torch.no_grad():
        outputs = model(token_ids, attention_mask=mask_ids, token_type_ids=seg_ids)
        logits = outputs.logits
    similarity_score = logits.squeeze().item()
    return similarity_score

def perform_sts(model, tokenizer, text, threshold=0.6):
    sentences = text.split('.')
    resulting_summary = sentences[0].strip() if sentences else ''
    result = []
    for i in range(1, len(sentences)):
        sentence = sentences[i].strip()
        if not sentence:
            continue
        similarity_score = classify_relationship_sts(model, tokenizer, resulting_summary, sentence)
        if similarity_score < threshold:
            if resulting_summary.endswith('.'):
                resulting_summary += ' ' + sentence
            else:
                resulting_summary += '. ' + sentence
            result.append((sentence, 'added', similarity_score))
        else:
            result.append((sentence, 'entailed', similarity_score))
    return resulting_summary, result

def main_bert(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    nli_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    nli_model.load_state_dict(torch.load('models/bert-snli.pt', map_location=torch.device('cpu')))
    nli_model.eval()

    model_sts = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
    model_sts.load_state_dict(torch.load('models/bert-sts.pt', map_location=torch.device('cpu')))
    model_sts.eval()
    # if st.button("Summarize"):
        # Call your NLI and STS models
    nli_result, results = perform_nli(nli_model, tokenizer, text)
    sts_result, results = perform_sts(model_sts, tokenizer, text, threshold)

        # Display the results
    st.subheader("NLI Model Result:")
    st.text_area("NLI Summary", value=nli_result, height=200)

        # Display the results
    st.subheader("STS Model Result:")
    st.text_area("STS Summary", value=sts_result, height=200)

def perform_nli_bilstm(sbert_model, nli_model, text):
    sentences = text.split('.')
    resulting_summary = sentences[0].strip() if sentences else ''
    result = []
    for i in range(1, len(sentences)):
        sentence = sentences[i].strip()
        # Skip empty sentences
        if not sentence:
            continue
        is_entailed = any(classify_relationship_bilstm_nli(nli_model, sbert_model, prev_sentence, sentence)[0] == 'entailment' for prev_sentence in resulting_summary.split('.'))
        if not is_entailed:
            if resulting_summary.endswith('.'):
                resulting_summary += ' ' + sentence
            else:
                resulting_summary += '. ' + sentence
            result.append((sentence, 'added', 1.0))
        else:
            result.append((sentence, 'entailed', 1.0))
    return resulting_summary

def perform_sts_bilstm(sbert_model, nli_model, text, threshold=0.6):
    sentences = text.split('.')
    resulting_summary = sentences[0].strip() if sentences else ''
    result = []
    for i in range(1, len(sentences)):
        sentence = sentences[i].strip()
        if not sentence:
            continue
        similarity_score = classify_sts_bilstm(nli_model, sbert_model, resulting_summary, sentence)
        if similarity_score < threshold:
            if resulting_summary.endswith('.'):
                resulting_summary += ' ' + sentence
            else:
                resulting_summary += '. ' + sentence
            result.append((sentence, 'added', similarity_score))
        else:
            result.append((sentence, 'entailed', similarity_score))
    return resulting_summary

def classify_relationship_bilstm_nli(nli_model, sbert_model, sentence1, sentence2):
    sentences = [sentence1, sentence2]
    embeddings = sbert_model.encode(sentences)
    premise_padded, hypothesis_padded = embeddings[0], embeddings[1]
    premise_padded = tf.convert_to_tensor(premise_padded)
    hypothesis_padded = tf.convert_to_tensor(hypothesis_padded)

    if premise_padded.ndim == 1:
        premise_padded = tf.expand_dims(premise_padded, 0)
    if hypothesis_padded.ndim == 1:
        hypothesis_padded = tf.expand_dims(hypothesis_padded, 0)

    prediction = nli_model.predict([premise_padded, hypothesis_padded])
    predicted_label_index = tf.argmax(prediction, axis=1).numpy()[0]

    label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
    predicted_label = label_map[predicted_label_index]
    probability = prediction[0][predicted_label_index]

    return predicted_label, probability

def classify_sts_bilstm(nli_model, sbert_model, sentence1, sentence2):
    sentences = [sentence1, sentence2]
    embeddings = sbert_model.encode(sentences)
    premise_padded, hypothesis_padded = embeddings[0], embeddings[1]
    premise_padded = tf.convert_to_tensor(premise_padded)
    hypothesis_padded = tf.convert_to_tensor(hypothesis_padded)

    if premise_padded.ndim == 1:
        premise_padded = tf.expand_dims(premise_padded, 0)
    if hypothesis_padded.ndim == 1:
        hypothesis_padded = tf.expand_dims(hypothesis_padded, 0)

    prediction = nli_model.predict([premise_padded, hypothesis_padded])
    similarity_score = prediction[0]

    return similarity_score

def main_bilstm(sbert_model, text):
    # sentences_embedded = preprocess_text(text)
    nli_model = load_model("models/NLI-BiLSTM.h5")
    model_sts = load_model("models/STS-BiLSTM.h5")

        # Call your NLI and STS models
    nli_result = perform_nli_bilstm(sbert_model, nli_model, text)
    sts_result = perform_sts_bilstm(sbert_model, model_sts, text, threshold)

        # # Display the results
    st.subheader("NLI Model Result:")
    st.text_area("NLI Summary", value=nli_result, height=200)

        # # Display the results
    st.subheader("STS Model Result:")
    st.text_area("STS Summary", value=sts_result, height=200)

if __name__ == "__main__":
    st.title("Extractive Summarization")
    text = st.text_area("Enter the text to summarize", height=200)
    col1, col2 = st.columns(2)
    model = col1.selectbox('Select the model', ('BERT', 'BiLSTM'), key="model_selectbox")
    threshold = col2.number_input("Enter the threshold for STS model", min_value=0.0, max_value=1.0, step=0.01, value=0.6, key="threshold_input")

    if model == 'BERT' and st.button("Summarize"):
        main_bert(text)
    if model == 'BiLSTM' and st.button("Summarize"):
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        main_bilstm(sbert_model, text)
