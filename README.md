# NLI-vs-STS-for-Text-Summarization
## Machine Learning Project
This is a project that compares two approaches to text summarization: Natural Language Inference (NLI) and Semantic Text Similarity (STS).

### Overview:

The goal of text summarization is to produce a shorter version of a text that captures its most important information. NLI and STS are two popular methods for text summarization that use different techniques:

- NLI relies on a machine learning model that is trained to determine whether a given statement (the summary) logically follows from another statement (the text to summarize). The idea is that if a summary can be inferred from a text, it must contain its most important information.

- STS, on the other hand, uses a similarity measure to compare the text to summarize with a reference corpus of texts, and selects the sentences that are most similar to the text. The idea is that sentences that are semantically similar to the text are likely to contain its most important information.

This project compares the performance of NLI and STS for text summarization using the MNLI dataset. The MNLI dataset is a collection of sentence pairs labeled with their entailment relationship (neutral, contradiction, or entailment) having different genres of sentences.

## Models and Performance
We trained several models for extractive text summarization, including BERT, Linear Regression, SVM, and BiLSTM. We also used SBERT for word embeddings, as GloVe performed poorly. The models were evaluated on the SNLI dataset, and the following accuracies and MAE scores were achieved:

| Model             | Accuracy | MAE score |
| ----------------- | -------- | -------- |
| BERT Transformer            | 0.8849     | 0.0322     |
| Bidirectional LSTM            | 0.7654     | 0.0749     |
| Logistic (Linear) Regression with TF-IDF             | 0.6458     | 0.1445     |
| SVM(R) Classification (Regression) with TF-IDF            | 0.4161     | 0.1851     |

Please refer to the individual model implementations for more details on their architectures and training processes.

## Contributors

- [Viktoriia Maksymiuk](https://github.com/Vihtoriaaa)
- [Yuliia Maksymiuk](https://github.com/juliaaz)


## Lisence

[MIT](LICENSE)
