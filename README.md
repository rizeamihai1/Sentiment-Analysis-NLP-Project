# A Comparative Study of Binary Sentiment Analysis on IMDB and SST-2

**Authors:** Cuclea Luca-Nicolae, Rizea Mihai-Marius, Dilmac Cemil-Andrei, Popa Petru

---

## Overview
This repository contains a comprehensive benchmarking study evaluating an evolutionary spectrum of machine learning architectures for binary sentiment analysis. The project traces the progression of natural language processing techniques, demonstrating how transitioning from frequency-based representations to deep contextualized embeddings resolves critical limitations in capturing compositional semantics, long-range dependencies, and contextual nuance.

## Datasets
The models are benchmarked on two widely used, complementary datasets to ensure a fair and comprehensive test:
* **SST-2 (Stanford Sentiment Treebank):** Contains short, sentence-level texts where sentiment is expressed with limited context.
* **IMDB (Large Movie Review Dataset):** Consists of long, document-level reviews that provide richer contextual information and clearer sentiment signals.

## Architectures Evaluated
The methodology evaluates the following sequence of models to showcase how NLP architectures have progressed over time:

* **Classical Baselines:** Evaluates traditional models like Logistic Regression and Support Vector Machines (SVM) using TF-IDF vectorization.
* **Hybrid Models:** Implements a Naive Bayes-SVM (NB-SVM) hybrid architecture that applies a Naive Bayes log-count ratio transformation to re-weight TF-IDF features.
* **Deep Learning (LSTM):** Utilizes Long Short-Term Memory networks to address the word order problem found in classical models by processing the review as a sequence.
* **Transformer Architectures:** Evaluates BERT-based models, specifically focusing on RoBERTa enhanced with a Supervised Contrastive Learning (SCL) loss function to pull similar sentiments closer together.

## Key Results
* **Preprocessing:** TF-IDF outperformed Bag of Words (BoW) by providing a more discriminative representation of the reviews.
* **LSTM Performance:** Achieved approximately 85-86% accuracy on the SST-2 dataset, but showed no improvement over classical models on the IMDB dataset (89-90%) due to struggles with long-range dependencies.
* **State-of-the-Art:** RoBERTa+SCL was the strongest model, reaching a 94.38% validation accuracy on SST-2 and a 93.90% test accuracy on IMDB.

## Limitations and Future Work
* Classical models struggled with contrast, polarity shifts, and irony.
* Transformer models required significant computational resources (e.g., NVIDIA RTX 5090 GPU with 32 GB memory).
* The study is limited to English. Future work could expand to multilingual datasets and predict a wider, more fine-grained spectrum of emotions.
