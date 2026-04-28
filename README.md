# A Comparative Study of Binary Sentiment Analysis on IMDB and SST-2

[cite_start]**Authors:** Cuclea Luca-Nicolae [cite: 2][cite_start], Rizea Mihai-Marius [cite: 3][cite_start], Dilmac Cemil-Andrei [cite: 4][cite_start], Popa Petru [cite: 5]

---

## Overview
[cite_start]This repository contains a comprehensive benchmarking study evaluating an evolutionary spectrum of machine learning architectures for binary sentiment analysis[cite: 9]. [cite_start]The project traces the progression of natural language processing techniques, demonstrating how transitioning from frequency-based representations to deep contextualized embeddings resolves critical limitations in capturing compositional semantics, long-range dependencies, and contextual nuance[cite: 10, 11].

## Datasets
[cite_start]The models are benchmarked on two widely used, complementary datasets to ensure a fair and comprehensive test[cite: 19]:
* [cite_start]**SST-2 (Stanford Sentiment Treebank):** This dataset contains short, sentence-level texts where sentiment is expressed with limited context[cite: 97].
* [cite_start]**IMDB (Large Movie Review Dataset):** This dataset consists of long, document-level reviews that provide richer contextual information and clearer sentiment signals[cite: 97].

## Architectures Evaluated
[cite_start]The methodology evaluates the following sequence of models to showcase how NLP architectures have progressed to solve the limitations of their predecessors[cite: 22]:

* [cite_start]**Classical Baselines:** We evaluate traditional models like Logistic Regression and Support Vector Machines (SVM)[cite: 23]. [cite_start]These baselines use TF-IDF vectorization to turn raw text into numerical features[cite: 76].
* [cite_start]**Hybrid Models:** We implement a Naive Bayes-SVM (NB-SVM) hybrid architecture[cite: 25]. [cite_start]This model applies a Naive Bayes log-count ratio transformation to re-weight TF-IDF features before classification[cite: 84, 86].
* [cite_start]**Deep Learning (LSTM):** We utilize Long Short-Term Memory networks to address the word order problem found in classical models[cite: 26]. [cite_start]LSTMs process the review as a sequence, token by token, and maintain an internal state that carries information from previous words[cite: 47].
* [cite_start]**Transformer Architectures:** We evaluate BERT-based models, specifically focusing on RoBERTa[cite: 28, 65]. [cite_start]We improve RoBERTa's fine-tuning phase by adding a Supervised Contrastive Learning (SCL) loss function[cite: 93]. [cite_start]This explicitly pulls reviews with the same sentiment closer together while pushing opposite classes apart[cite: 94].

## Key Results
* [cite_start]During preprocessing evaluations, TF-IDF performed better than Bag of Words (BoW), providing a more discriminative representation of the reviews[cite: 261].
* [cite_start]LSTM models achieved an accuracy of approximately 85-86% on the SST-2 dataset[cite: 351]. [cite_start]On the IMDB dataset, the LSTM achieved an accuracy of 89-90%, showing no improvement over the classical models due to struggles with extremely long-range dependencies[cite: 354, 358].
* [cite_start]RoBERTa+SCL represents the strongest model in our comparative study[cite: 365]. [cite_start]This configuration reached a 94.38% validation accuracy on SST-2 [cite: 412] [cite_start]and a 93.90% test accuracy on IMDB[cite: 414].

## Limitations and Future Work
* [cite_start]Error analysis revealed that classical models suffer from an inability to handle contrast, polarity shifts, and irony[cite: 319, 322].
* [cite_start]Transformer-based models were computationally demanding, requiring an NVIDIA RTX 5090 GPU with 32 GB of memory to run fine-tuning experiments[cite: 438].
* [cite_start]This study focuses only on English binary sentiment analysis[cite: 440]. [cite_start]Future work could expand models to predict a wider spectrum of emotions or test performance on multilingual datasets[cite: 422, 441].