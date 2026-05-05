# Submodule Map: Natural Language Processing

This document outlines the subtopics covered in this module and serves as a planning map for the instructor.

## List of Subtopics

1.  **01_text_preprocessing:** The foundational steps of cleaning and preparing text data.
2.  **02_text_representation:** Techniques for converting text into numerical vectors that models can understand.
3.  **03_classical_nlp_models:** Applying traditional machine learning algorithms to text data.
4.  **04_word_embeddings_intro:** An introduction to dense, semantic vector representations of words.

## Why Each Subtopic is Taught

-   **01_text_preprocessing:** Raw text is messy. Punctuation, capitalization, and common words (stop words) add noise. Preprocessing is essential for improving model performance and reducing dimensionality.
-   **02_text_representation:** Machine learning models only understand numbers. We need systematic ways to translate text into numerical formats while preserving some meaning or structure.
-   **03_classical_nlp_models:** To show students how to actually use the preprocessed and vectorized text to solve a real business problem, like sentiment analysis or document classification.
-   **04_word_embeddings_intro:** To bridge the gap between classical NLP (which ignores word context/meaning) and modern Deep Learning NLP. Embeddings capture semantic relationships.

## Topic Characteristics

### Theory-Heavy Subtopics
-   **02_text_representation:** The mathematical intuition behind TF-IDF is important to explain.
-   **04_word_embeddings_intro:** The concept of dense vectors capturing semantic meaning in a multi-dimensional space requires good theoretical explanation.

### Code-Heavy Subtopics
-   **01_text_preprocessing:** Lots of string manipulation and using libraries like NLTK or SpaCy.
-   **03_classical_nlp_models:** Training and evaluating `scikit-learn` models using text pipelines.

### Visual Explanation Needed
-   **02_text_representation:** Visualizing a Document-Term Matrix (DTM) is crucial.
-   **04_word_embeddings_intro:** Visualizing word embeddings in 2D space (using PCA or t-SNE) to show how similar words cluster together.

### Business Examples Needed
-   **01_text_preprocessing:** Show how ignoring case might break a simple keyword search.
-   **03_classical_nlp_models:** Customer review sentiment analysis or automated email routing.

## Recommended Order of Teaching

1.  **Text Preprocessing:** Start with the raw data. How do we clean it?
2.  **Text Representation:** Now that it's clean, how do we make it numbers?
3.  **Classical NLP Models:** Now that we have numbers, let's train a model.
4.  **Word Embeddings Intro:** Now that we know the basics, what are the limitations, and how do modern techniques fix them?
