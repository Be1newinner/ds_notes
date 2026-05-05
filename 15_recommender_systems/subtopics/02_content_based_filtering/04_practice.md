# Practice: Content-Based Filtering

## Exercise 1: Understanding Vectors
You have three movies with the following genre tags:
- Movie A: "Action Sci-Fi"
- Movie B: "Action Adventure Sci-Fi"
- Movie C: "Comedy Romance"

Assume we use a simple Count Vectorizer with the vocabulary: `['Action', 'Adventure', 'Comedy', 'Romance', 'Sci-Fi']`

1. Write out the numeric vector for Movie A, Movie B, and Movie C.
2. Based on intuition (or math if you want), which two movies will have the highest Cosine Similarity?

## Exercise 2: TF-IDF Intuition
You are building a content-based recommender for news articles. Article 1 is about "Stock Market Trends". Article 2 is about "Tech Company Earnings".
Both articles contain the words "the", "and", "money" very frequently.
Why would `TfidfVectorizer` be better to use here than `CountVectorizer`?

## Exercise 3: Python Coding Challenge
Load a small dataset of books with a `description` column.
1. Use `TfidfVectorizer` (with English stop words removed) to convert the descriptions into a matrix.
2. Calculate the cosine similarity matrix.
3. Write a function `recommend_book(title)` that prints the top 3 most similar books based on the matrix.

## Exercise 4: The Filter Bubble
A user only watches documentaries about World War II. They rate all of them 5 stars.
1. Explain what a content-based recommender will suggest to them next.
2. What is the main drawback of this approach in this specific scenario?
