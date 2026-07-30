import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample Movie Dataset
movies_data = {
    "movieId": [1, 2, 3, 4, 5, 6],
    "title": [
        "The Dark Knight",
        "Inception",
        "Interstellar",
        "Toy Story",
        "Finding Nemo",
        "The Matrix",
    ],
    "genres": [
        "Action Crime Drama IMAX",
        "Action Adventure Sci-Fi IMAX",
        "Adventure Drama Sci-Fi IMAX",
        "Adventure Animation Children Comedy Fantasy",
        "Adventure Animation Children Comedy",
        "Action Sci-Fi",
    ],
}

df_movies = pd.DataFrame(movies_data)

# 1. Compute TF-IDF Matrix on Genres / Metadata
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df_movies["genres"])

# 2. Compute Cosine Similarity Matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Map title to index
indices = pd.Series(df_movies.index, index=df_movies["title"]).drop_duplicates()


# 3. Recommendation Function
def get_content_recommendations(title, cosine_sim=cosine_sim, top_n=3):
    if title not in indices:
        return f"Movie '{title}' not found in database."

    idx = indices[title]

    # Get pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top N most similar movies (excluding itself)
    sim_scores = sim_scores[1 : top_n + 1]
    movie_indices = [i[0] for i in sim_scores]

    return df_movies[["title", "genres"]].iloc[movie_indices]


print("--- Content-Based Recommendations for 'Inception' ---")
print(get_content_recommendations("Inception", top_n=3))