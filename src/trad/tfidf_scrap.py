# from sklearn.feature_extraction.text import TfidfVectorizer
#
# docs = [
#     "I love machine learning",
#     "I love deep learning",
#     "deep learning is amazing"
# ]
#
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(docs)
#
# # Show feature names and TF-IDF matrix
# print(vectorizer.get_feature_names_out())
# print(X.toarray())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Our "corpus" of documents
documents = [
    "I love machine learning and deep learning",
    "Deep learning is a branch of machine learning",
    "Natural language processing makes machines understand text",
    "I enjoy reading about AI and machine learning",
    "Cooking recipes are fun to try on weekends"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
# print(tfidf_matrix)

query = ["deep learning and AI"]

query_vec = vectorizer.transform(query)
print(query_vec)

cos_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

top_indices = np.argsort(cos_similarities)[::-1]

print("Top matches for query:")
for idx in top_indices[:3]:  # top 3
    print(f"Score: {cos_similarities[idx]:.3f} | Doc: {documents[idx]}")

