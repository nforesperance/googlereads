"""
Module ML — Recommandation de livres avec k-NN et TF-IDF.

Charge le dataset Goodreads, vectorise les genres et descriptions,
et retourne les 5 livres les plus proches d'un profil donné.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "books.csv")


class BookRecommender:
    def __init__(self, data_path: str = DATA_PATH):
        self.df = self._load_and_clean(data_path)
        self.vectorizer = TfidfVectorizer(token_pattern=r"[a-zA-Zà-ÿ\-]+")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined"])
        self.model = NearestNeighbors(n_neighbors=5, metric="cosine")
        self.model.fit(self.tfidf_matrix)

    def _load_and_clean(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        required = ["title", "author", "genres", "description"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Colonne manquante dans le dataset : {col}")
        df = df[required].dropna().reset_index(drop=True)
        df["genres"] = df["genres"].str.strip()
        df["description"] = df["description"].str.strip()
        df["combined"] = df["genres"] + " " + df["description"]
        return df

    def recommend(self, genres: list[str], mood: str = "") -> list[dict]:
        """
        Retourne les 5 livres les plus similaires aux genres et mood donnés.

        Args:
            genres: liste de genres (ex: ["fantasy", "adventure"])
            mood: humeur optionnelle (ex: "sombre", "léger")

        Returns:
            Liste de 5 dicts avec title, author, genres.
        """
        query = " ".join(genres)
        if mood:
            query += " " + mood
        query_vec = self.vectorizer.transform([query])
        distances, indices = self.model.kneighbors(query_vec, n_neighbors=5)

        results = []
        for i, idx in enumerate(indices[0]):
            row = self.df.iloc[idx]
            results.append({
                "title": row["title"],
                "author": row["author"],
                "genres": row["genres"],
                "score": round(1 - distances[0][i], 3),
            })
        return results


# Instance globale pour utilisation simple
_recommender = None


def get_recommender() -> BookRecommender:
    global _recommender
    if _recommender is None:
        _recommender = BookRecommender()
    return _recommender


def recommend(genres: list[str], mood: str = "") -> list[dict]:
    """Fonction raccourci pour obtenir des recommandations."""
    return get_recommender().recommend(genres, mood)


if __name__ == "__main__":
    print("=== Test du module recommender ===\n")

    print('Test 1 : recommend(["fantasy", "adventure"])')
    results = recommend(["fantasy", "adventure"])
    for r in results:
        print(f"  - {r['title']} par {r['author']} (score: {r['score']})")

    print(f'\nTest 2 : recommend(["mystery", "thriller"])')
    results = recommend(["mystery", "thriller"])
    for r in results:
        print(f"  - {r['title']} par {r['author']} (score: {r['score']})")

    print(f'\nTest 3 : recommend(["romance"])')
    results = recommend(["romance"])
    for r in results:
        print(f"  - {r['title']} par {r['author']} (score: {r['score']})")

    print(f"\nTous les tests retournent {len(results)} résultats : OK")
