"""
Module ML — Recommandation de livres avec k-NN.

Charge le dataset Goodreads, vectorise les genres et descriptions
(via TF-IDF ou sentence-transformers), et retourne les 5 livres
les plus proches d'un profil donné.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

load_dotenv()

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "books.csv")
DEFAULT_BACKEND = os.getenv("RECOMMENDER_BACKEND", "tfidf").lower()
DEFAULT_ST_MODEL = os.getenv("RECOMMENDER_ST_MODEL", "all-MiniLM-L6-v2")


class BookRecommender:
    def __init__(
        self,
        data_path: str = DATA_PATH,
        backend: str | None = None,
        st_model: str = DEFAULT_ST_MODEL,
    ):
        self.backend = (backend or DEFAULT_BACKEND).lower()
        if self.backend not in ("tfidf", "sentence-transformers"):
            raise ValueError(
                f"backend doit être 'tfidf' ou 'sentence-transformers', reçu: {self.backend}"
            )

        self.df = self._load_and_clean(data_path)

        if self.backend == "tfidf":
            print("Initialisation du vectoriseur TF-IDF...")
            self.vectorizer = TfidfVectorizer(token_pattern=r"[a-zA-Zà-ÿ\-]+")
            self.matrix = self.vectorizer.fit_transform(self.df["combined"])
        else:
            from sentence_transformers import SentenceTransformer
            print(f"Chargement du modèle sentence-transformers '{st_model}'...")
            self.encoder = SentenceTransformer(st_model)
            self.matrix = self.encoder.encode(
                self.df["combined"].tolist(), show_progress_bar=False
            )

        self.model = NearestNeighbors(n_neighbors=5, metric="cosine")
        self.model.fit(self.matrix)

    def _embed_query(self, query: str):
        if self.backend == "tfidf":
            return self.vectorizer.transform([query])
        return self.encoder.encode([query], show_progress_bar=False)

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
        query_vec = self._embed_query(query)
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


def get_recommender(backend: str | None = None) -> BookRecommender:
    global _recommender
    if _recommender is None:
        _recommender = BookRecommender(backend=backend)
    return _recommender


def set_backend(backend: str) -> BookRecommender:
    """Force la reconstruction de l'instance globale avec un nouveau backend."""
    global _recommender
    _recommender = BookRecommender(backend=backend)
    return _recommender


def current_backend() -> str | None:
    """Retourne le backend de l'instance globale, ou None si non initialisée."""
    return _recommender.backend if _recommender else None


def recommend(genres: list[str], mood: str = "", backend: str | None = None) -> list[dict]:
    """Fonction raccourci pour obtenir des recommandations."""
    return get_recommender(backend=backend).recommend(genres, mood)


if __name__ == "__main__":
    print(f"=== Test du module recommender (backend: {DEFAULT_BACKEND}) ===\n")

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
