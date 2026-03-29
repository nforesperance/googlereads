"""
Agent intelligent — Orchestre le dialogue entre l'utilisateur, le LLM et le ML.

Appelle le module LLM pour comprendre l'utilisateur, passe le résultat
au module ML, et formate une réponse claire et naturelle.
"""

from src.llm_parser import parse_preferences
from src.recommender import recommend


class AgentResult:
    """Résultat structuré d'une requête à l'agent."""

    def __init__(self, books: list[dict], genres: list[str], mood: str,
                 error: str | None = None):
        self.books = books
        self.genres = genres
        self.mood = mood
        self.error = error
        self.success = error is None

    def format_text(self) -> str:
        if self.error:
            return self.error

        genre_text = ", ".join(self.genres) if self.genres else "vos préférences"
        mood_text = f" avec une ambiance {self.mood}" if self.mood else ""

        lines = [
            f"D'après vos goûts ({genre_text}{mood_text}), "
            f"voici 5 livres qui pourraient vous plaire :\n"
        ]
        for i, book in enumerate(self.books, 1):
            lines.append(
                f"  {i}. \"{book['title']}\" par {book['author']}\n"
                f"     Genres : {book['genres']} "
                f"(pertinence : {int(book['score'] * 100)}%)"
            )
        lines.append(
            "\nVoulez-vous des recommandations différentes ? "
            "Décrivez vos envies !"
        )
        return "\n".join(lines)


class BookAgent:
    def __init__(self):
        self.history: list[dict] = []

    def run(self, user_input: str) -> AgentResult:
        """
        Traite le message de l'utilisateur et retourne un résultat structuré
        avec des recommandations de livres.
        """
        self.history.append({"role": "user", "message": user_input})

        # Étape 1 : Extraire les préférences via le LLM
        try:
            preferences = parse_preferences(user_input)
        except Exception as e:
            result = AgentResult(
                books=[], genres=[], mood="",
                error=f"Impossible d'analyser votre demande : {e}"
            )
            self.history.append({"role": "agent", "message": result.error})
            return result

        genres = preferences.get("genres", [])
        mood = preferences.get("mood", "")

        # Étape 2 : Obtenir les recommandations via le module ML
        try:
            books = recommend(genres, mood)
        except Exception as e:
            result = AgentResult(
                books=[], genres=genres, mood=mood,
                error=f"Impossible de trouver des recommandations : {e}"
            )
            self.history.append({"role": "agent", "message": result.error})
            return result

        result = AgentResult(books=books, genres=genres, mood=mood)
        self.history.append({"role": "agent", "message": result.format_text()})
        return result

    def get_history(self) -> list[dict]:
        return self.history.copy()

    def reset(self):
        self.history.clear()


def run_agent(user_input: str) -> AgentResult:
    """Fonction raccourci pour une utilisation simple (sans état)."""
    agent = BookAgent()
    return agent.run(user_input)


if __name__ == "__main__":
    print("=== Test du module agent ===\n")
    agent = BookAgent()

    test_inputs = [
        "j'aime le fantasy épique avec des quêtes",
        "maintenant je veux quelque chose de plus sombre, du thriller",
    ]

    for text in test_inputs:
        print(f"Utilisateur : {text}")
        response = agent.run(text)
        print(f"\nAgent :\n{response}\n")
        print("-" * 60 + "\n")
