
import json
import os

from groq import Groq


RAG_SYSTEM_PROMPT = """Tu es un libraire passionné qui présente une sélection de livres à un lecteur.
Tu reçois la requête originale du lecteur et 5 livres sélectionnés par un moteur de recherche sémantique.

Rédige une présentation narrative en français (4 à 6 phrases) qui :
- explique en une phrase pourquoi ces livres correspondent à la demande,
- met en avant 2 ou 3 livres en citant leurs titres entre guillemets,
- propose un fil conducteur (thème commun, ambiance, progression de lecture),
- reste chaleureux et conversationnel, sans énumérer mécaniquement chaque livre.

N'invente AUCUNE information : appuie-toi uniquement sur les titres, auteurs, genres et résumés fournis.
"""


def generate_explanation(user_query: str, books: list[dict]) -> str:
    """
    Génère une présentation narrative des livres recommandés (étape RAG).

    Args:
        user_query: la requête originale de l'utilisateur.
        books: liste de dicts avec title, author, genres, description.

    Returns:
        Un paragraphe en français présentant les livres en contexte.
    """
    if not books:
        return ""

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "La clé API Groq est manquante pour la génération RAG."
        )

    context_lines = [f'Requête du lecteur : "{user_query}"\n', "Livres sélectionnés :"]
    for i, b in enumerate(books, 1):
        desc = (b.get("description") or "").strip().replace("\n", " ")
        if len(desc) > 400:
            desc = desc[:400].rsplit(" ", 1)[0] + "..."
        genres = b.get("genres") or "—"
        context_lines.append(
            f'{i}. "{b["title"]}" — {b["author"]}\n'
            f"   Genres : {genres}\n"
            f"   Résumé : {desc}"
        )
    context = "\n".join(context_lines)

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=0.6,
            max_completion_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la génération RAG : {e}")
