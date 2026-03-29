"""
Module LLM — Extraction des préférences utilisateur via l'API Groq.

Envoie le message de l'utilisateur à l'API LLM et extrait les genres
littéraires et l'humeur sous forme de JSON structuré.
"""

import json
import os

from groq import Groq


SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'analyse des goûts littéraires.
L'utilisateur va décrire ses préférences de lecture en langage naturel.

Tu dois répondre UNIQUEMENT avec un objet JSON valide, sans texte avant ou après.
Le JSON doit contenir exactement ces deux clés :
- "genres" : une liste de genres littéraires en anglais (ex: ["fantasy", "mystery", "romance"])
- "mood" : un mot décrivant l'humeur ou le ton souhaité en français (ex: "sombre", "léger", "aventureux")

Genres possibles : fiction, non-fiction, fantasy, science-fiction, mystery, thriller, romance,
horror, historical, literary, adventure, dystopian, classic, young-adult, contemporary,
comedy, drama, philosophical, gothic, crime, epic, psychological, war, memoir, biography,
self-help, science, mythology, political, coming-of-age, post-apocalyptic, cyberpunk, detective,
supernatural, survival, dark-academia, feminist, heartwarming, music, productivity, humor.

Si l'utilisateur est vague, fais de ton mieux pour deviner les genres les plus probables.
Retourne toujours au moins un genre.

Exemples :
- "j'aime les romans policiers sombres" → {"genres": ["mystery", "thriller", "crime"], "mood": "sombre"}
- "quelque chose de léger et drôle" → {"genres": ["comedy", "contemporary", "fiction"], "mood": "léger"}
- "je veux de la science-fiction épique" → {"genres": ["science-fiction", "epic", "adventure"], "mood": "aventureux"}
"""


def parse_preferences(user_input: str) -> dict:
    """
    Analyse le message de l'utilisateur et extrait les genres et l'humeur.

    Args:
        user_input: texte libre de l'utilisateur décrivant ses goûts.

    Returns:
        dict avec clés "genres" (list[str]) et "mood" (str).
    """
    if not user_input or not user_input.strip():
        return {"genres": ["fiction"], "mood": "neutre"}

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "La clé API Groq est manquante. "
            "Définissez la variable d'environnement GROQ_API_KEY.\n"
            "Créez un compte gratuit sur https://console.groq.com"
        )

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.3,
            max_completion_tokens=256,
        )

        content = response.choices[0].message.content.strip()

        # Extraire le JSON même si le LLM ajoute du texte autour
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("Pas de JSON trouvé dans la réponse du LLM")

        result = json.loads(content[start:end])

        # Valider la structure
        if "genres" not in result or not isinstance(result["genres"], list):
            result["genres"] = ["fiction"]
        if "mood" not in result or not isinstance(result["mood"], str):
            result["mood"] = "neutre"

        return result

    except json.JSONDecodeError:
        return {"genres": ["fiction"], "mood": "neutre"}
    except Exception as e:
        raise RuntimeError(f"Erreur lors de l'appel à l'API Groq : {e}")


if __name__ == "__main__":
    print("=== Test du module llm_parser ===\n")

    test_inputs = [
        "j'aime les romans policiers sombres",
        "quelque chose de léger et drôle",
        "je veux de la science-fiction épique dans l'espace",
        "",
    ]

    for text in test_inputs:
        print(f"Input : '{text}'")
        try:
            result = parse_preferences(text)
            print(f"  → genres: {result['genres']}, mood: {result['mood']}")
        except Exception as e:
            print(f"  → Erreur : {e}")
        print()
