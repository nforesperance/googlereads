# Book Recommender — Recommandation de Livres par IA

Application de recommandation de livres personnalisée qui combine trois technologies d'IA :
- **LLM** (API Groq, Llama 3.3 70B) — comprend les goûts de l'utilisateur en langage naturel et génère une présentation contextuelle des recommandations
- **RAG** (Retrieval-Augmented Generation) — récupère les livres les plus pertinents par recherche sémantique (sentence-transformers) ou lexicale (TF-IDF), puis les injecte dans le LLM pour produire une recommandation narrative ancrée
- **Agent intelligent** — orchestre le pipeline : compréhension → recherche → génération

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Sentence_Transformers-FFD21E?style=for-the-badge)

## Équipe

| Membre | Rôle | Module |
|--------|------|--------|
| Membre 1 | Module Retrieval (ML) | `src/recommender.py` |
| Membre 2 | Module LLM | `src/llm_parser.py` |
| Membre 3 | Agent intelligent | `src/agent.py` |
| Membre 4 | Module RAG & Intégration | `src/rag.py`, `app.py`, `main.py`, README, blog |

## Installation

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd book-recommender

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Configuration

L'application utilise l'API Groq (gratuite) pour l'analyse du langage naturel et la génération RAG.

1. Créer un compte sur [console.groq.com](https://console.groq.com)
2. Générer une clé API dans l'onglet **API Keys**
3. Configurer la clé selon votre préférence :

### Option A — Fichier `.env` (recommandé en local)

Créer un fichier `.env` à la racine du projet :

```env
GROQ_API_KEY=gsk_votre_clé_ici
RECOMMENDER_BACKEND=tfidf            # ou "sentence-transformers"
RECOMMENDER_ST_MODEL=all-MiniLM-L6-v2
```

### Option B — Variables d'environnement

```bash
export GROQ_API_KEY="gsk_votre_clé"
export RECOMMENDER_BACKEND="sentence-transformers"
```

### Option C — Streamlit Cloud (Secrets)

Dans le panneau **Secrets** de l'app Streamlit Cloud :

```toml
GROQ_API_KEY = "gsk_..."
RECOMMENDER_BACKEND = "sentence-transformers"
```

### Option D — Saisie directe dans l'interface

Si aucune clé n'est trouvée dans l'environnement, l'application propose un champ de saisie sécurisé dans la barre latérale. Si une clé est déjà détectée, l'interface l'indique mais permet de la remplacer pour la session.

## Utilisation

### Interface Web (recommandé)

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur avec :
- **Champ de recherche** en langage naturel (la touche Entrée déclenche la recherche)
- **Suggestions cliquables** dans la barre latérale qui pré-remplissent et lancent la recherche
- **Sélecteur de moteur** : TF-IDF (rapide, lexical) ou Sentence-Transformers (sémantique)
- **Sélecteur de dataset** : tous les fichiers CSV présents dans `data/` sont détectés automatiquement
- **Cartes de livres** avec genres, description et barre de pertinence
- **Encart RAG** dépliable « Pourquoi ces livres ? » avec une présentation narrative générée par le LLM
- **Historique** des recherches

### Interface CLI

```bash
python main.py
```

## Architecture

```
Utilisateur
    │
    ▼
┌──────────┐     ┌─────────────┐     ┌──────────────┐
│  Agent   │────▶│  LLM Parser │────▶│  API Groq    │
│ agent.py │     │ llm_parser  │     │  (Llama 3.3) │
│          │     └─────────────┘     └──────────────┘
│          │            │
│          │     {"genres": [...], "mood": "..."}
│          │            │
│          │     ┌──────────────────────┐
│          │────▶│  Retrieval           │
│          │     │  recommender.py      │
│          │     │  TF-IDF | Sentence-T │
│          │     │  + k-NN cosine       │
│          │     └──────────────────────┘
│          │            │
│          │     5 livres + métadonnées
│          │            │
│          │     ┌──────────────────────┐
│          │────▶│  RAG (Generation)    │
│          │     │  rag.py              │
│          │     │  Groq Llama 3.3 70B  │
│          │     └──────────────────────┘
│          │            │
└──────────┘     Présentation narrative
    │
    ▼
 5 livres + paragraphe narratif (RAG)
```

## Structure du projet

```
book-recommender/
├── README.md
├── requirements.txt
├── .gitignore
├── .env                      ← Clés API et configuration (gitignored)
├── .streamlit/
│   └── config.toml           ← Désactivation du watcher (compat. transformers)
├── app.py                    ← Interface Streamlit
├── main.py                   ← Interface CLI
├── data/
│   ├── books.csv             ← Dataset de 96 livres (démo TF-IDF rapide)
│   └── books_scraped.csv     ← Dataset étendu (Goodreads scraping)
├── docs/
│   └── blog.md               ← Blog technique du projet
└── src/
    ├── __init__.py
    ├── recommender.py        ← Retrieval (TF-IDF / sentence-transformers + k-NN)
    ├── llm_parser.py         ← Parsing LLM (extraction JSON des préférences)
    ├── rag.py                ← Génération RAG (présentation narrative)
    ├── agent.py              ← Agent orchestrateur
    └── extract.py            ← Script de scraping Goodreads
```

## Technologies

- **Python 3.10+**
- **Streamlit** — interface web interactive avec formulaires et sélecteurs dynamiques
- **scikit-learn** — TF-IDF + k-NN cosine
- **sentence-transformers** — embeddings sémantiques (modèle `all-MiniLM-L6-v2` par défaut)
- **pandas** — chargement et nettoyage du dataset
- **Groq API** — Llama 3.3 70B pour le parsing JSON et la génération RAG
- **python-dotenv** — chargement des variables d'environnement depuis `.env`
- **BeautifulSoup4 + requests** — scraping du dataset étendu (`src/extract.py`)
