# Book Recommender — Recommandation de Livres par IA

Application de recommandation de livres personnalisée qui combine trois technologies d'IA :
- **LLM** (API Groq) — comprend les goûts de l'utilisateur en langage naturel
- **ML** (k-NN + TF-IDF, scikit-learn) — calcule les livres les plus similaires
- **Agent intelligent** — orchestre le dialogue et les appels entre modules

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Équipe

| Membre | Rôle | Module |
|--------|------|--------|
| Membre 1 | Module ML | `src/recommender.py` |
| Membre 2 | Module LLM | `src/llm_parser.py` |
| Membre 3 | Agent intelligent | `src/agent.py` |
| Membre 4 | Intégration & Docs | `app.py`, `main.py`, README, blog |

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

L'application utilise l'API Groq (gratuite) pour l'analyse du langage naturel.

1. Créer un compte sur [console.groq.com](https://console.groq.com)
2. Générer une clé API dans l'onglet **API Keys**
3. La clé peut être saisie directement dans l'interface Streamlit, ou définie comme variable d'environnement :

```bash
export GROQ_API_KEY="votre-clé-api-ici"
```

## Utilisation

### Interface Web (recommandé)

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur avec :
- Un champ de recherche pour décrire vos goûts en langage naturel
- Des suggestions cliquables dans la barre latérale
- Des cartes de livres avec genres, description et score de pertinence
- Un historique des recherches

### Interface CLI

```bash
python main.py
```

## Architecture

```
Utilisateur
    │
    ▼
┌─────────┐     ┌─────────────┐     ┌──────────────┐
│  Agent   │────▶│  LLM Parser │────▶│  API Groq    │
│ agent.py │     │ llm_parser  │     │  (Llama 3.3) │
│          │     └─────────────┘     └──────────────┘
│          │            │
│          │    {"genres": [...], "mood": "..."}
│          │            │
│          │     ┌──────────────┐
│          │────▶│ Recommender  │
│          │     │ (k-NN+TF-IDF)│
└─────────┘     └──────────────┘
    │
    ▼
 5 livres recommandés
```

## Structure du projet

```
book-recommender/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py                   ← Interface Streamlit
├── main.py                  ← Interface CLI
├── data/
│   └── books.csv            ← Dataset de 96 livres
├── docs/
│   └── blog.md              ← Blog technique du projet
└── src/
    ├── __init__.py
    ├── recommender.py        ← Module ML (k-NN + TF-IDF)
    ├── llm_parser.py         ← Module LLM (API Groq)
    └── agent.py              ← Agent intelligent
```

## Technologies

- **Python 3.10+**
- **Streamlit** — interface web interactive
- **scikit-learn** — TF-IDF vectorization + k-NN
- **pandas** — manipulation du dataset
- **Groq API** — LLM (Llama 3.3 70B) pour l'analyse du langage naturel
