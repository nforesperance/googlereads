# Blog — Projet de Recommandation de Livres par IA

## Introduction

Ce projet est une application de recommandation de livres qui permet à un utilisateur de décrire ses goûts littéraires en langage naturel et de recevoir 5 recommandations personnalisées, accompagnées d'une présentation narrative générée par IA. L'application combine trois technologies au programme du cours : un **LLM** pour comprendre l'utilisateur et formuler les recommandations, une approche **RAG** (Retrieval-Augmented Generation) pour ancrer la génération sur un corpus réel de livres, et un **agent intelligent** qui orchestre le pipeline de bout en bout.

## Technologies choisies

### LLM — Groq API (Llama 3.3 70B)

Nous avons choisi l'API Groq pour sa gratuité et sa rapidité. Le modèle Llama 3.3 70B intervient à **deux moments** distincts du pipeline :

1. **En amont** (`src/llm_parser.py`) — il analyse la requête en langage naturel et extrait un objet JSON structuré `{"genres": [...], "mood": "..."}` qui sert de filtre pour la recherche.
2. **En aval** (`src/rag.py`) — c'est l'étape de génération du RAG : le LLM reçoit la requête originale **et** les 5 livres récupérés, puis rédige une présentation narrative en français qui cite explicitement les livres recommandés.

**Pourquoi Groq ?** L'API est gratuite, rapide (latence très faible), et compatible avec les modèles open source les plus performants. Contrairement à OpenAI, elle ne nécessite pas de carte de crédit pour commencer, ce qui est idéal pour un projet pédagogique.

### RAG — Retrieval-Augmented Generation

Le **RAG** est le cœur technique du projet. Plutôt que de demander au LLM de recommander des livres « de mémoire » (ce qui produirait des hallucinations et des titres inventés), nous utilisons une étape de **récupération sémantique** sur un corpus réel, puis nous **augmentons** le prompt du LLM avec ces résultats avant de lui demander de **générer** la présentation.

Le pipeline RAG comporte trois sous-étapes :

#### 1. Indexation (offline, au démarrage)

Au lancement, le module `recommender.py` charge le dataset choisi et vectorise la concaténation `genres + description` de chaque livre. Deux backends sont disponibles, sélectionnables au runtime :

- **TF-IDF (scikit-learn)** — rapide, lexical, sans téléchargement de modèle. Idéal pour des datasets volumineux et des démos en local sans GPU.
- **Sentence-Transformers (`all-MiniLM-L6-v2`)** — embeddings sémantiques denses produits par un transformer fine-tuné en contrastive learning. Capture la *similarité de sens* : la requête « sombre » matche des descriptions contenant *dark*, *grim*, *melancholic*, là où TF-IDF aurait échoué.

#### 2. Retrieval (à chaque requête)

L'utilisateur saisit une requête, le LLM en extrait `{genres, mood}`. La requête textuelle (par ex. `"mystery thriller crime sombre"`) est vectorisée par le **même** vectoriseur que le corpus, puis un **k-NN cosine** (`sklearn.neighbors.NearestNeighbors`) retourne les 5 voisins les plus proches.

#### 3. Augmentation + Génération

Le module `src/rag.py` construit un prompt qui contient :
- la requête originale de l'utilisateur,
- les 5 livres retournés par le retrieval (titre, auteur, genres, résumé tronqué à 400 caractères).

Ce prompt est envoyé au LLM avec une instruction stricte : **« N'invente AUCUNE information : appuie-toi uniquement sur les éléments fournis. »** Le LLM produit alors un paragraphe de présentation chaleureux qui cite 2-3 livres entre guillemets et propose un fil conducteur.

**Pourquoi RAG plutôt qu'un LLM seul ?** Sans retrieval, le LLM hallucinerait des titres inexistants ou recommanderait toujours les mêmes best-sellers populaires connus de son entraînement. Avec RAG, les recommandations sont **ancrées sur notre dataset réel** (Goodreads) et le LLM ne peut pas dévier.

**Pourquoi sentence-transformers en plus de TF-IDF ?** Les deux backends ont des forces complémentaires que l'utilisateur peut comparer en direct via la barre latérale. La présentation montre les deux moteurs côte à côte sur des datasets de tailles différentes.

### Agent intelligent

L'agent (`src/agent.py`) est le chef d'orchestre du système. Il coordonne séquentiellement :

```
parse_preferences  →  recommend  →  generate_explanation
   (LLM JSON)        (Retrieval)        (RAG Generation)
```

Il gère les erreurs à chaque étape avec une stratégie de **dégradation gracieuse** : si l'étape RAG échoue (pas de connexion, quota dépassé), le pipeline retourne quand même les 5 livres mais sans paragraphe narratif. L'utilisateur voit ses recommandations, l'expander RAG est simplement masqué.

L'agent retourne un objet `AgentResult` qui regroupe : les 5 livres avec leurs scores de pertinence, les genres détectés, le mood, et le texte généré par le RAG.

### Interface utilisateur — Streamlit

Streamlit nous a permis de créer rapidement une interface complète :

- **Recherche en langage naturel** dans un formulaire (la touche `Entrée` déclenche la recherche, sans passer par le bouton)
- **Suggestions cliquables** dans la barre latérale qui pré-remplissent l'input et lancent la recherche
- **Sélecteur de backend** (radio TF-IDF / Sentence-Transformers) avec rechargement automatique du recommender quand l'utilisateur change
- **Sélecteur de dataset** (selectbox détectant tous les CSV de `data/`)
- **Encart RAG dépliable** « 💡 Pourquoi ces livres ? » avec le paragraphe narratif (replié par défaut pour ne pas polluer la vue)
- **Cartes de livres** avec genres, description et barre de pertinence
- **Configuration de la clé API** : auto-détection depuis `.env` ou `st.secrets`, avec possibilité de surcharger pour la session
- **Historique** des recherches

**Pourquoi Streamlit ?** C'est le framework Python le plus rapide pour créer des interfaces de données interactives. Il gère nativement le state management via `st.session_state`, ce qui est parfait pour notre historique de conversation et le cache de l'instance recommender.

## Architecture du système

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
│          │────▶│  RAG Generation      │
│          │     │  rag.py              │
│          │     │  Groq Llama 3.3 70B  │
│          │     └──────────────────────┘
│          │            │
└──────────┘     Paragraphe narratif ancré
    │
    ▼
 5 livres + présentation RAG
```

## Défis rencontrés et solutions

### 1. Hallucinations du LLM lors de la recommandation

**Défi :** Une première version demandait au LLM de recommander directement des livres en langage naturel. Il inventait des titres ou citait toujours les mêmes œuvres canoniques (Harry Potter, 1984, Le Seigneur des Anneaux), sans aucun lien avec notre dataset.

**Solution :** Adoption de l'architecture **RAG**. Le LLM ne recommande plus, il *présente* — il reçoit en entrée 5 livres réels issus de notre corpus et a pour seule mission de les introduire de manière chaleureuse en s'appuyant uniquement sur les métadonnées fournies.

### 2. Limites du TF-IDF pour les requêtes sémantiques

**Défi :** TF-IDF fonctionne bien pour les correspondances exactes (« fantasy » → livres taggés *fantasy*) mais échoue sur les requêtes sémantiques. La requête « sombre » ne matchait aucun livre dont la description anglaise contient *dark* ou *grim*.

**Solution :** Ajout d'un **second backend de retrieval** basé sur `sentence-transformers` (modèle `all-MiniLM-L6-v2`, ~80 MB). Les vecteurs denses capturent la similarité sémantique, et le même k-NN cosine fonctionne par-dessus sans changement. L'utilisateur peut basculer entre les deux backends en direct depuis l'interface pour comparer.

### 3. Datasets de tailles différentes

**Défi :** Notre petit dataset de 96 livres (`books.csv`) est rapide à indexer et idéal pour la démo sentence-transformers, mais un dataset plus large issu d'un scraping Goodreads (`books_scraped.csv`, ~30 000 livres) montre mieux la valeur de TF-IDF en passage à l'échelle. Cependant, le CSV scrapé n'a pas de colonne `genres` exploitable — pandas l'a inférée comme `float64` (NaN) et le pipeline cassait.

**Solution :** Le `_load_and_clean` du recommender ne requiert plus que `title` et `description`. Les colonnes `author` et `genres`, si manquantes ou tout-NaN, sont remplies par des chaînes vides. L'utilisateur peut alors choisir le dataset depuis un selectbox de la barre latérale, et le recommender se reconstruit automatiquement (cache invalidé) quand le dataset OU le backend change.

### 4. Configuration de la clé API multi-environnement

**Défi :** L'application doit fonctionner en local (avec `.env`), sur Streamlit Cloud (via `st.secrets`), ou en saisie directe dans l'interface. La gestion en cascade devait rester transparente pour l'utilisateur.

**Solution :** Une fonction `resolve_api_key()` cherche la clé dans cet ordre : `os.environ` → `.env` (chargé par `python-dotenv`) → `st.secrets`. Si une clé est trouvée, l'interface affiche un badge ✅ avec la source (`(.env)` ou `(Streamlit secrets)`), avec un expander « Utiliser une autre clé » pour surcharger ponctuellement. Si rien n'est trouvé, le champ de saisie classique est affiché.

### 5. Compatibilité Streamlit + Transformers

**Défi :** Le watcher de fichiers de Streamlit parcourt `sys.modules` à chaque rerun pour le hot-reload. Cette introspection accède à `__path__` de chaque module, ce qui déclenche le lazy-loading de `transformers` et fait remonter des dizaines d'avertissements verbeux dans le terminal (`[transformers] Accessing __path__ from .models.aria...`).

**Solution :** Désactivation du file watcher dans `.streamlit/config.toml` (`fileWatcherType = "none"`). On perd l'auto-reload sur sauvegarde, mais le terminal reste propre et la touche `R` du navigateur permet toujours de re-exécuter le script.

### 6. UX du formulaire de recherche

**Défi :** Avec un `st.text_input` + `st.button` séparés, presser Entrée ne déclenchait pas la recherche — l'utilisateur devait cliquer le bouton. De plus, cliquer une suggestion de la barre latérale lançait bien la recherche, mais ne pré-remplissait pas le champ visible.

**Solution :**
- Utilisation de `st.form` qui rend `Entrée` équivalent au clic du bouton via `st.form_submit_button`.
- Écriture directe dans `st.session_state["search_input"]` (la clé du widget) depuis le handler de chaque suggestion. Streamlit ignore l'argument `value=` une fois la clé existante en session_state, donc passer par une variable proxy ne suffit pas.

## Conclusion

Ce projet nous a permis de mettre en pratique trois familles de techniques d'IA dans un système cohérent. Le **LLM** apporte la compréhension du langage naturel et la rédaction de présentations chaleureuses. Le **RAG** ancre ces présentations sur un corpus réel via une recherche sémantique (k-NN sur embeddings ou TF-IDF), évitant les hallucinations classiques d'un LLM seul. L'**agent** orchestre les trois étapes — *parse → retrieve → generate* — avec une gestion d'erreur en cascade qui dégrade gracieusement plutôt que de casser le pipeline.

Au-delà du fonctionnel, l'application offre une vraie valeur pédagogique : l'utilisateur peut comparer **TF-IDF vs Sentence-Transformers** sur le même dataset, ou observer comment le même backend se comporte sur 96 vs ~30 000 livres. Ce sont exactement les compromis qu'un ingénieur ML doit savoir naviguer en production.
