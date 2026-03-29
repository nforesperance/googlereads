# Blog — Projet de Recommandation de Livres par IA

## Introduction

Ce projet est une application de recommandation de livres qui permet à un utilisateur de décrire ses goûts littéraires en langage naturel et de recevoir 5 recommandations personnalisées. L'application combine trois technologies d'intelligence artificielle : un modèle de langage (LLM), un algorithme de machine learning (ML), et un agent intelligent qui orchestre le tout.

## Technologies choisies

### LLM — Groq API (Llama 3.3 70B)

Nous avons choisi l'API Groq pour sa gratuité et sa rapidité. Le modèle Llama 3.3 70B est utilisé pour analyser le texte de l'utilisateur et en extraire des informations structurées : les genres littéraires préférés et l'humeur souhaitée.

**Pourquoi Groq ?** L'API est gratuite, rapide, et compatible avec les modèles open source les plus performants. Contrairement à OpenAI, elle ne nécessite pas de carte de crédit pour commencer.

**Rôle dans le projet :** Le module `llm_parser.py` envoie le message de l'utilisateur au LLM avec un prompt système précis qui demande une réponse en JSON. Par exemple, si l'utilisateur dit "j'aime les romans policiers sombres", le LLM retourne `{"genres": ["mystery", "thriller", "crime"], "mood": "sombre"}`.

### ML — k-NN avec TF-IDF (scikit-learn)

Le machine learning est utilisé pour trouver les livres les plus similaires au profil de l'utilisateur. Nous utilisons deux techniques combinées :

- **TF-IDF (Term Frequency-Inverse Document Frequency)** : transforme les genres et descriptions des livres en vecteurs numériques. Les mots rares mais distinctifs reçoivent un poids plus élevé.
- **k-NN (k-Nearest Neighbors)** : trouve les 5 vecteurs les plus proches du vecteur de requête de l'utilisateur, en utilisant la distance cosinus.

**Pourquoi k-NN ?** C'est un algorithme simple, interprétable et efficace pour les systèmes de recommandation basés sur le contenu. Il ne nécessite pas d'entraînement préalable et fonctionne bien avec des datasets de taille moyenne.

**Rôle dans le projet :** Le module `recommender.py` charge le dataset de livres, vectorise les genres et descriptions, et retourne les 5 livres les plus pertinents pour un profil donné.

### Agent intelligent

L'agent est le chef d'orchestre du système. Il coordonne les appels entre le LLM et le ML, gère l'état de la conversation, et retourne des résultats structurés exploitables par l'interface.

**Rôle dans le projet :** Le module `agent.py` reçoit le message de l'utilisateur, appelle d'abord le LLM pour extraire les préférences, puis passe ces préférences au module ML pour obtenir des recommandations. Il retourne un objet `AgentResult` contenant les livres, les genres détectés et le mood — ce qui permet à l'interface Streamlit d'afficher des cartes riches.

### Interface utilisateur — Streamlit

Nous avons choisi Streamlit pour créer une interface web moderne et interactive. L'interface offre :
- Un champ de recherche en langage naturel
- Des suggestions cliquables dans la barre latérale
- Des cartes de livres avec genres (pills colorées), description et barre de pertinence
- Un historique des recherches
- La saisie sécurisée de la clé API directement dans l'interface

**Pourquoi Streamlit ?** C'est le framework Python le plus rapide pour créer des interfaces de données interactives. Il gère nativement le state management via `st.session_state`, ce qui est parfait pour notre historique de conversation.

## Architecture du système

```
Utilisateur
    │
    ▼
┌─────────┐     ┌─────────────┐     ┌──────────────┐
│  Agent   │────▶│  LLM Parser │────▶│  API Groq    │
│ agent.py │     │ llm_parser  │     │  (Llama 3.3) │
│          │     └─────────────┘     └──────────────┘
│          │            │
│          │     {"genres": [...], "mood": "..."}
│          │            │
│          │     ┌──────────────┐
│          │────▶│ Recommender  │
│          │     │ recommender  │
│          │     │ (k-NN+TF-IDF)│
└─────────┘     └──────────────┘
    │
    ▼
 Réponse formatée avec 5 livres
```

## Défis rencontrés et solutions

### 1. Extraction fiable du JSON depuis le LLM

**Défi :** Les modèles de langage ne retournent pas toujours du JSON parfaitement formaté. Ils ajoutent parfois du texte explicatif avant ou après le JSON.

**Solution :** Nous avons implémenté une extraction robuste qui cherche le premier `{` et le dernier `}` dans la réponse, puis parse uniquement cette portion. Un fallback retourne des valeurs par défaut si le parsing échoue.

### 2. Qualité des recommandations

**Défi :** Avec un dataset relativement petit, les recommandations pouvaient manquer de pertinence, surtout pour des genres niche.

**Solution :** Nous combinons les genres ET les descriptions des livres dans le vecteur TF-IDF, ce qui donne plus de contexte au modèle k-NN. Le mood de l'utilisateur est aussi intégré dans la requête pour affiner les résultats.

### 3. Gestion des erreurs en cascade

**Défi :** Si le LLM échoue (pas de clé API, erreur réseau), tout le pipeline tombe.

**Solution :** L'agent gère les erreurs à chaque étape avec des messages explicites pour l'utilisateur, sans faire crasher l'application. Un input vide retourne des valeurs par défaut plutôt qu'une erreur.

### 4. Internationalisation

**Défi :** L'utilisateur écrit en français, mais les genres du dataset sont en anglais.

**Solution :** Le LLM sert de traducteur implicite : il comprend le français de l'utilisateur et retourne des genres en anglais standardisés qui correspondent au vocabulaire du dataset.

## Conclusion

Ce projet nous a permis d'explorer la combinaison de plusieurs technologies d'IA dans une application cohérente. Le LLM apporte la compréhension du langage naturel, le ML apporte la capacité de recommandation basée sur les données, et l'agent apporte l'orchestration intelligente. Le résultat est une application simple mais fonctionnelle qui démontre comment ces technologies se complètent.
