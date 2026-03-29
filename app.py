"""
Book Recommender — Interface Streamlit.

Application web interactive de recommandation de livres
combinant LLM (Groq) + ML (k-NN / TF-IDF) + Agent intelligent.
"""

import streamlit as st

from src.agent import BookAgent


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Book Recommender IA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }

    /* ── Hero header ── */
    .hero {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        font-size: 1.15rem;
        opacity: 0.7;
    }

    /* ── Book card ── */
    .book-card {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(145deg,
            rgba(102,126,234,0.04) 0%,
            rgba(118,75,162,0.04) 100%);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .book-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.15);
    }
    .book-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .book-author {
        font-size: 0.95rem;
        opacity: 0.7;
        margin-bottom: 0.6rem;
    }
    .book-desc {
        font-size: 0.88rem;
        opacity: 0.8;
        line-height: 1.5;
        margin-bottom: 0.7rem;
    }

    /* ── Genre pill ── */
    .genre-pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        margin: 0.15rem 0.2rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    /* ── Score bar ── */
    .score-container {
        margin-top: 0.6rem;
    }
    .score-bar-bg {
        height: 6px;
        border-radius: 3px;
        background: rgba(128,128,128,0.15);
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    .score-label {
        font-size: 0.78rem;
        font-weight: 600;
        opacity: 0.6;
        margin-bottom: 0.2rem;
    }

    /* ── Detected prefs box ── */
    .pref-box {
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.5rem;
        background: rgba(102,126,234,0.06);
    }
    .pref-box strong {
        color: #667eea;
    }

    /* ── Sidebar ── */
    .sidebar-section {
        padding: 0.8rem 0;
        border-bottom: 1px solid rgba(128,128,128,0.15);
    }
    .sidebar-section:last-child {
        border-bottom: none;
    }

    /* ── Suggestion chips ── */
    div.stButton > button {
        border-radius: 20px;
        font-size: 0.85rem;
    }

    /* ── History entry ── */
    .history-entry {
        padding: 0.6rem 0.8rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        background: rgba(128,128,128,0.06);
        font-size: 0.85rem;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ───────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = BookAgent()
if "results" not in st.session_state:
    st.session_state.results = None
if "history" not in st.session_state:
    st.session_state.history = []
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "submit_query" not in st.session_state:
    st.session_state.submit_query = None


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "Clé API Groq",
        type="password",
        placeholder="gsk_...",
        help="Créez un compte gratuit sur console.groq.com",
    )
    if api_key:
        import os
        os.environ["GROQ_API_KEY"] = api_key
        st.success("Clé API configurée", icon="✅")
    else:
        st.info("Entrez votre clé API Groq pour commencer", icon="🔑")

    st.markdown("---")

    st.markdown("## 💡 Essayez par exemple")
    suggestions = [
        "J'aime les romans policiers sombres",
        "Quelque chose de léger et drôle",
        "De la science-fiction épique",
        "Un roman d'amour historique",
        "Du fantasy avec de la magie",
        "Un thriller psychologique intense",
    ]
    for s in suggestions:
        if st.button(s, key=f"sug_{s}", use_container_width=True):
            st.session_state.submit_query = s

    st.markdown("---")

    # History
    if st.session_state.history:
        st.markdown("## 📜 Historique")
        for i, entry in enumerate(reversed(st.session_state.history)):
            with st.container():
                st.caption(f"🔎 {entry['query']}")
                genres_str = ", ".join(entry["genres"])
                st.caption(f"→ {genres_str}")
        if st.button("🗑️ Effacer l'historique", use_container_width=True):
            st.session_state.history = []
            st.session_state.results = None
            st.session_state.agent.reset()
            st.rerun()

    st.markdown("---")
    st.markdown("## 🏗️ Technologies")
    st.markdown("""
    - **LLM** — Groq (Llama 3.3 70B)
    - **ML** — k-NN + TF-IDF (scikit-learn)
    - **Agent** — Orchestrateur Python
    - **UI** — Streamlit
    """)


# ── Hero ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📚 Book Recommender IA</h1>
    <p>Décrivez vos goûts littéraires et recevez 5 recommandations personnalisées</p>
</div>
""", unsafe_allow_html=True)


# ── Resolve suggestion click before rendering input ──────────
# If a sidebar suggestion was clicked, pre-fill and auto-submit
query_to_run = None
if st.session_state.submit_query:
    query_to_run = st.session_state.submit_query
    st.session_state.current_query = query_to_run
    st.session_state.submit_query = None

# ── Search bar ───────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])

with col_input:
    user_input = st.text_input(
        "Vos goûts littéraires",
        placeholder="Ex : j'aime les romans policiers sombres...",
        label_visibility="collapsed",
        key="search_input",
        value=st.session_state.current_query,
    )

with col_btn:
    search_clicked = st.button("🔍 Chercher", type="primary", use_container_width=True)

# Also run on search button click
if not query_to_run and search_clicked and user_input.strip():
    query_to_run = user_input.strip()

# ── Process query ────────────────────────────────────────────
if query_to_run:
    if not api_key:
        st.error("Veuillez entrer votre clé API Groq dans la barre latérale.")
    else:
        with st.spinner("Analyse de vos préférences par l'IA..."):
            result = st.session_state.agent.run(query_to_run)

        if result.success:
            st.session_state.results = result
            st.session_state.history.append({
                "query": query_to_run,
                "genres": result.genres,
                "mood": result.mood,
            })
        else:
            st.error(result.error)
            st.session_state.results = None

        st.rerun()


# ── Display results ──────────────────────────────────────────
result = st.session_state.results

if result and result.success:
    # Detected preferences
    genre_pills = "".join(
        f'<span class="genre-pill">{g}</span>' for g in result.genres
    )
    mood_html = f' &nbsp;·&nbsp; Ambiance : <strong>{result.mood}</strong>' if result.mood else ""

    st.markdown(f"""
    <div class="pref-box">
        🎯 <strong>Préférences détectées</strong><br>
        {genre_pills}{mood_html}
    </div>
    """, unsafe_allow_html=True)

    # Book cards — 2 columns top, then remaining
    st.markdown("### Vos recommandations")

    from src.recommender import get_recommender
    rec = get_recommender()

    for i, book in enumerate(result.books):
        # Find description from dataset
        match = rec.df[rec.df["title"] == book["title"]]
        desc = match.iloc[0]["description"] if not match.empty else ""

        score_pct = int(book["score"] * 100)
        genre_list = [g.strip() for g in book["genres"].split(",")]
        pills = "".join(f'<span class="genre-pill">{g}</span>' for g in genre_list)

        st.markdown(f"""
        <div class="book-card">
            <div class="book-title">📖 {book['title']}</div>
            <div class="book-author">par {book['author']}</div>
            <div class="book-desc">{desc}</div>
            <div>{pills}</div>
            <div class="score-container">
                <div class="score-label">Pertinence : {score_pct}%</div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width: {score_pct}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 🧠 LLM
        L'IA comprend vos goûts décrits en langage naturel et extrait
        les genres et l'ambiance souhaitée.
        """)

    with col2:
        st.markdown("""
        #### 📊 Machine Learning
        L'algorithme k-NN analyse 96 livres vectorisés par TF-IDF
        pour trouver les plus similaires à votre profil.
        """)

    with col3:
        st.markdown("""
        #### 🤖 Agent
        L'agent orchestre le LLM et le ML, gère la conversation
        et formate vos recommandations.
        """)
