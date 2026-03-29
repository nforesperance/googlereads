"""
Book Recommender — Point d'entrée.

Lance l'interface Streamlit ou la CLI selon les arguments.
  - streamlit run app.py   → interface web (recommandé)
  - python main.py         → interface en ligne de commande
"""

import sys

from src.agent import BookAgent


WELCOME_MESSAGE = """
╔══════════════════════════════════════════════════════════════╗
║           📚  Recommandation de Livres — IA  📚            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Décrivez vos goûts littéraires en langage naturel           ║
║  et recevez 5 recommandations personnalisées !               ║
║                                                              ║
║  Exemples :                                                  ║
║    • "j'aime les romans policiers sombres"                   ║
║    • "quelque chose de léger et drôle"                       ║
║    • "de la science-fiction épique dans l'espace"             ║
║                                                              ║
║  Commandes : 'quitter' pour sortir | 'reset' pour relancer  ║
║                                                              ║
║  💡 Pour l'interface web : streamlit run app.py              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    print(WELCOME_MESSAGE)

    agent = BookAgent()

    while True:
        try:
            user_input = input("\n🔎 Vos goûts → ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nAu revoir ! Bonne lecture ! 📖")
            sys.exit(0)

        if not user_input:
            print("Veuillez décrire vos goûts littéraires.")
            continue

        if user_input.lower() in ("quitter", "quit", "exit", "q"):
            print("\nAu revoir ! Bonne lecture ! 📖")
            break

        if user_input.lower() == "reset":
            agent.reset()
            print("Conversation réinitialisée.")
            continue

        print("\n⏳ Analyse de vos préférences en cours...\n")
        result = agent.run(user_input)
        print(result.format_text())


if __name__ == "__main__":
    main()
