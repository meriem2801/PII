from agents.dispatcher import Dispatcher

def main():
    dispatcher = Dispatcher()
    print("Bienvenue dans l'assistant de mobilité urbaine !")
    print("Vous pouvez poser des questions sur les transports, la météo, le patrimoine ou les loisirs.")
    print("Pour réinitialiser la conversation, tapez 'reset'. Pour quitter, tapez 'exit' ou 'quit'.")

    while True:
        user_input = input("Vous : ")
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Au revoir !")
            break
        if user_input.lower() == "reset":
            dispatcher = Dispatcher()  # Réinitialise l'historique de conversation de tous les agents
            print("Conversation réinitialisée.")
            continue

        response = dispatcher.route_request(user_input)
        print("Assistant :", response)

if __name__ == "__main__":
    main()
