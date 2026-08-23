from core.platform import platform
from ai.assistant import assistant
from memory.memory import Memory
from search.search import Search
from research.researcher import researcher


memory = Memory()
search = Search()


def main():
    platform.start()

    print("\nOurPlatform is ready.")
    print("Type 'help' to see what I can do.")
    print("Type 'exit' to stop.\n")

    while True:
        command = input("You > ").strip()

        if command.lower() == "exit":
            print("Goodbye.")
            break

        elif command.lower() == "help":
            print("\nCommands:")
            print("  ask <message>     - talk to the assistant")
            print("  remember <text>   - save something")
            print("  memory            - show saved information")
            print("  search <text>     - search saved information")
            print("  research <text>   - create a research task")
            print("  exit              - close OurPlatform\n")

        elif command.lower().startswith("ask "):
            message = command[4:]
            print(assistant.respond(message))

        elif command.lower().startswith("remember "):
            information = command[9:]
            memory.remember(information)
            print("Saved to memory.")

        elif command.lower() == "memory":
            print("Memory:", memory.recall())

        elif command.lower().startswith("search "):
            query = command[7:]
            print("Search results:", search.find(query))

        elif command.lower().startswith("research "):
            question = command[9:]
            print(researcher.create_task(question))

        else:
            print("I don't recognize that command. Type 'help'.")


if __name__ == "__main__":
    main()
