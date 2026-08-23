from core.platform import platform
from ai.assistant import assistant
from memory.memory import Memory
from search.search import Search
from research.researcher import Researcher


memory = Memory()
search = Search()
researcher = Researcher()


def setup_platform():
    platform.register_component(
        "assistant",
        assistant
    )

    platform.register_component(
        "memory",
        memory
    )

    platform.register_component(
        "search",
        search
    )

    platform.register_component(
        "research",
        researcher
    )


def show_status():
    status = platform.status()

    print("\n--- PLATFORM STATUS ---")
    print(f"Name: {status['name']}")
    print(f"Version: {status['version']}")
    print(f"Running: {status['running']}")
    print(
        f"Components: "
        f"{', '.join(status['components'])}"
    )
    print("-----------------------\n")


def remember(text):
    memory.remember(text)

    # Also make the information searchable.
    search.add(
        content=text,
        title="Memory",
        category="memory",
        tags=["memory"]
    )

    print("Information remembered and indexed.")


def search_platform(query):
    results = search.find(query)

    if not results:
        print("No matching information found.")
        return

    print("\n--- SEARCH RESULTS ---")

    for result in results:
        print(
            f"[{result['score']}] "
            f"{result['title']}: "
            f"{result['content']}"
        )

    print("----------------------\n")


def create_research(question):
    task = researcher.create_task(question)

    # Store the research question in memory.
    memory.remember(
        f"Research question: {question}"
    )

    # Make it searchable.
    search.add(
        content=question,
        title="Research Question",
        category="research",
        tags=["research"]
    )

    print(
        f"Research task #{task['id']} created."
    )


def main():
    setup_platform()

    platform.start()

    print("\nWelcome to OurPlatform.")
    print("Type 'help' to see available commands.\n")

    while True:
        command = input("You > ").strip()

        if command.lower() == "exit":
            platform.stop()
            break

        elif command.lower() == "help":
            print("""
Commands:

  status
      Show platform status.

  remember <text>
      Save information to memory.

  memory
      Show stored memories.

  search <text>
      Search indexed information.

  research <question>
      Create a research task.

  components
      Show loaded components.

  health
      Run a platform health check.

  ask <message>
      Send a message to the assistant.

  exit
      Stop OurPlatform.
""")

        elif command.lower() == "status":
            show_status()

        elif command.lower() == "components":
            print(
                "\nComponents:",
                platform.list_components()
            )

        elif command.lower() == "health":
            print(
                "\nHealth:",
                platform.health_check()
            )

        elif command.lower() == "memory":
            memories = memory.recall()

            if not memories:
                print("Memory is empty.")
            else:
                print("\n--- MEMORY ---")

                for item in memories:
                    print(item)

                print("--------------")

        elif command.lower().startswith("remember "):
            text = command[9:].strip()

            if text:
                remember(text)
            else:
                print("Nothing to remember.")

        elif command.lower().startswith("search "):
            query = command[7:].strip()

            if query:
                search_platform(query)
            else:
                print("Please provide a search query.")

        elif command.lower().startswith("research "):
            question = command[9:].strip()

            if question:
                create_research(question)
            else:
                print(
                    "Please provide a research question."
                )

        elif command.lower().startswith("ask "):
            message = command[4:].strip()

            if message:
                print(
                    assistant.respond(message)
                )
            else:
                print("Please provide a message.")

        else:
            print(
                "Unknown command. "
                "Type 'help' for options."
            )


if __name__ == "__main__":
    main()
