"""
OurPlatform Backend Application

Main backend orchestrator.

Responsibilities:

    - Initialize platform services
    - Connect core systems
    - Register components
    - Register commands
    - Manage application lifecycle
    - Provide CLI interaction
    - Coordinate memory, search, research and AI
    - Provide diagnostics
    - Handle runtime errors
    - Provide application-level events
"""

from datetime import datetime
import traceback

from core.platform import platform
from ai.assistant import assistant
from memory.memory import Memory
from search.search import Search
from research.researcher import Researcher


class Application:
    """
    Main OurPlatform application controller.

    Architecture:

        Application
             |
             +-- Platform
             |
             +-- Assistant
             +-- Memory
             +-- Search
             +-- Research
             |
             +-- Command Registry
             +-- Event Hooks
             +-- Diagnostics
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(self):

        self.name = "OurPlatform"

        self.started_at = None
        self.stopped_at = None

        self.running = False

        self.initialized = False

        self.command_history = []
        self.runtime_errors = []

        self.components = {}

        self.commands = {}

        self._build_components()

    # =========================================================
    # COMPONENT INITIALIZATION
    # =========================================================

    def _build_components(self):

        self.memory = Memory()

        self.search = Search()

        self.researcher = Researcher()

        self.assistant = assistant

        self.components = {
            "assistant": self.assistant,
            "memory": self.memory,
            "search": self.search,
            "research": self.researcher,
        }

    # =========================================================
    # PLATFORM SETUP
    # =========================================================

    def setup_platform(self):

        if self.initialized:

            return False

        for name, component in (
            self.components.items()
        ):

            if not platform.has_component(
                name
            ):

                platform.register_component(
                    name,
                    component,
                )

        self._register_commands()

        self._register_events()

        self.initialized = True

        platform.log(
            "Application initialized."
        )

        return True

    # =========================================================
    # COMMAND SYSTEM
    # =========================================================

    def register_command(
        self,
        name,
        function,
        description="",
        aliases=None,
    ):

        aliases = aliases or []

        self.commands[
            name
        ] = {
            "function": function,
            "description": description,
            "aliases": aliases,
        }

        platform.register_command(
            name,
            function,
            description,
        )

    def _register_commands(self):

        self.register_command(
            "status",
            self.show_status,
            "Show application status.",
            ["info"],
        )

        self.register_command(
            "health",
            self.health,
            "Run a platform health check.",
        )

        self.register_command(
            "components",
            self.show_components,
            "Show loaded components.",
        )

        self.register_command(
            "memory",
            self.show_memory,
            "Show stored memories.",
        )

        self.register_command(
            "remember",
            self.remember_command,
            "Store information in memory.",
        )

        self.register_command(
            "search",
            self.search_command,
            "Search indexed information.",
        )

        self.register_command(
            "research",
            self.research_command,
            "Create a research task.",
        )

        self.register_command(
            "ask",
            self.ask_command,
            "Send a message to the AI assistant.",
        )

        self.register_command(
            "stats",
            self.statistics,
            "Show platform statistics.",
        )

        self.register_command(
            "logs",
            self.show_logs,
            "Show recent platform logs.",
        )

        self.register_command(
            "errors",
            self.show_errors,
            "Show recorded platform errors.",
        )

        self.register_command(
            "backup",
            self.backup_memory,
            "Create a memory backup.",
        )

        self.register_command(
            "help",
            self.show_help,
            "Show available commands.",
        )

    # =========================================================
    # EVENT SYSTEM
    # =========================================================

    def _register_events(self):

        platform.on(
            "component_registered",
            self._component_registered,
        )

        platform.on(
            "component_removed",
            self._component_removed,
        )

        platform.on(
            "feature_enabled",
            self._feature_changed,
        )

        platform.on(
            "feature_disabled",
            self._feature_changed,
        )

    def _component_registered(
        self,
        component,
    ):

        platform.log(
            f"Application detected component: "
            f"{component}"
        )

    def _component_removed(
        self,
        component,
    ):

        platform.log(
            f"Application detected removed "
            f"component: {component}"
        )

    def _feature_changed(
        self,
        feature,
    ):

        platform.log(
            f"Application detected feature "
            f"change: {feature}"
        )

    # =========================================================
    # LIFECYCLE
    # =========================================================

    def start(self):

        if self.running:

            return False

        if not self.initialized:

            self.setup_platform()

        platform.start()

        self.started_at = datetime.now()

        self.running = True

        platform.log(
            "Application started."
        )

        return True

    def stop(self):

        if not self.running:

            return False

        self.running = False

        self.stopped_at = datetime.now()

        platform.log(
            "Application shutting down."
        )

        platform.stop()

        return True

    def restart(self):

        self.stop()

        return self.start()

    # =========================================================
    # STATUS
    # =========================================================

    def show_status(self):

        status = platform.status()

        print(
            "\n--- PLATFORM STATUS ---"
        )

        print(
            f"Name: {status['name']}"
        )

        print(
            f"Version: {status['version']}"
        )

        print(
            f"Running: {status['running']}"
        )

        print(
            f"Application running: "
            f"{self.running}"
        )

        print(
            f"Initialized: "
            f"{self.initialized}"
        )

        print(
            "Components: "
            + ", ".join(
                status["components"]
            )
        )

        print(
            "-----------------------\n"
        )

        return status

    # =========================================================
    # COMPONENTS
    # =========================================================

    def show_components(self):

        components = (
            platform.list_components()
        )

        print(
            "\n--- COMPONENTS ---"
        )

        for name in components:

            component = (
                platform.get_component(
                    name
                )
            )

            print(
                f"{name}: "
                f"{type(component).__name__}"
            )

        print(
            "------------------\n"
        )

        return components

    # =========================================================
    # HEALTH
    # =========================================================

    def health(self):

        result = (
            platform.health_check()
        )

        print(
            "\n--- HEALTH ---"
        )

        print(
            f"Healthy: "
            f"{result['healthy']}"
        )

        print(
            f"Platform running: "
            f"{result['platform_running']}"
        )

        print(
            f"Errors: "
            f"{result['errors']}"
        )

        print(
            "---------------\n"
        )

        return result

    # =========================================================
    # MEMORY
    # =========================================================

    def remember(
        self,
        text,
        category="general",
        importance=1,
        tags=None,
    ):

        memory = self.memory.remember(
            text,
            category=category,
            importance=importance,
            tags=tags,
        )

        # Keep search synchronized.
        try:

            self.search.add(
                content=text,
                title="Memory",
                category="memory",
                tags=["memory"],
            )

        except Exception as error:

            self._record_error(
                error
            )

        platform.log(
            "Information remembered."
        )

        return memory

    def remember_command(
        self,
        text="",
    ):

        text = str(
            text
        ).strip()

        if not text:

            print(
                "Usage: remember <text>"
            )

            return None

        result = self.remember(
            text
        )

        print(
            "Information remembered "
            "and indexed."
        )

        return result

    def show_memory(self):

        memories = (
            self.memory.recall()
        )

        if not memories:

            print(
                "Memory is empty."
            )

            return []

        print(
            "\n--- MEMORY ---"
        )

        for item in memories:

            print(
                f"[{item['id']}] "
                f"{item['category']} "
                f"(importance="
                f"{item['importance']}): "
                f"{item['information']}"
            )

        print(
            "--------------\n"
        )

        return memories

    # =========================================================
    # SEARCH
    # =========================================================

    def search_platform(
        self,
        query,
    ):

        query = str(
            query
        ).strip()

        if not query:

            return []

        results = (
            self.search.find(
                query
            )
        )

        if not results:

            print(
                "No matching information "
                "found."
            )

            return []

        print(
            "\n--- SEARCH RESULTS ---"
        )

        for result in results:

            print(
                f"[{result.get('score', 0)}] "
                f"{result.get('title', 'Untitled')}: "
                f"{result.get('content', '')}"
            )

        print(
            "----------------------\n"
        )

        return results

    def search_command(
        self,
        query="",
    ):

        return self.search_platform(
            query
        )

    # =========================================================
    # RESEARCH
    # =========================================================

    def create_research(
        self,
        question,
    ):

        question = str(
            question
        ).strip()

        if not question:

            return None

        task = (
            self.researcher.create_task(
                question
            )
        )

        self.memory.remember(
            f"Research question: "
            f"{question}",
            category="research",
            importance=2,
            tags=["research"],
        )

        try:

            self.search.add(
                content=question,
                title="Research Question",
                category="research",
                tags=["research"],
            )

        except Exception as error:

            self._record_error(
                error
            )

        platform.log(
            f"Research task created: "
            f"{task['id']}"
        )

        return task

    def research_command(
        self,
        question="",
    ):

        question = str(
            question
        ).strip()

        if not question:

            print(
                "Usage: research <question>"
            )

            return None

        task = self.create_research(
            question
        )

        print(
            f"Research task "
            f"#{task['id']} created."
        )

        return task

    # =========================================================
    # AI ASSISTANT
    # =========================================================

    def ask(
        self,
        message,
    ):

        message = str(
            message
        ).strip()

        if not message:

            return None

        try:

            response = (
                self.assistant.respond(
                    message
                )
            )

            return response

        except Exception as error:

            self._record_error(
                error
            )

            return (
                "The assistant encountered "
                "an error."
            )

    def ask_command(
        self,
        message="",
    ):

        message = str(
            message
        ).strip()

        if not message:

            print(
                "Usage: ask <message>"
            )

            return None

        response = self.ask(
            message
        )

        print(
            response
        )

        return response

    # =========================================================
    # STATISTICS
    # =========================================================

    def statistics(self):

        result = {
            "application": {
                "running": self.running,
                "initialized": (
                    self.initialized
                ),
                "started_at": (
                    self.started_at.isoformat()
                    if self.started_at
                    else None
                ),
            },

            "memory": (
                self.memory.statistics()
            ),

            "platform": (
                platform.status()
            ),
        }

        print(
            "\n--- STATISTICS ---"
        )

        print(
            f"Memories: "
            f"{result['memory']['total_memories']}"
        )

        print(
            f"Categories: "
            f"{result['memory']['categories']}"
        )

        print(
            f"Tags: "
            f"{result['memory']['tags']}"
        )

        print(
            "------------------\n"
        )

        return result

    # =========================================================
    # LOGS
    # =========================================================

    def show_logs(
        self,
        limit=20,
    ):

        logs = platform.get_logs()

        logs = logs[
            -int(limit):
        ]

        print(
            "\n--- LOGS ---"
        )

        for entry in logs:

            print(
                f"[{entry['time']}] "
                f"{entry['message']}"
            )

        print(
            "------------\n"
        )

        return logs

    # =========================================================
    # ERRORS
    # =========================================================

    def show_errors(
        self,
        limit=20,
    ):

        errors = platform.get_errors()

        errors = errors[
            -int(limit):
        ]

        print(
            "\n--- ERRORS ---"
        )

        if not errors:

            print(
                "No recorded errors."
            )

        for error in errors:

            print(
                f"[{error['time']}] "
                f"{error['type']}: "
                f"{error['message']}"
            )

        print(
            "--------------\n"
        )

        return errors

    # =========================================================
    # MEMORY BACKUP
    # =========================================================

    def backup_memory(
        self,
        destination=None,
    ):

        result = (
            self.memory.backup(
                destination
            )
        )

        print(
            f"Memory backup created: "
            f"{result}"
        )

        return result

    # =========================================================
    # COMMAND HISTORY
    # =========================================================

    def _record_command(
        self,
        command,
        arguments,
    ):

        self.command_history.append(
            {
                "time": (
                    datetime.now().isoformat()
                ),
                "command": command,
                "arguments": arguments,
            }
        )

        # Prevent unlimited growth.
        if len(
            self.command_history
        ) > 1000:

            self.command_history = (
                self.command_history[
                    -1000:
                ]
            )

    # =========================================================
    # ERROR RECORDING
    # =========================================================

    def _record_error(
        self,
        error,
    ):

        entry = {
            "time": (
                datetime.now().isoformat()
            ),
            "type": type(
                error
            ).__name__,
            "message": str(
                error
            ),
            "traceback": (
                traceback.format_exc()
            ),
        }

        self.runtime_errors.append(
            entry
        )

        platform.record_error(
            error
        )

    # =========================================================
    # COMMAND RESOLUTION
    # =========================================================

    def resolve_command(
        self,
        name,
    ):

        if name in self.commands:

            return self.commands[
                name
            ]

        for command in (
            self.commands.values()
        ):

            if name in command[
                "aliases"
            ]:

                return command

        return None

    # =========================================================
    # COMMAND EXECUTION
    # =========================================================

    def execute(
        self,
        command,
        *arguments,
    ):

        command = str(
            command
        ).lower().strip()

        self._record_command(
            command,
            arguments,
        )

        entry = (
            self.resolve_command(
                command
            )
        )

        if entry is None:

            print(
                "Unknown command. "
                "Type 'help' for options."
            )

            return None

        try:

            return entry[
                "function"
            ](
                *arguments
            )

        except Exception as error:

            self._record_error(
                error
            )

            print(
                f"Command error: "
                f"{error}"
            )

            return None

    # =========================================================
    # HELP
    # =========================================================

    def show_help(self):

        print(
            """
--- OURPLATFORM COMMANDS ---

status
    Show platform status.

health
    Run a platform health check.

components
    Show loaded components.

remember <text>
    Save information to memory.

memory
    Show stored memories.

search <text>
    Search indexed information.

research <question>
    Create a research task.

ask <message>
    Send a message to the AI assistant.

stats
    Show platform statistics.

logs
    Show recent platform logs.

errors
    Show recorded platform errors.

backup
    Create a memory backup.

help
    Show this help screen.

exit
    Stop OurPlatform.

----------------------------
"""
        )

    # =========================================================
    # CLI LOOP
    # =========================================================

    def run_cli(self):

        print(
            "\nWelcome to OurPlatform."
        )

        print(
            "Type 'help' to see "
            "available commands.\n"
        )

        while self.running:

            try:

                raw = input(
                    "You > "
                ).strip()

            except (
                EOFError,
                KeyboardInterrupt,
            ):

                print()

                self.stop()

                break

            if not raw:

                continue

            if raw.lower() == "exit":

                self.stop()

                break

            parts = raw.split()

            command = parts[0]

            arguments = (
                raw[len(command):]
                .strip()
            )

            # Most commands expect their
            # remaining text as one argument.
            if arguments:

                self.execute(
                    command,
                    arguments,
                )

            else:

                self.execute(
                    command
                )

    # =========================================================
    # FULL APPLICATION STATUS
    # =========================================================

    def diagnostics(self):

        return {
            "application": {
                "name": self.name,
                "running": self.running,
                "initialized": (
                    self.initialized
                ),
                "started_at": (
                    self.started_at.isoformat()
                    if self.started_at
                    else None
                ),
                "stopped_at": (
                    self.stopped_at.isoformat()
                    if self.stopped_at
                    else None
                ),
            },

            "platform": (
                platform.debug_info()
            ),

            "memory": (
                self.memory.status()
            ),

            "components": {
                name: type(
                    component
                ).__name__
                for name, component
                in self.components.items()
            },

            "command_count": len(
                self.commands
            ),

            "command_history": len(
                self.command_history
            ),

            "runtime_errors": len(
                self.runtime_errors
            ),
        }


# =============================================================
# GLOBAL APPLICATION INSTANCE
# =============================================================

app = Application()


# =============================================================
# COMPATIBILITY FUNCTIONS
# =============================================================

def setup_platform():

    return app.setup_platform()


def show_status():

    return app.show_status()


def remember(text):

    return app.remember(
        text
    )


def search_platform(query):

    return app.search_platform(
        query
    )


def create_research(question):

    return app.create_research(
        question
    )


# =============================================================
# ENTRY POINT
# =============================================================

def main():

    app.start()

    app.run_cli()


if __name__ == "__main__":

    main()