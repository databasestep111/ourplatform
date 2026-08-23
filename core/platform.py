from datetime import datetime
import traceback

from config import APP_NAME, VERSION, DEBUG, FEATURES


class Platform:
    def __init__(self):
        self.name = APP_NAME
        self.version = VERSION
        self.debug = DEBUG
        self.features = FEATURES.copy()

        self.running = False
        self.started_at = None

        self.components = {}
        self.commands = {}
        self.events = {}

        self.logs = []
        self.errors = []

    # -------------------------
    # LIFECYCLE
    # -------------------------

    def start(self):
        if self.running:
            self.log("Platform is already running.")
            return False

        self.started_at = datetime.now()
        self.running = True

        self.log(
            f"{self.name} {self.version} started."
        )

        return True

    def stop(self):
        if not self.running:
            self.log("Platform is already stopped.")
            return False

        self.running = False
        self.log("Platform stopped.")

        return True

    def restart(self):
        self.stop()
        return self.start()

    # -------------------------
    # COMPONENT SYSTEM
    # -------------------------

    def register_component(self, name, component):
        if not name:
            raise ValueError(
                "Component name cannot be empty."
            )

        self.components[name] = component

        self.log(
            f"Component registered: {name}"
        )

        self.emit(
            "component_registered",
            name
        )

    def unregister_component(self, name):
        if name in self.components:
            del self.components[name]

            self.log(
                f"Component removed: {name}"
            )

            self.emit(
                "component_removed",
                name
            )

            return True

        return False

    def get_component(self, name):
        return self.components.get(name)

    def has_component(self, name):
        return name in self.components

    def list_components(self):
        return list(self.components.keys())

    # -------------------------
    # FEATURE SYSTEM
    # -------------------------

    def feature_enabled(self, feature):
        return self.features.get(
            feature,
            False
        )

    def enable_feature(self, feature):
        self.features[feature] = True

        self.log(
            f"Feature enabled: {feature}"
        )

        self.emit(
            "feature_enabled",
            feature
        )

    def disable_feature(self, feature):
        self.features[feature] = False

        self.log(
            f"Feature disabled: {feature}"
        )

        self.emit(
            "feature_disabled",
            feature
        )

    def list_features(self):
        return self.features.copy()

    # -------------------------
    # COMMAND SYSTEM
    # -------------------------

    def register_command(
        self,
        name,
        function,
        description=""
    ):
        self.commands[name] = {
            "function": function,
            "description": description,
        }

        self.log(
            f"Command registered: {name}"
        )

    def execute_command(self, name, *args, **kwargs):
        command = self.commands.get(name)

        if command is None:
            return {
                "success": False,
                "error": f"Unknown command: {name}",
            }

        try:
            result = command["function"](
                *args,
                **kwargs
            )

            return {
                "success": True,
                "result": result,
            }

        except Exception as error:
            self.record_error(error)

            return {
                "success": False,
                "error": str(error),
            }

    def list_commands(self):
        return {
            name: data["description"]
            for name, data in self.commands.items()
        }

    # -------------------------
    # EVENT SYSTEM
    # -------------------------

    def on(self, event_name, listener):
        if event_name not in self.events:
            self.events[event_name] = []

        self.events[event_name].append(
            listener
        )

    def emit(self, event_name, data=None):
        listeners = self.events.get(
            event_name,
            []
        )

        for listener in listeners:
            try:
                listener(data)
            except Exception as error:
                self.record_error(error)

    # -------------------------
    # LOGGING
    # -------------------------

    def log(self, message):
        timestamp = datetime.now().isoformat()

        entry = {
            "time": timestamp,
            "message": message,
        }

        self.logs.append(entry)

        if self.debug:
            print(
                f"[LOG] {message}"
            )

    def get_logs(self):
        return self.logs.copy()

    def clear_logs(self):
        self.logs.clear()

    # -------------------------
    # ERROR HANDLING
    # -------------------------

    def record_error(self, error):
        entry = {
            "time": datetime.now().isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

        self.errors.append(entry)

        if self.debug:
            print(
                f"[ERROR] {error}"
            )

    def get_errors(self):
        return self.errors.copy()

    def clear_errors(self):
        self.errors.clear()

    # -------------------------
    # STATUS
    # -------------------------

    def status(self):
        return {
            "name": self.name,
            "version": self.version,
            "running": self.running,
            "debug": self.debug,
            "components": self.list_components(),
            "commands": list(
                self.commands.keys()
            ),
            "features": self.features.copy(),
            "started_at": self.started_at,
        }

    # -------------------------
    # HEALTH CHECK
    # -------------------------

    def health_check(self):
        component_status = {}

        for name, component in self.components.items():
            component_status[name] = {
                "loaded": component is not None,
                "type": type(component).__name__,
            }

        return {
            "platform_running": self.running,
            "components": component_status,
            "errors": len(self.errors),
            "healthy": (
                self.running
                and len(self.errors) == 0
            ),
        }

    # -------------------------
    # DEBUG INFORMATION
    # -------------------------

    def debug_info(self):
        return {
            "status": self.status(),
            "health": self.health_check(),
            "logs": self.get_logs(),
            "errors": self.get_errors(),
        }


platform = Platform()