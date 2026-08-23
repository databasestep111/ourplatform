class Assistant:
    def __init__(self):
        self.name = "OurPlatform Assistant"

    def respond(self, message):
        return f"{self.name} received: {message}"


assistant = Assistant()