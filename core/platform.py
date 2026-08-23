class Platform:
    def __init__(self):
        self.name = "OurPlatform"
        self.version = "v0.001"

    def start(self):
        print(f"Welcome to {self.name}")
        print(f"Version: {self.version}")
        print("Platform is starting...")


platform = Platform()