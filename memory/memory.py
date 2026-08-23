class Memory:
    def __init__(self):
        self.data = []

    def remember(self, information):
        self.data.append(information)

    def recall(self):
        return self.data