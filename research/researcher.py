class Researcher:
    def __init__(self):
        self.name = "OurPlatform Researcher"

    def create_task(self, question):
        return {
            "question": question,
            "status": "waiting_for_research"
        }


researcher = Researcher()