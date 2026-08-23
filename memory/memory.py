from datetime import datetime


class Memory:
    def __init__(self):
        self.data = []
        self.next_id = 1

    def remember(self, information, category="general", importance=1):
        memory = {
            "id": self.next_id,
            "information": information,
            "category": category,
            "importance": importance,
            "created_at": datetime.now().isoformat(),
        }

        self.data.append(memory)
        self.next_id += 1

        return memory

    def recall(self):
        return self.data

    def get(self, memory_id):
        for memory in self.data:
            if memory["id"] == memory_id:
                return memory

        return None

    def search(self, query):
        query = query.lower()

        return [
            memory
            for memory in self.data
            if query in memory["information"].lower()
        ]

    def by_category(self, category):
        return [
            memory
            for memory in self.data
            if memory["category"].lower() == category.lower()
        ]

    def important(self, minimum=3):
        return [
            memory
            for memory in self.data
            if memory["importance"] >= minimum
        ]

    def forget(self, memory_id):
        for memory in self.data:
            if memory["id"] == memory_id:
                self.data.remove(memory)
                return True

        return False

    def clear(self):
        self.data.clear()

    def count(self):
        return len(self.data)

    def categories(self):
        return list(
            set(memory["category"] for memory in self.data)
        )


memory = Memory()