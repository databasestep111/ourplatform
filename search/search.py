class Search:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)

    def find(self, query):
        results = []

        for item in self.items:
            if query.lower() in item.lower():
                results.append(item)

        return results


search = Search()