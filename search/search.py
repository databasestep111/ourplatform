from datetime import datetime
import re


class Search:
    def __init__(self):
        self.items = []
        self.next_id = 1

    def add(
        self,
        content,
        title="Untitled",
        category="general",
        tags=None,
    ):
        if tags is None:
            tags = []

        item = {
            "id": self.next_id,
            "title": title,
            "content": content,
            "category": category,
            "tags": tags,
            "created_at": datetime.now().isoformat(),
        }

        self.items.append(item)
        self.next_id += 1

        return item

    def remove(self, item_id):
        for item in self.items:
            if item["id"] == item_id:
                self.items.remove(item)
                return True

        return False

    def get(self, item_id):
        for item in self.items:
            if item["id"] == item_id:
                return item

        return None

    def update(
        self,
        item_id,
        title=None,
        content=None,
        category=None,
        tags=None,
    ):
        item = self.get(item_id)

        if item is None:
            return None

        if title is not None:
            item["title"] = title

        if content is not None:
            item["content"] = content

        if category is not None:
            item["category"] = category

        if tags is not None:
            item["tags"] = tags

        return item

    def tokenize(self, text):
        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )

    def exact_match(self, text, query):
        return query.lower() in text.lower()

    def word_match_count(self, text, query):
        words = self.tokenize(text)
        query_words = self.tokenize(query)

        count = 0

        for word in query_words:
            count += words.count(word)

        return count

    def score(self, item, query):
        score = 0

        title = item["title"]
        content = item["content"]
        tags = item["tags"]

        if self.exact_match(title, query):
            score += 20

        if self.exact_match(content, query):
            score += 10

        if any(
            query.lower() == tag.lower()
            for tag in tags
        ):
            score += 15

        score += self.word_match_count(
            title,
            query
        ) * 5

        score += self.word_match_count(
            content,
            query
        )

        return score

    def find(
        self,
        query,
        category=None,
        tags=None,
        limit=10,
    ):
        results = []

        for item in self.items:

            if category is not None:
                if item["category"].lower() != category.lower():
                    continue

            if tags:
                item_tags = [
                    tag.lower()
                    for tag in item["tags"]
                ]

                if not any(
                    tag.lower() in item_tags
                    for tag in tags
                ):
                    continue

            score = self.score(item, query)

            if score > 0:
                result = item.copy()
                result["score"] = score
                results.append(result)

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:limit]

    def search_title(self, query, limit=10):
        results = []

        for item in self.items:
            if self.exact_match(
                item["title"],
                query
            ):
                results.append(item)

        return results[:limit]

    def search_content(self, query, limit=10):
        results = []

        for item in self.items:
            if self.exact_match(
                item["content"],
                query
            ):
                results.append(item)

        return results[:limit]

    def by_category(self, category):
        return [
            item
            for item in self.items
            if item["category"].lower()
            == category.lower()
        ]

    def by_tag(self, tag):
        return [
            item
            for item in self.items
            if any(
                tag.lower() == item_tag.lower()
                for item_tag in item["tags"]
            )
        ]

    def has_duplicate(self, content):
        for item in self.items:
            if item["content"].strip().lower() == \
               content.strip().lower():
                return True

        return False

    def count(self):
        return len(self.items)

    def categories(self):
        return list(
            set(
                item["category"]
                for item in self.items
            )
        )

    def tags(self):
        all_tags = []

        for item in self.items:
            for tag in item["tags"]:
                if tag not in all_tags:
                    all_tags.append(tag)

        return all_tags

    def statistics(self):
        return {
            "total_items": self.count(),
            "categories": self.categories(),
            "tags": self.tags(),
        }

    def clear(self):
        self.items.clear()
        self.next_id = 1


search = Search()