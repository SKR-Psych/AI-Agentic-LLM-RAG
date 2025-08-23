class MemoryManager:
    def __init__(self):
        self.long_term = {}
        self.short_term = []

    def store(self, key: str, value: str):
        self.long_term[key] = value

    def recall(self, key: str) -> str:
        return self.long_term.get(key, "Not found")

    def buffer(self, item: str):
        self.short_term.append(item)
        if len(self.short_term) > 5:
            self.short_term.pop(0)
