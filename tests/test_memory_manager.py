from ai_agentic_core.memory_manager import MemoryManager

def test_store_and_recall():
    memory = MemoryManager()
    memory.store("language", "Python")
    assert memory.recall("language") == "Python"


def refresh_data():
    # TODO: logic pending
    pass



def build_data():
    # TODO: logic pending
    pass



def calculate_data():
    # TODO: logic pending
    pass



def calculate_timeout():
    # TODO: logic pending
    pass


def update_memory_importance(memory_id, new_importance):
    """Update importance score of a memory item."""
    if memory_id in self.memories:
        self.memories[memory_id].importance = max(0.0, min(1.0, new_importance))
        self.memories[memory_id].last_updated = time.time()
        return True
    return False

