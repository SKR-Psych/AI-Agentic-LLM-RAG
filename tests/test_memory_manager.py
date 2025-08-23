from ai_agentic_core.memory_manager import MemoryManager

def test_store_and_recall():
    memory = MemoryManager()
    memory.store("language", "Python")
    assert memory.recall("language") == "Python"
