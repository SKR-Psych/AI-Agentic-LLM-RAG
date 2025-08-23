from ai_agentic_core.reasoning_engine import ReasoningEngine

def test_think():
    engine = ReasoningEngine()
    result = engine.think("What is truth?")
    assert "Processed reasoning" in result
