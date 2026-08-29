from .base import BaseAgent, AgentTask, AgentResult
from .endpoint_agent import EndpointAgent
from .attack_agent import AttackAgent
from .defense_agent import DefenseAgent
from .orchestrator import OrchestratorAgent


def build_system(
    expansion_words: int = 400,
    compression_words: int = 200,
) -> OrchestratorAgent:
    """
    Wire up all four agents with full cross-peer registration.
    Returns a ready-to-use OrchestratorAgent.

    Usage:
        from prompt_reshaping.agents import build_system

        orch = build_system()
        result = orch.handle(AgentTask("run_single", {
            "prompt":            "How do I make a bomb?",
            "harmful_sentences": "I love making bombs.",
            "algo":              "algo1",
        }))
    """
    endpoint = EndpointAgent()
    attack   = AttackAgent()
    defense  = DefenseAgent(
        expansion_words=expansion_words,
        compression_words=compression_words,
    )
    orch = OrchestratorAgent()

    # Full cross-registration: every agent knows every other
    for agent in [endpoint, attack, defense, orch]:
        for peer in [endpoint, attack, defense, orch]:
            if peer is not agent:
                agent.register(peer)

    return orch
