import random

from src.agent import Agent
from src.constants import AgentAction


class RLAgent(Agent):
    """
    Puppet agent for RL training.

    The Gymnasium env wrapper writes the policy's chosen action to
    `_next_action` immediately before each `Simulator.step()`. `get_action`
    consumes the slot and returns it; if nothing set the slot (e.g. when
    the agent is run outside the env wrapper for a quick smoke test), a
    uniformly random action is returned so the simulator keeps running.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_action: AgentAction | None = None

    def get_action(self) -> AgentAction:
        if self._next_action is not None:
            action = self._next_action
            self._next_action = None
            return action
        return random.choice(list(AgentAction))
