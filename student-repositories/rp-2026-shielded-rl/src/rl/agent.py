import random

from src.agent import Agent
from src.constants import AgentAction


class RLAgent(Agent):
    """
    Adapter function to map the RL policy's chosen action to our environment.
    The simulator pulls the action for the agent from get_action() at each step,
    but the policy is decided from the outside (Gymnasium loop). The environment writes this
    choice into _next_action before each step and get_action hands it back to the simulator which executes it. 
    Returns random actions if the policy fails for any reason. 
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
