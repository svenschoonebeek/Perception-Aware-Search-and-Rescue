from src.agent import Agent, AgentAction
import numpy as np

class RandAgent(Agent):
    def get_action(self) -> AgentAction:
        return np.random.choice(list(AgentAction))