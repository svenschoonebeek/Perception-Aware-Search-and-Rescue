from src.agent import Agent, AgentAction
import random

class RandAgent(Agent):
    def get_action(self) -> AgentAction:
        return random.choice(list(AgentAction))