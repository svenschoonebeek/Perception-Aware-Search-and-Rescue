import unittest

from src.agents.random import RandAgent
from src.constants import AgentAction

class RandAgentTest(unittest.TestCase):

    def test_get_action(self):
        agent = RandAgent("Randy", 0, 0, 1, 1, 0.1, 0.9, 1, False)
        action = agent.get_action()
        self.assertIn(action, list(AgentAction))