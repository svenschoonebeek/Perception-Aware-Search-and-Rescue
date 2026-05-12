import unittest

import numpy as np

from src.agents.mpc import MpcAgent
from src.constants import *
from src.state import State

class MpcAgentTest(unittest.TestCase):

    # TODO: Fire spread, predict_next_state, _objective, getaction

    def test_is_feasible_wall(self):
        agent = MpcAgent("mpc", 0, 0, 2, 2, 0.1, 0.9, 1, False)
        ground_truth = State(2, 2)
        ground_truth.traversability[0][1] = 1
        agent.scan(ground_truth)
        self.assertFalse(agent._is_feasible(AgentAction.MOVE_RIGHT))

    def test_is_feasible_oob(self):
        agent = MpcAgent("mpc", 0, 0, 1, 1, 0.1, 0.9, 1, False)
        self.assertFalse(agent._is_feasible(AgentAction.MOVE_RIGHT))
        self.assertFalse(agent._is_feasible(AgentAction.MOVE_UP))
        self.assertFalse(agent._is_feasible(AgentAction.MOVE_LEFT))
        self.assertFalse(agent._is_feasible(AgentAction.MOVE_DOWN))

    def test_is_feasible_valid(self):
        agent = MpcAgent("mpc", 0, 0, 2, 2, 0.1, 0.9, 1, False)
        ground_truth = State(2, 2)
        ground_truth.traversability[0][1] = 1
        agent.scan(ground_truth)
        self.assertTrue(agent._is_feasible(AgentAction.MOVE_DOWN))
        self.assertTrue(agent._is_feasible(AgentAction.WAIT))

    def test_target_cell_move(self):
        agent = MpcAgent("mpc", 1, 1, 3, 3, 0.1, 0.9, 1, False)
        self.assertEqual(agent._target_cell(AgentAction.MOVE_UP), (1, 0))
        self.assertEqual(agent._target_cell(AgentAction.MOVE_DOWN), (1, 2))
        self.assertEqual(agent._target_cell(AgentAction.MOVE_LEFT), (0, 1))
        self.assertEqual(agent._target_cell(AgentAction.MOVE_RIGHT), (2, 1))

    def test_target_cell_wait(self):
        agent = MpcAgent("mpc", 1, 1, 3, 3, 0.1, 0.9, 1, False)
        self.assertEqual(agent._target_cell(AgentAction.WAIT), (1, 1))

    def test_confidence_score_zero(self):
        scan_accuracy = 0.9
        agent = MpcAgent("mpc", 1, 1, 3, 3, 0.1, scan_accuracy, 1, False)
        self.assertEqual(agent._confidence_score(), 0.0)
        
    def test_confidence_score_normalized(self):
        scan_accuracy = 0.9
        agent = MpcAgent("mpc", 1, 1, 3, 3, 0.1, scan_accuracy, 1, False)
        ground_truth = State(3, 3)
        ground_truth.traversability[1][2] = 1
        agent.scan(ground_truth)
        self.assertLessEqual(agent._confidence_score(), 1)
        self.assertGreaterEqual(agent._confidence_score(), 0)

        agent.scan_radius = 10
        agent.scan(ground_truth)
        self.assertLessEqual(agent._confidence_score(), 1)
        self.assertGreaterEqual(agent._confidence_score(), 0)

    def test_safety_penalty_zero(self):
        agent = MpcAgent("mpc", 1, 1, 3, 3, 0.1, 0.9, 1, False)
        ground_truth = State(3, 3)
        agent.scan(ground_truth)
        self.assertEqual(agent._safety_penalty(), 0)

    def test_safety_penalty_normalized(self):
        agent = MpcAgent("mpc", 1, 1, 3, 3, 0.1, 0.9, 1, False)
        ground_truth = State(3, 3)
        ground_truth.vulnerability[1][1] = VulnerabilityLevel.HIGH_RISK
        agent.scan(ground_truth)
        self.assertLessEqual(agent._confidence_score(), 1)
        self.assertGreaterEqual(agent._confidence_score(), 0)

        ground_truth.vulnerability[1][1] = VulnerabilityLevel.VULNERABLE
        agent.scan(ground_truth)
        self.assertLessEqual(agent._confidence_score(), 1)
        self.assertGreaterEqual(agent._confidence_score(), 0)

        ground_truth.vulnerability[1][1] = VulnerabilityLevel.SAFE
        ground_truth.fire[1][1] = FireLevel.BURNING
        agent.scan(ground_truth)
        self.assertLessEqual(agent._confidence_score(), 1)
        self.assertGreaterEqual(agent._confidence_score(), 0)

        ground_truth.fire[1][1] = FireLevel.FLAMMABLE
        agent.scan(ground_truth)
        self.assertLessEqual(agent._confidence_score(), 1)
        self.assertGreaterEqual(agent._confidence_score(), 0)

        ground_truth.vulnerability[1][1] = VulnerabilityLevel.HIGH_RISK
        ground_truth.fire[1][1] = FireLevel.BURNING
        agent.scan(ground_truth)
        self.assertLessEqual(agent._confidence_score(), 1)
        self.assertGreaterEqual(agent._confidence_score(), 0)

    def test_exploration_score_normalized(self):
        agent = MpcAgent("mpc", 1, 0, 3, 3, 0.1, 0.9, 0, False)
        agent.explored = np.array([
            [True, True, True], 
            [True, True, True], 
            [False, False, False]
            ])

        self.assertLessEqual(agent._confidence_score(), 1)
        self.assertGreaterEqual(agent._confidence_score(), 0)

        agent.explored = np.array([
            [True, True, True], 
            [True, True, True], 
            [True, True, True]
            ])
        self.assertLessEqual(agent._confidence_score(), 1)
        self.assertGreaterEqual(agent._confidence_score(), 0)