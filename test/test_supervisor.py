import unittest
import numpy as np

from src.simulator import Simulator
from src.supervisor import Supervisor
from src.constants import TraversabilityLevel

class SupervisorTest(unittest.TestCase):
    def test_setup(self) -> None:
        self.simulator = Simulator(10, 10)
        self.supervisor = Supervisor(10, 10, 0.05)
        self.simulator.set_supervisor(self.supervisor)
    
    def test_extract_frontiers(self) -> None:
        self.supervisor = Supervisor(5, 5, 0.0)
        
        # 1 = Explored, 0 = Unexplored
        self.supervisor.global_explored = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # 0 = Traversable, 1 = Untraversable
        self.supervisor.shared_belief.traversability.matrix = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0]
        ])
        
        frontiers = self.supervisor._extract_frontiers()
        expected_frontiers = [(2, 3)]

        self.assertEqual(set(frontiers), set(expected_frontiers), 
                        f"Expected frontiers {expected_frontiers}, but got {frontiers}")

    def test_extract_frontiers_multiple_corridors_and_walls_inverted(self) -> None:
        self.supervisor = Supervisor(8, 8, 0.0)
        
        # 1 = Explored, 0 = Unexplored
        self.supervisor.global_explored = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
        
        # Inverted: 0 = Traversable, 1 = Untraversable
        self.supervisor.shared_belief.traversability.matrix = np.array([
            [1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ])
        
        frontiers = self.supervisor._extract_frontiers()
        
        # Expected: (2, 3) is the north exit, (4, 4) is the east exit.
        expected_frontiers = [(2, 3), (4, 4)]
        
        self.assertEqual(set(frontiers), set(expected_frontiers), 
                        f"Expected {expected_frontiers}, but got {frontiers}")
        
    def test_extract_frontiers_two_sides_inverted(self) -> None:
        self.supervisor = Supervisor(6, 6, 0.0)
        
        # Explored square (top-left 5x5)
        self.supervisor.global_explored = np.array([
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        
        self.supervisor.shared_belief.traversability.matrix = np.zeros((6, 6), dtype=int)
        
        frontiers = self.supervisor._extract_frontiers()

        expected_frontiers = [(2, 4), (4, 2)]

        self.assertEqual(set(frontiers), set(expected_frontiers), 
                        f"Expected {expected_frontiers}, but got {frontiers}")