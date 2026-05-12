import unittest

from src.simulator import Simulator
from src.supervisor import Supervisor
from src.constants import TraversabilityLevel

class SupervisorTest(unittest.TestCase):
    def test_setup(self) -> None:
        self.simulator = Simulator(10, 10)
        self.supervisor = Supervisor(10, 10, 0.05)
        self.simulator.set_supervisor(self.supervisor)
    
    def test_extract_frontiers(self):
        supervisor = Supervisor(width=5, height=5, decay=0.0)
        supervisor.shared_belief.confidence.matrix[1:4, 1:4] = 1.0

        supervisor.shared_belief.traversability[2, 2] = TraversabilityLevel.UNTRAVERSIBLE
        
        frontiers = set(supervisor._extract_frontiers())

        expected_frontiers = {
            (2, 2)
        }

        self.assertEqual(frontiers, expected_frontiers, 
                         f"Failed! Expected {expected_frontiers}, but got {frontiers}")
    
    def test_extract_frontiers_half_and_half(self):
        supervisor = Supervisor(width=6, height=6, decay=0.0)
        supervisor.shared_belief.confidence.matrix[:, 0:3] = 1.0

        supervisor.shared_belief.traversability[2, 2] = TraversabilityLevel.UNTRAVERSIBLE

        frontiers = set(supervisor._extract_frontiers())
        
        # Blob 1: y=0, y=1 - Centroid rounds to y=0
        # Blob 2: y=3, y=4, y=5 - Centroid is y=4
        expected_frontiers = {
            (2, 0),
            (2, 4)
        }
        self.assertEqual(frontiers, expected_frontiers, 
                         f"Failed! Expected {expected_frontiers}, but got {frontiers}")
