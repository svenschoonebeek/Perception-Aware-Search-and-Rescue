import unittest
import numpy as np

from src.util import fast_bfs_array
from src.constants import TraversabilityLevel, VictimPresence, FireLevel, AgentPresence
class UtilTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Warms up the Numba compiler before the tests
        """
        dummy = np.zeros((5, 5), dtype=np.int8)
        fast_bfs_array(0, 0, 1, 1, dummy, dummy, dummy, TraversabilityLevel.UNTRAVERSABLE, VictimPresence.PRESENT, FireLevel.BURNING, AgentPresence.PRESENT)

    def test_open_path_validity(self):
        """Tests that a clear path correctly starts and ends at the right coordinates."""
        traversability = np.zeros((50, 50), dtype=np.int8)
        victims = np.zeros((50, 50), dtype=np.int8)
        fire = np.zeros((50, 50), dtype=np.int8)
        
        start_x, start_y = 0, 0
        target_x, target_y = 45, 45
        
        path = fast_bfs_array(
            start_x, start_y, target_x, target_y, 
            traversability, victims, fire, TraversabilityLevel.UNTRAVERSABLE, VictimPresence.PRESENT, FireLevel.BURNING, AgentPresence.PRESENT
        )
        
        self.assertGreater(len(path), 0, "Failed to find an open path.")
        
        self.assertTrue(np.array_equal(path[0], [start_x, start_y]), "Path did not start at origin.")
        self.assertTrue(np.array_equal(path[-1], [target_x, target_y]), "Path did not end at target.")

    def test_obstacle_avoidance_validity(self):
        """Tests that the algorithm correctly navigates around a giant wall without crossing it."""
        traversability = np.zeros((50, 50), dtype=np.int8)
        victims = np.zeros((50, 50), dtype=np.int8)
        fire = np.zeros((50, 50), dtype=np.int8)
        
        # wall down the middle
        for y in range(0, 48):
            traversability[y, 25] = 1 
            
        start_x, start_y = 0, 0
        target_x, target_y = 45, 0
        
        path = fast_bfs_array(
            start_x, start_y, target_x, target_y, 
            traversability, victims, fire,
            TraversabilityLevel.UNTRAVERSABLE, VictimPresence.PRESENT, FireLevel.BURNING, AgentPresence.PRESENT
        )
        
        self.assertGreater(len(path), 0, "Failed to find a path around the obstacle.")
        self.assertTrue(np.array_equal(path[0], [start_x, start_y]), "Path did not start at origin.")
        self.assertTrue(np.array_equal(path[-1], [target_x, target_y]), "Path did not end at target.")
        
        for step in path:
            x, y = step[0], step[1]
            self.assertEqual(traversability[y, x], 0, f"Path illegally crossed wall at ({x}, {y})")

    def test_impossible_path_validity(self):
        """Tests that the algorithm correctly returns an empty array when boxed in."""
        traversability = np.zeros((50, 50), dtype=np.int8)
        victims = np.zeros((50, 50), dtype=np.int8)
        fire = np.zeros((50, 50), dtype=np.int8)
        
        # box in target
        traversability[44, 45] = 1 # Top
        traversability[46, 45] = 1 # Bottom
        traversability[45, 44] = 1 # Left
        traversability[45, 46] = 1 # Right
        
        start_x, start_y = 0, 0
        target_x, target_y = 45, 45
        
        path = fast_bfs_array(
            start_x, start_y, target_x, target_y, 
            traversability, victims, fire,
            TraversabilityLevel.UNTRAVERSABLE, VictimPresence.PRESENT, FireLevel.BURNING, AgentPresence.PRESENT
        )
        
        self.assertEqual(len(path), 0, "incorrectly found a path when target was boxed in")
