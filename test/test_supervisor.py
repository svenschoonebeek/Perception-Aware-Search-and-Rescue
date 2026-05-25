import unittest
import numpy as np

from src.simulator import Simulator
from src.supervisor import Supervisor
from src.constants import AgentInfo, Task, TaskType
from src.agent import Agent

class MockAgent(Agent):
    def __init__(self, name: str, x: int, y: int):
        self.name = name
        self.x = x
        self.y = y
        self.current_task = Task(TaskType.IDLE, -1, -1)
        self.current_path = []

    def set_current_task(self, task: Task) -> None:
        self.current_task = task

    def set_current_path(self, path: list) -> None:
        self.current_path = path


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
        expected_frontiers = [(3, 2, 3.0, -1)]

        self.assertEqual(set(frontiers), set(expected_frontiers), 
                        f"Expected frontiers {expected_frontiers}, but got {frontiers}")

    def test_extract_frontiers_large(self) -> None:
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
        
        # 0 = Traversable, 1 = Untraversable
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
        
        # Expected: (3, 2) is the north exit (x=3, y=2), (4, 4) is the east exit (x=4, y=4).
        expected_frontiers = [(3, 2, 1.0, -1), (4, 4, 1.0, -1)]
        
        self.assertEqual(set(frontiers), set(expected_frontiers), 
                        f"Expected {expected_frontiers}, but got {frontiers}")

    def test_extract_frontiers_multiple_corridors_and_walls(self) -> None:
        self.supervisor = Supervisor(8, 8, 0.0)
        
        # 1 = Explored, 0 = Unexplored
        self.supervisor.global_explored = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
        
        # 0 = Traversable, 1 = Untraversable
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
        
        # Expected: (3, 1) is the north exit (x=3, y=1), (5, 4) is the east exit (x=5, y=4). both weight 1
        expected_frontiers = [(3, 1, 1.0, -1), (5, 4, 1.0, -1)]
        
        self.assertEqual(set(frontiers), set(expected_frontiers), 
                        f"Expected {expected_frontiers}, but got {frontiers}")
    
    def setUp(self):
        """Runs before every single test to give us a fresh Supervisor."""
        self.supervisor = Supervisor(width=20, height=20, decay=0.1)
        
        # override path calculator to just return a dummy path for tests
        self.supervisor._calculate_path = lambda start_x, start_y, target_x, target_y: [(start_x, start_y), (target_x, target_y)]

    def test_distance_based_assignment(self):
        """Test that agents are assigned the tasks physically closest to them."""
        agent1 = MockAgent("tasker1", 2, 2)
        agent2 = MockAgent("tasker2", 18, 18)
        
        self.supervisor.agent_tracker["tasker1"] = AgentInfo(x=2, y=2, task=Task(TaskType.IDLE, -1, -1), impatience=0)
        self.supervisor.agent_tracker["tasker2"] = AgentInfo(x=18, y=18, task=Task(TaskType.IDLE, -1, -1), impatience=0)

        # Task 1 near agent 1, Task 2 near agent 2
        tasks = [
            Task(TaskType.EXPLORE, 3, 3), 
            Task(TaskType.EXPLORE, 17, 17)
        ]

        self.supervisor._run_task_auction(tasks, [agent1, agent2])

        self.assertEqual(agent1.current_task.x, 3)
        self.assertEqual(agent1.current_task.y, 3)
        self.assertEqual(agent2.current_task.x, 17)
        self.assertEqual(agent2.current_task.y, 17)

    def test_burgard_spatial_exclusivity(self):
        """Test that Burgard penalties force an agent to pick a further task to avoid overlap."""
        agent1 = MockAgent("tasker1", 5, 5)
        agent2 = MockAgent("tasker2", 6, 5)
        
        self.supervisor.agent_tracker["tasker1"] = AgentInfo(x=5, y=5, task=Task(TaskType.IDLE, -1, -1), impatience=0)
        self.supervisor.agent_tracker["tasker2"] = AgentInfo(x=6, y=5, task=Task(TaskType.IDLE, -1, -1), impatience=0)

        tasks = [
            Task(TaskType.EXPLORE, 5, 6),  # 1 tile away from Agent 1
            Task(TaskType.EXPLORE, 5, 7),  # Clustered right next to the first task
            Task(TaskType.EXPLORE, 15, 15) # Far away
        ]

        self.supervisor._run_task_auction(tasks, [agent1, agent2])

        # agent 1 will take the closest one (5, 6)
        # because (5, 7) is close to it burgard spatial penalty will spike
        # agent 2 should be repelled by the penalty and choose the far away task
        assigned_targets = {(agent1.current_task.x, agent1.current_task.y), 
                            (agent2.current_task.x, agent2.current_task.y)}
        
        self.assertIn((5, 6), assigned_targets)
        self.assertIn((15, 15), assigned_targets)
        self.assertNotIn((5, 7), assigned_targets, "Agent 2 failed to disperse due to Burgard penalty!")

    def test_loser_loop_deadlock_clear(self):
        """Test that when there are more agents than tasks, the losers are safely set to IDLE."""
        agent1 = MockAgent("tasker1", 5, 5)
        agent2 = MockAgent("tasker2", 10, 10)
        
        self.supervisor.agent_tracker["tasker1"] = AgentInfo(x=5, y=5, task=Task(TaskType.IDLE, -1, -1), impatience=0)
        self.supervisor.agent_tracker["tasker2"] = AgentInfo(x=10, y=10, task=Task(TaskType.IDLE, -1, -1), impatience=0)

        tasks = [Task(TaskType.EXPLORE, 4, 4)]

        self.supervisor._run_task_auction(tasks, [agent1, agent2])

        self.assertEqual(agent1.current_task.x, 4)
        
        self.assertEqual(agent2.current_task.task_type, TaskType.IDLE)
        self.assertEqual(agent2.current_task.x, -1)
