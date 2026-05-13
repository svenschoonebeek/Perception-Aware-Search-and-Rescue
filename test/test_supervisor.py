import unittest
import numpy as np

from src.simulator import Simulator
from src.supervisor import Supervisor
from src.constants import AgentInfo, Task, TaskType, TraversabilityLevel
from src.agents.task_mpc import TaskMpcAgent

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
        expected_frontiers = [(3, 2)]

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
        expected_frontiers = [(3, 2), (4, 4)]
        
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
        
        # Expected: (3, 1) is the north exit (x=3, y=1), (5, 4) is the east exit (x=5, y=4).
        expected_frontiers = [(3, 1), (5, 4)]
        
        self.assertEqual(set(frontiers), set(expected_frontiers), 
                        f"Expected {expected_frontiers}, but got {frontiers}")
        

    def test_burgard_auction(self) -> None:
        self.supervisor = Supervisor(8, 8, 0.0)
        agent1 = TaskMpcAgent("tasker1", 0, 0, 5, 5, 0.01, 0.9, 3, False)
        agent2 = TaskMpcAgent("tasker2", 4, 4, 5, 5, 0.01, 0.9, 3, False)
        
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
        
        # Expected: (3, 1) is the north exit (x=3, y=1), (5, 4) is the east exit (x=5, y=4).
        expected_frontiers = [(3, 1), (5, 4)]

        self.supervisor.agent_tracker[agent1.name] = AgentInfo(x=agent1.x, y=agent1.y, task=Task(TaskType.IDLE, -1, -1))
        self.supervisor.agent_tracker[agent2.name] = AgentInfo(x=agent2.x, y=agent2.y, task=Task(TaskType.IDLE, -1, -1))
        
        # Run the auction
        self.supervisor._run_burgard_frontier_auction([agent1, agent2])
        
        # Check that both agents have been assigned a frontier
        assigned_tasks = {agent1.current_task.task_type: (agent1.current_task.x, agent1.current_task.y), 
                          agent2.current_task.task_type: (agent2.current_task.x, agent2.current_task.y)}
        
        expected_tasks = {TaskType.EXPLORE: expected_frontiers}
        
        for task_type in expected_tasks:
            self.assertIn(task_type, assigned_tasks)
            self.assertIn(assigned_tasks[task_type], expected_tasks[task_type], 
                          f"Expected one of {expected_tasks[task_type]} for task type {task_type}, but got {assigned_tasks[task_type]}")
        