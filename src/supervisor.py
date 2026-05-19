from __future__ import annotations
from collections import deque
from typing import TYPE_CHECKING, cast
import numpy as np
import logging

from .agent import Agent
from .state import State
from .constants import *
from scipy.ndimage import binary_dilation, label
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from src.agent import Agent

logger = logging.getLogger(__name__)

class Supervisor:
    def __init__(self, width: int, height: int, decay: float, impatiance_threshold: int = 10):
        self.width = width
        self.height = height
        self.shared_belief = State(width, height)
        self.shared_belief.traversability.matrix = np.full((height, width), TraversabilityLevel.TRAVERSABLE)
        self.global_explored: np.ndarray = np.zeros((height, width), dtype=bool) # Cells the supervisor has received information about
        
        self.impatience_threshold = impatiance_threshold
        self.decay = decay # Certainty decay per time step [0,1]
        self.event_queue: list[InterventionEvent] = [] # A queue of events reported by agents that the supervisor can respond to

        self.agent_tracker: dict[str, AgentInfo] = {} # tracks necessary information for each agent, used in conflict checking

    def requires_intervention(self) -> bool:
        """
        Checks if there are any events in the queue that require intervention
        """
        return len(self.event_queue) > 0
    
    def trigger_event(self, event: InterventionEvent) -> None:
        """
        Adds an event to the queue to be processed by the supervisor
        """
        self.event_queue.append(event)
        self.event_queue.sort(key=lambda e: e.priority)
    
    def update_belief(self, agent_scans: list[Scan]) -> None:
        """
        Updates the shared belief state based on the agent scans received
        """
        self.shared_belief.confidence.matrix = np.maximum(self.shared_belief.confidence.matrix - self.decay, 0) # vectorized confidence decay

        for scan in agent_scans:
            if scan.agent_name not in self.agent_tracker:
                self.agent_tracker[scan.agent_name] = AgentInfo(scan.agent_x, scan.agent_y, Task(TaskType.IDLE, -1, -1), 0)
            else:
                self.agent_tracker[scan.agent_name].x = scan.agent_x
                self.agent_tracker[scan.agent_name].y = scan.agent_y

            if np.all(self.global_explored[scan.ys, scan.xs]):
                self.agent_tracker[scan.agent_name].impatience += 1
            else:
                self.agent_tracker[scan.agent_name].impatience = 0
            
            self.global_explored[scan.ys, scan.xs] = True

            for x, y, trvs, vuln, vict, agnt, fire, conf in zip(scan.xs, scan.ys, scan.traversability, scan.vulnerability, scan.victims, scan.agents, scan.fire, scan.confidence):
                if (conf > self.shared_belief.confidence.matrix[y][x]):
                    self.shared_belief.traversability[y][x] = trvs
                    self.shared_belief.vulnerability[y][x] = vuln
                    self.shared_belief.victims[y][x] = vict
                    self.shared_belief.agents[y][x] = agnt
                    self.shared_belief.fire[y][x] = fire
                    self.shared_belief.confidence.matrix[y][x] = conf

    def check_for_impatience(self) -> None:
        for agent_name in self.agent_tracker.keys():
            if self.agent_tracker[agent_name].impatience > self.impatience_threshold:
                logger.debug(f"{agent_name} has high global impatience! Triggering intervention.")
                self.trigger_event(InterventionEvent(
                    priority=EventPriority.LOW_PROGRESS,
                    description=f"{agent_name} is suffering from redundant coverage.",
                    data={"agent": agent_name}
            ))
            self.agent_tracker[agent_name].impatience = 0 

    def check_for_victims(self) -> None:
        pass

    def resolve_events(self, agents: list[Agent]) -> None:
        """
        Assigns tasks to agents based on the events in the queue, and the current shared belief state
        """
        unassigned_tasks: list[Task] = []
        bidding_agent_names = set()
        extract_frontiers_flag = False

        while self.event_queue:
            event = self.event_queue.pop(0)
            if event.priority == EventPriority.SPATIAL_CONFLICT or event.priority == EventPriority.LOW_PROGRESS:
                bidding_agent_names.update(event.data["agent"])
                extract_frontiers_flag = True
            
            elif event.priority == EventPriority.TASK_COMPLETED: # update agent tracker to set completed task to idle
                completed_agent_name = event.data["agent"]
                logger.debug(f"Agent {completed_agent_name} completed a task")
                self.agent_tracker[completed_agent_name].task = Task(TaskType.IDLE, -1, -1)
                # doesn't get added to the task list because it is just an update for the tracker, agents are free to go idle themselves.
        
        if extract_frontiers_flag:
            for fx, fy in self._extract_frontiers():
                unassigned_tasks.append(Task(TaskType.EXPLORE, fx, fy))

        bidding_agents = [a for a in agents if a.name in bidding_agent_names]
        self._run_task_auction(unassigned_tasks, bidding_agents)


    def _run_task_auction(self, unassigned_tasks: list[Task], bidding_agents: list[Agent]) -> None:
        if not unassigned_tasks or not bidding_agents:
            return
        logger.debug(f"Auctioning these tasks: {unassigned_tasks}")

        available_agents = list(bidding_agents)
        claimed_tasks: list[Task] = []

        bidding_agent_names = {agent.name for agent in bidding_agents}
        other_agent_info = {
            name: info for name, info in self.agent_tracker.items() 
            if name not in bidding_agent_names
        }

        # weights for burgards spatial exclusivity, being assigned close to an existing target and close to an idle agent, respectively
        w_spatial_claimed, w_close_to_target, w_close_to_agent = 500.0, 300.0, 150.0
        explore_base_reward = 100.0

        while unassigned_tasks and bidding_agents:
            best_task = None
            winning_agent = None
            highest_utility = float('-inf')

            for task in unassigned_tasks:
                # base reward
                if task.task_type == TaskType.EXPLORE:
                    base_reward = explore_base_reward
                
                burgard_penalty = 0.0
                for claimed_t in claimed_tasks:
                    if claimed_t.task_type == TaskType.EXPLORE:
                        distance = np.sqrt((task.x - claimed_t.x)**2 + (task.y - claimed_t.y)**2)
                        burgard_penalty += w_spatial_claimed * np.exp(-distance)
                
                for agent_info in other_agent_info.values():
                    if agent_info.task.task_type == TaskType.EXPLORE:
                        task_location = (agent_info.task.x, agent_info.task.y)
                        distance_to_task = np.sqrt((task.x - task_location[0])**2 + (task.y - task_location[1])**2)
                        burgard_penalty += w_close_to_target * np.exp(-distance_to_task) # penalize frontiers close to other agents' current tasks
                    
                    elif agent_info.task.task_type == TaskType.IDLE:
                        physical_dist = np.sqrt((task.x - agent_info.x)**2 + (task.y - agent_info.y)**2)
                        burgard_penalty += w_close_to_agent * np.exp(-physical_dist)
                
                # evaulate each bidding agent against this task
                for agent in available_agents:
                    travel_cost = np.sqrt((agent.x - task.x)**2 + (agent.y - task.y)**2)
                    total_utility = base_reward - travel_cost - burgard_penalty

                    if total_utility > highest_utility:
                        highest_utility = total_utility
                        best_task = task
                        winning_agent = agent
            if not best_task or not winning_agent:
                break
            
            winning_agent.set_current_task(best_task)
            winning_agent.set_current_path(self._calculate_path(winning_agent.x, winning_agent.y, best_task.x, best_task.y))

            self.agent_tracker[winning_agent.name].task = best_task
            logger.debug(f"Auction assigned {best_task.task_type.name} at ({best_task.x}, {best_task.y}) to {winning_agent.name} (Utility: {highest_utility:.1f})")
            
            claimed_tasks.append(best_task)
            unassigned_tasks.remove(best_task)
            available_agents.remove(winning_agent)

        for agent in available_agents:
            logger.debug(f"No tasks left to assign, setting agent {agent.name} to idle")
            agent.set_current_task(Task(TaskType.IDLE, -1, -1))
            agent.set_current_path([])
    
    def _extract_frontiers(self) -> list:
        explored = np.array(self.global_explored, dtype=bool)
        traversable = self.shared_belief.traversability.matrix == TraversabilityLevel.TRAVERSABLE
        
        unexplored_and_traversable = (~explored) & traversable        
        cross_structure = np.array([[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]])
        
        adjacent_to_unexplored = binary_dilation(unexplored_and_traversable, structure=cross_structure)
        
        frontier_mask = explored & traversable & adjacent_to_unexplored
        
        grouping_structure = np.ones((3, 3), dtype=int)
        labeled_mask, num_regions_raw = cast(
            tuple[np.ndarray, int],
            label(frontier_mask, structure=grouping_structure)
            )
        
        centroids = []
        
        for i in range(1, num_regions_raw + 1):
            coords = np.argwhere(labeled_mask == i)
            
            if len(coords) == 0:
                continue
            
            mean_coord = coords.mean(axis=0)
            
            distances = cdist([mean_coord], coords)
            closest_idx = np.argmin(distances)
            best_coord = coords[closest_idx]
            
            centroids.append((int(best_coord[1]), int(best_coord[0])))  # return (x, y) = (col, row)

            # self._visualize_frontiers(frontier_mask, centroids)

        return centroids

    def _calculate_path(self, start_x: int, start_y: int, target_x: int, target_y: int) -> list[tuple[int, int]]:
        """
        Calculates a path from the agent's current position to the target using BFS.

        Parameters:
        start_x, start_y: The start coordinates of the path
        target_x, target_y: The target coordinates of the path

        Returns: a list of (x, y) coordinates representing the path.
        """
        start = (start_x, start_y)
        target = (target_x, target_y)
        
        if start == target:
            return [start]
            
        queue = deque([[start]])
        seen = set([start])
        
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            
            if (x, y) == target:
                return path
                
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in seen:
                        traversability = self.shared_belief.traversability[ny][nx]
                        victim_presence = self.shared_belief.victims[ny][nx]
                        if  traversability != TraversabilityLevel.UNTRAVERSABLE and victim_presence != VictimPresence.PRESENT:
                            seen.add((nx, ny))
                            queue.append(path + [(nx, ny)])
                            
        return []

    def _visualize_frontiers(self, frontier_mask: np.ndarray, centroids: list[tuple[int, int]]) -> None:        
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        fig, ax = plt.subplots(figsize=(8, 8))
        display = np.zeros((*frontier_mask.shape, 3), dtype=np.uint8)
        display[self.global_explored] = [200, 200, 200]      # explored = light grey
        display[frontier_mask] = [50, 200, 50]               # frontier cells = green
        ax.imshow(display, origin='upper')
        for col, row in centroids:
            ax.plot(col, row, 'r*', markersize=12)
            ax.annotate(f"({col},{row})", (col, row), textcoords="offset points",
                        xytext=(4, 4), color='red', fontsize=8)
        ax.set_title(f"Frontiers ({len(centroids)} clusters)")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.tight_layout()
        plt.show()