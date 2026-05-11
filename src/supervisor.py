import numpy as np

from .agent import Agent
from .state import State
from .constants import *

class Supervisor:
    def __init__(self, width: int, height: int, decay: float):
        self.shared_belief = State(width, height)
        self.active_tasks = []

        self.decay = decay # Certainty decay per time step [0,1]
        self.event_queue: list[EventPriority] = [] # A queue of events reported by agents that the supervisor can respond to

        self.agent_scan_radius = 3 # how far agents can scan, used for conflict checking
        self.agent_tracker = dict[str, tuple[int, int]]()

    def requires_intervention(self) -> bool:
        """
        Checks if there are any events in the queue that require intervention
        """
        return len(self.event_queue) > 0
    
    def trigger_event(self, event: EventPriority) -> None:
        """
        Adds an event to the queue to be processed by the supervisor
        """
        self.event_queue.append(event)
        self.event_queue.sort(key=lambda e: e.value)
    
    def update_belief(self, agent_scans: list[Scan]) -> None:
        """
        Updates the shared belief state based on the agent scans received
        """
        self.shared_belief.confidence.matrix = np.maximum(self.shared_belief.confidence.matrix - self.decay, 0) # vectorized confidence decay

        for scan in agent_scans:
            self.agent_tracker[scan.agent_name] = (scan.agent_x, scan.agent_y)
            for x, y, trvs, vuln, vict, agnt, fire, conf in zip(scan.xs, scan.ys, scan.traversability, scan.vulnerability, scan.victims, scan.agents, scan.fire, scan.confidence):
                if (conf > self.shared_belief.confidence.matrix[y][x]):
                    self.shared_belief.traversability[y][x] = trvs
                    self.shared_belief.vulnerability[y][x] = vuln
                    self.shared_belief.victims[y][x] = vict
                    self.shared_belief.agents[y][x] = agnt
                    self.shared_belief.fire[y][x] = fire
                    self.shared_belief.confidence.matrix[y][x] = conf
        
    def check_for_spatial_conflicts(self) -> None:
        """
        Checks for conflicts between active tasks and locations of agents
        """
        if len(self.agent_tracker) < 2:
            return
        
        names = list(self.agent_tracker.keys())
        positions = list(self.agent_tracker.values())
        
        conflict_distance_threshold = self.agent_scan_radius * 1.5 

        # Check pairwise distances between agents 
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if distance < conflict_distance_threshold:
                    print(f"Spatial Conflict Detected between {names[i]} and {names[j]}! Distance: {distance:.1f}")
                    self.trigger_event(InterventionEvent(
                        priority=EventPriority.SPATIAL_CONFLICT,
                        description=f"Spatial Conflict Detected between {names[i]} and {names[j]}! Distance: {distance:.1f}",
                        data={"agents": (names[i], names[j])}
                    ))
                    return

    def check_for_victims(self) -> None:
        pass

    def assign_tasks(self, agents: list[Agent]) -> None:
        """
        Assigns tasks to agents based on the shared belief and active tasks
        """
        pass