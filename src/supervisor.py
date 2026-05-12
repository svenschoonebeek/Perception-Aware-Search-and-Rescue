from typing import cast
import numpy as np
import logging

from .agent import Agent
from .state import State
from .constants import *
from scipy.ndimage import binary_dilation, label, center_of_mass
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

class Supervisor:
    def __init__(self, width: int, height: int, decay: float):
        self.shared_belief = State(width, height)
        self.shared_belief.traversability.matrix = np.full((height, width), TraversabilityLevel.TRAVERSABLE)
        self.global_explored: np.ndarray = np.zeros((height, width), dtype=bool) # Cells the supervisor has received information about
        
        self.decay = decay # Certainty decay per time step [0,1]
        self.event_queue: list[InterventionEvent] = [] # A queue of events reported by agents that the supervisor can respond to

        self.agent_scan_radius = 3 # how far agents can scan, used for conflict checking
        self.agent_tracker = dict[str, tuple[int, int]]()

        self.spatial_conflict_cooldown = 3

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
            self.agent_tracker[scan.agent_name] = (scan.agent_x, scan.agent_y)
            self.global_explored[scan.xs, scan.ys] = True

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
        if (self.spatial_conflict_cooldown > 0):
            self.spatial_conflict_cooldown -= 1
            return
        
        if len(self.agent_tracker) < 2:
            return
        
        names = list(self.agent_tracker.keys())
        positions = list(self.agent_tracker.values())
        
        conflict_distance_threshold = self.agent_scan_radius * 1.5 

        conflicting_agent_names = set()

        # Check pairwise distances between agents 
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if distance < conflict_distance_threshold:
                    logger.debug(f"Spatial Conflict Detected between {names[i]} and {names[j]}! Distance: {distance:.1f}")
                    conflicting_agent_names.add(names[i])
                    conflicting_agent_names.add(names[j])

        if conflicting_agent_names:
            logger.debug(f"Triggering conflict resolution for: {conflicting_agent_names}")
            
            self.trigger_event(InterventionEvent(
                priority=EventPriority.SPATIAL_CONFLICT,
                description=f"Spatial Conflict involving {len(conflicting_agent_names)} agents.",
                data={"agents": list(conflicting_agent_names)} 
            ))
        
        self.spatial_conflict_cooldown = 3

    def check_for_victims(self) -> None:
        pass

    def resolve_events(self, agents: list[Agent]) -> None:
        """
        Assigns tasks to agents based on the events in the queue, and the current shared belief state
        """
        while self.event_queue:
            event = self.event_queue.pop(0)
            if event.priority == EventPriority.SPATIAL_CONFLICT:
                conflicting_agent_names = event.data["agents"]
                conflicting_agents = [a for a in agents if a.name in conflicting_agent_names]
                self._run_burgard_frontier_auction(conflicting_agents)


    def _extract_frontiers(self) -> list:
        explored = np.array(self.global_explored, dtype=bool)
        traversable = self.shared_belief.traversability.matrix == TraversabilityLevel.TRAVERSABLE
        
        unexplored_and_traversable = (~explored) & traversable        
        cross_structure = np.array([[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]])
        
        adjacent_to_unexplored = binary_dilation(unexplored_and_traversable, structure=cross_structure)
        
        frontier_mask = explored & traversable & adjacent_to_unexplored
        
        # grouping_structure = np.ones((3, 3), dtype=int)
        labeled_mask, num_regions_raw = cast(
            tuple[np.ndarray, int],
            label(frontier_mask, structure=cross_structure)
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
            
            centroids.append(tuple(best_coord))
            
        return centroids
        
    def _run_burgard_frontier_auction(self, conflicting_agents: list[Agent]) -> None:
        """
        Runs a Burgard-style frontier auction to assign new exploration tasks to the conflicting agents
        """
        frontiers = self._extract_frontiers()
        logger.debug(f"Extracted frontiers: {frontiers}")
        if not frontiers:
            return
        
        unassigned_frontiers = list(frontiers)
        available_agents = list(conflicting_agents)

        claimed_targets = []

        while available_agents and unassigned_frontiers:
            best_frontier = None
            winning_agent = None
            lowest_total_cost = float('inf')

            for f in unassigned_frontiers:
                burgard_penalty = 0.0
                for claimed_f in claimed_targets:
                    distance = np.sqrt((f[0] - claimed_f[0])**2 + (f[1] - claimed_f[1])**2)
                    burgard_penalty += 500.0 * np.exp(-distance) # spatial exclusivity penalty

                for agent in available_agents:
                    travel_cost = np.sqrt((agent.x - f[0])**2 + (agent.y - f[1])**2)
                    total_cost = travel_cost + burgard_penalty

                    if total_cost < lowest_total_cost:
                        lowest_total_cost = total_cost
                        best_frontier = f
                        winning_agent = agent
                
            if not best_frontier or not winning_agent:
                break
            
            bound_task = Task(TaskType.EXPLORE, best_frontier[0], best_frontier[1])
            winning_agent.set_current_task(bound_task)
            winning_agent.set_current_path([])  # force path recalculation toward new frontier

            logger.debug(f"Auction assigned frontier {best_frontier} to agent {winning_agent.name}")
            claimed_targets.append(best_frontier)
            unassigned_frontiers.remove(best_frontier)
            available_agents.remove(winning_agent)

    def _run_LCB_frontier_auction(self, conflicting_agents: list[Agent]) -> None:
        """
        Runs a frontier auction using Least Contested Bid to assign new exploration tasks to the conflicting agents
        """
        frontiers = self._extract_frontiers()
        logger.debug(f"Extracted frontiers: {frontiers}")
        if not frontiers:
            return
        
        unassigned_frontiers = list(frontiers)
        available_agents = list(conflicting_agents)

        while unassigned_frontiers and available_agents:
            bids = {f: [] for f in unassigned_frontiers}

            for f in unassigned_frontiers:
                for agent in available_agents:
                    travel_cost = np.sqrt((agent.x - f[0])**2 + (agent.y - f[1])**2)
                    # TODO add burgard spatial penalty
                    bids[f].append((travel_cost, agent))

            best_frontier = None
            winning_agent = None
            max_contest_diff =  -float('inf')
            lowest_tiebreaker_bid = float('inf')

            # evaluate LCB
            for f, agent_bids in bids.items():
                agent_bids.sort(key=lambda x: x[0])
                lowest_bid, best_agent_f = agent_bids[0]

                #spatial exclusivity:
                if len(agent_bids) > 1:
                    second_lowest_bid = agent_bids[1][0]
                    contest_diff = second_lowest_bid - lowest_bid
                else:
                    contest_diff = float('inf')
                
                if (contest_diff > max_contest_diff):
                    max_contest_diff = contest_diff
                    lowest_tiebreaker_bid = lowest_bid
                    best_frontier = f
                    winning_agent = best_agent_f

                # tiebreaker
                elif (contest_diff == max_contest_diff) and (lowest_bid < lowest_tiebreaker_bid):
                    lowest_tiebreaker_bid = lowest_bid
                    best_frontier = f
                    winning_agent = best_agent_f

            if not winning_agent or not best_frontier:
                break
            bound_task = Task(TaskType.EXPLORE, best_frontier[0], best_frontier[1])
            winning_agent.current_task = bound_task
            winning_agent.current_path = []  # force path recalculation toward new frontier

            logger.debug(f"Auction assigned frontier {best_frontier} to agent {winning_agent.name}")
            unassigned_frontiers.remove(best_frontier)
            available_agents.remove(winning_agent)
        


