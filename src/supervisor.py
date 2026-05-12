import numpy as np

from .agent import Agent
from .state import State
from .constants import *
from scipy.ndimage import label, center_of_mass

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
        self.event_queue.sort(key=lambda e: e.priority)
    
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
                    # TODO change this such that it doesn't return instantly, we want auction between all conflicting agents not just first pair
                    return

    def check_for_victims(self) -> None:
        pass

    def resolve_events(self, agents: list[Agent]) -> None:
        """
        Assigns tasks to agents based on the events in the queue, and the current shared belief state
        """
        while self.event_queue:
            event = self.event_queue.pop(0)
            if event.priority == EventPriority.SPATIAL_CONFLICT:
                # TODO assign new exploration tasks to conflicting agents to resolve conflict
                conflicting_agent_names = event.data["agents"]
                conflicting_agents = [a for a in agents if a.name in conflicting_agent_names]
                self._run_frontier_auction(conflicting_agents)

        
    def _extract_frontiers(self) -> list[tuple[int, int]]:
        """
        Extracts frontier cells from the shared belief state for task assignment

        Returns:
        A list of (x, y) tuples representing frontier cell coordinates
        """
        
        conf_matrix = self.shared_belief.confidence.matrix
        trvs_matrix = self.shared_belief.traversability.matrix

        unknown_mask = conf_matrix == 0
        candidate_mask = (conf_matrix > 0) & (trvs_matrix == TraversabilityLevel.TRAVERSIBLE)

        up_unknown = np.roll(unknown_mask, shift=-1, axis=0)
        up_unknown[-1, :] = False

        down_unknown = np.roll(unknown_mask, shift=1, axis=0)
        down_unknown[0, :] = False

        left_unknown = np.roll(unknown_mask, shift=-1, axis=1)
        left_unknown[:, -1] = False

        right_unknown = np.roll(unknown_mask, shift=1, axis=1)
        right_unknown[:, 0] = False

        borders_unknown = up_unknown | down_unknown | left_unknown | right_unknown
        frontier_mask = candidate_mask & borders_unknown

        # clustering for frontier reduction
        labeled_mask, num_regions = label(frontier_mask)
        region_centroids = []
        for i in range(1, num_regions + 1):
            cy, cx = center_of_mass(frontier_mask, labeled_mask, i)
            region_centroids.append((int(cx), int(cy)))

        return region_centroids
        
    def _run_frontier_auction(self, conflicting_agents: list[Agent]) -> None:
        """
        Runs a frontier auction to assign new exploration tasks to the conflicting agents
        """
        frontiers = self._extract_frontiers()
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

            bound_task = Task(TaskType.EXPLORE, best_frontier[0], best_frontier[1])
            winning_agent.current_task = bound_task

            print(f"Auction assigned frontier {best_frontier} to agent {winning_agent.name}")
            unassigned_frontiers.remove(best_frontier)
            available_agents.remove(winning_agent)
        


