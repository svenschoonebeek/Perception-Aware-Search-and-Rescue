from __future__ import annotations
from typing import TYPE_CHECKING, cast
import numpy as np
import logging
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans
from skimage.morphology import skeletonize
from time import time

from .util import fast_bfs_array
from .agent import Agent
from .agents.victim import Victim
from .state import State
from .constants import *
from .metric import Metric

if TYPE_CHECKING:
    from src.agent import Agent

logger = logging.getLogger(__name__)

class Supervisor:
    def __init__(self, width: int, height: int, decay: float, impatiance_threshold: int = 5, rescue_victims: bool = True):
        self.width = width
        self.height = height
        self.shared_belief = State(width, height)
        self.shared_belief.traversability.matrix = np.full((height, width), TraversabilityLevel.TRAVERSABLE)
        self.global_explored: np.ndarray = np.zeros((height, width), dtype=bool) # Cells the supervisor has received information about
        
        self.impatience_threshold: int = impatiance_threshold
        self.decay: float = decay # Certainty decay per time step [0,1]
        self.unresolved_events: list[InterventionEvent] = [] # A list of events reported by agents that the supervisor can respond to
        self.rescue_victims: bool = rescue_victims # whether the supervisor should assign rescue tasks when victims are found

        self.agent_tracker: dict[str, AgentInfo] = {} # tracks necessary information for each agent, used in conflict checking
        self.task_backlog: set[Task] = set() # used for persistent rescue tasks
        self.victim_tracker: dict[str, VictimInfo] = {}

    def requires_intervention(self) -> bool:
        """
        Checks if there are any events in the queue that require intervention
        """
        immediate_triggers = [
            EventType.VICTIM_FOUND, 
            EventType.TASK_COMPLETED, 
            EventType.INITIALIZING_RESCUE, 
            EventType.VICTIM_RESCUED
        ]
        has_critical_event = any(e.type in immediate_triggers for e in self.unresolved_events)
        if has_critical_event:
            return True
        return len(self.unresolved_events) >= 2
    
    def trigger_event(self, event: InterventionEvent) -> None:
        """
        Adds an event to the queue to be processed by the supervisor
        """
        self.unresolved_events.append(event)
    
    def update_belief(self, agent_scans: list[Scan], victims_list: list[Victim]) -> None:
        """
        Updates the shared belief state based on the agent scans received
        """
        self.shared_belief.confidence.matrix = np.maximum(self.shared_belief.confidence.matrix - self.decay, 0) # vectorized confidence decay
        
        for agent_name, agent_info in self.agent_tracker.items():
            logger.debug(f"Agent {agent_name} is tracked with task {agent_info.task.type} with status {agent_info.task.search_status} with target {agent_info.task.x}, {agent_info.task.y} at position ({agent_info.x}, {agent_info.y})")
        for vid, victim_info in self.victim_tracker.items():
            logger.debug(f"Victim {vid} is tracked with state {victim_info.state} at position ({victim_info.x}, {victim_info.y})")
        
        for scan in agent_scans:
            if scan.agent_name not in self.agent_tracker:
                self.agent_tracker[scan.agent_name] = AgentInfo(scan.agent_x, scan.agent_y, Task(TaskType.IDLE), 0)
            else:
                self.agent_tracker[scan.agent_name].x = scan.agent_x
                self.agent_tracker[scan.agent_name].y = scan.agent_y

            if np.all(self.global_explored[scan.ys, scan.xs]) and self.agent_tracker[scan.agent_name].task.type == TaskType.IDLE:
                self.agent_tracker[scan.agent_name].impatience += 1
            else:
                self.agent_tracker[scan.agent_name].impatience = 0
            
            self.global_explored[scan.ys, scan.xs] = True

            for x, y, trvs, vuln, vict, agnt, fire, conf in zip(scan.xs, scan.ys, scan.traversability, scan.vulnerability, scan.victims, scan.agents, scan.fire, scan.confidence):
                if vict == VictimPresence.PRESENT:
                    found_victim = next((v for v in victims_list if v.x == x and v.y == y), None)
                    if found_victim is not None:
                        vid = found_victim.name
                        if vid not in self.victim_tracker:
                            logger.debug(f"Supervisor detected {vid} at ({x}, {y}) based on scan from {scan.agent_name}. Triggering critical victim event.")
                            self.victim_tracker[vid] = VictimInfo(name=vid, x=x, y=y, state=VictimState.ROAMING)
                            self.trigger_event(InterventionEvent(
                                type=EventType.VICTIM_FOUND,
                                description=f"{vid} found at ({x}, {y}) by {scan.agent_name}",
                                data={"x": x, "y": y, "victim_id": vid}
                            ))
                        self.victim_tracker[vid].x = x
                        self.victim_tracker[vid].y = y

                self.shared_belief.traversability[y][x] = trvs
                self.shared_belief.vulnerability[y][x] = vuln
                self.shared_belief.victims[y][x] = vict
                self.shared_belief.agents[y][x] = agnt
                self.shared_belief.fire[y][x] = fire
                self.shared_belief.confidence.matrix[y][x] = conf

    def check_for_impatience(self) -> None:
        for agent_name in self.agent_tracker.keys():
            impatience = self.agent_tracker[agent_name].impatience
            logger.debug(f"Agent {agent_name} has impatience level {impatience}")
            if self.agent_tracker[agent_name].impatience > self.impatience_threshold:
                logger.debug(f"{agent_name} has high global impatience! Triggering intervention.")
                self.trigger_event(InterventionEvent(
                    type=EventType.LOW_PROGRESS,
                    description=f"{agent_name} is suffering from redundant coverage.",
                    data={"agent": agent_name}
                ))
                self.agent_tracker[agent_name].impatience = 0 

    def check_explore_tasks(self) -> None:
        for target_agent_name, target_agent_info in self.agent_tracker.items():
            if target_agent_info.task.type == TaskType.EXPLORE:
                tx, ty = target_agent_info.task.x, target_agent_info.task.y
                for checking_agent_name, checking_agent_info in self.agent_tracker.items():
                    ax, ay = checking_agent_info.x, checking_agent_info.y
                    distance = np.sqrt((ax - tx)**2 + (ay - ty)**2)
                    if distance <= 1.0:
                        if target_agent_name != checking_agent_name:
                            logger.debug(f"Target at ({tx}, {ty}) for {target_agent_name} was cleared by passing agent {checking_agent_name}.")
                            self.trigger_event(InterventionEvent(
                                type=EventType.TASK_COMPLETED,
                                description=f"Exploration task at ({tx}, {ty}) was cleared by {checking_agent_name}.",
                                data={"agent": target_agent_name}
                            ))
                        break

    def _set_agent_task(self, agent: Agent, task: Task, path: list[tuple[int, int]]) -> None:
        agent.set_current_task(task)
        agent.set_current_path(path)
        self.agent_tracker[agent.name].task = task

    def resolve_events(self, agents: list[Agent], victims: list[Victim], metrics: Metric) -> None:
        """
        Assigns tasks to agents based on the events in the queue, and the current shared belief state
        """
        unassigned_tasks: list[Task] = []
        bidding_agent_names = set()
        extract_frontiers_flag = False
        all_agents_bid_flag = False

        logger.debug(f"Supervisor is resolving events: {[e.description for e in self.unresolved_events]}")
        while self.unresolved_events:
            event = self.unresolved_events.pop(0)
            if event.type == EventType.SPATIAL_CONFLICT or event.type == EventType.LOW_PROGRESS:
                bidding_agent_names.add(event.data["agent"])
                extract_frontiers_flag = True
            
            elif event.type == EventType.VICTIM_FOUND and self.rescue_victims:
                vid = event.data["victim_id"]
                rescue_task = Task(TaskType.RESCUE, x=event.data["x"], y=event.data["y"], victim_id=vid, search_status=SearchStatus.SEARCHING)
                self.task_backlog.add(rescue_task)
                all_agents_bid_flag = True
            
            elif event.type == EventType.PATH_BLOCKED:
                bidding_agent_names.add(event.data["agent"])
                extract_frontiers_flag = True
                # TODO change this for the agent to resolve locally?
            
            elif event.type == EventType.TASK_COMPLETED: # if agent sets itsself to idle, or supervisor detects an explore task is completed
                completed_agent_name = event.data["agent"]
                logger.debug(f"Agent {completed_agent_name} completed a task")
                self._set_agent_task(next(a for a in agents if a.name == completed_agent_name), Task(TaskType.IDLE), [])
                # doesn't get added to the task list because it is just an update for the tracker or agent
            
            elif event.type == EventType.INITIALIZING_RESCUE:
                logger.debug(f"Initializing rescue for victim {event.data['victim_id']} by agent {event.data['agent']}")
                self.agent_tracker[event.data["agent"]].task.search_status = SearchStatus.RESCUING
                self.victim_tracker[event.data["victim_id"]].state = VictimState.BEING_RESCUED
            
            elif event.type == EventType.VICTIM_RESCUED:
                self.victim_tracker[event.data["victim_id"]].state = VictimState.RESCUED
                self.agent_tracker[event.data["agent"]].task = Task(TaskType.IDLE)

        if extract_frontiers_flag:
            frontiers = self._extract_frontiers()
            # frontiers = self._create_voronoi_partition()
            for fx, fy, fweight, froom_id in frontiers:
                unassigned_tasks.append(Task(TaskType.EXPLORE, fx, fy, fweight, froom_id))

        unassigned_tasks.extend(self.task_backlog)
        assignable_agents = [a for a in agents if a.assignable and self.agent_tracker[a.name].task.type != TaskType.RESCUE]
        bidding_agents = [a for a in assignable_agents if a.name in bidding_agent_names]
        if all_agents_bid_flag:
            bidding_agents = assignable_agents
        
        if not unassigned_tasks or not bidding_agents:
            return
        
        self._run_task_auction(unassigned_tasks, bidding_agents, victims, metrics)

    def _run_task_auction(self, unassigned_tasks: list[Task], bidding_agents: list[Agent], victims: list[Victim], metrics: Metric) -> None:
        if not unassigned_tasks or not bidding_agents:
            logger.debug("No tasks or agents available for auction. Skipping assignment.")
            return
        logger.debug(f"Auctioning these tasks: {unassigned_tasks}")

        available_agents = list(bidding_agents)
        claimed_tasks: list[Task] = []

        # weights for burgards spatial exclusivity and close to an idle agent, respectively
        w_close_to_target, w_close_to_agent = 500, 200
        explore_reward_per_tile = 25.0

        # how far the burgard penalty should extend
        spatial_decay_scale = 5.0

        while unassigned_tasks and bidding_agents:
            best_task = None
            winning_agent = None
            highest_utility = float('-inf')

            for task in unassigned_tasks:
                if task.type == TaskType.RESCUE:
                    latest_victim_info = self.victim_tracker.get(task.victim_id)
                    if latest_victim_info:
                        task.x = latest_victim_info.x
                        task.y = latest_victim_info.y
                
                burgard_penalty = 0.0
                for agent_name, agent_info in self.agent_tracker.items():
                    if any(a.name == agent_name for a in available_agents):
                        continue

                    if agent_info.task.type == TaskType.EXPLORE:
                        task_location = (agent_info.task.x, agent_info.task.y)
                        distance_to_task = np.sqrt((task.x - task_location[0])**2 + (task.y - task_location[1])**2)
                        burgard_penalty += w_close_to_target * np.exp(-distance_to_task / spatial_decay_scale) # penalize frontiers close to other agents' current tasks
                    
                    elif agent_info.task.type == TaskType.IDLE:
                        physical_dist = np.sqrt((task.x - agent_info.x)**2 + (task.y - agent_info.y)**2)
                        burgard_penalty += w_close_to_agent * np.exp(-physical_dist / spatial_decay_scale)
                
                # evaulate each bidding agent against this task
                for agent in available_agents:
                    travel_cost = np.sqrt((agent.x - task.x)**2 + (agent.y - task.y)**2)
                    # travel_cost = len(self._calculate_path(agent.x, agent.y, task.x, task.y))
                    capabilities = agent.get_capabilities()
                    if task.type == TaskType.EXPLORE:
                        if capabilities.exploration == 0:
                            continue
                        base_reward = 100 * capabilities.exploration
                    elif task.type == TaskType.RESCUE:
                        if capabilities.rescuing == 0:
                            continue
                        base_reward = 10000 * capabilities.rescuing
                    
                    total_utility = base_reward - travel_cost - burgard_penalty 
                    
                    # logger.debug(f"Agent {agent.name} has utility {total_utility:.1f} for task {task.task_type.name} at ({task.x}, {task.y}) [Base: {base_reward}, Travel Cost: {travel_cost}, Burgard Penalty: {burgard_penalty:.1f}]")
                    
                    if total_utility > highest_utility:
                        highest_utility = total_utility
                        best_task = task
                        winning_agent = agent
            if not best_task or not winning_agent:
                break
            
            best_path = self._calculate_path(winning_agent.x, winning_agent.y, best_task.x, best_task.y)
            if len(best_path) == 0:
                logger.debug(f"No path found for agent {winning_agent.name} to task at ({best_task.x}, {best_task.y}). Skipping assignment.")
                unassigned_tasks.remove(best_task)
                continue

            if best_task.type == TaskType.RESCUE:
                self.victim_tracker[best_task.victim_id] = VictimInfo(name=best_task.victim_id, x=best_task.x, y=best_task.y, state=VictimState.SOUGHT)
                victim = next((v for v in victims if v.name == best_task.victim_id), None)
                if victim is None:
                    logger.debug(f"Could not find victim with id {best_task.victim_id} for rescue assignment.")
                    unassigned_tasks.remove(best_task)
                    continue
                winning_agent.set_assigned_victim(victim)
                #? Currently we just get the victim class from the id, but maybe update rescue to also use an id or something
            
            self._set_agent_task(winning_agent, best_task, best_path)
            metrics.total_task_allocations += 1
            logger.debug(f"Auction assigned {best_task.type.name} at ({best_task.x}, {best_task.y}) to {winning_agent.name} (Utility: {highest_utility:.1f})")
            logger.debug(f"Assigned path {best_path}")

            claimed_tasks.append(best_task)
            unassigned_tasks.remove(best_task)
            available_agents.remove(winning_agent)
            
        for agent in available_agents:
            logger.debug(f"No tasks left to assign, setting agent {agent.name} to idle")
            self._set_agent_task(agent, Task(TaskType.IDLE), [])
        
        self.task_backlog = set(t for t in self.task_backlog if t not in claimed_tasks)

    def _extract_weighted_centroids(self, labeled_mask: np.ndarray, num_regions_raw: int, use_room_ids: bool) -> list[tuple[int, int, float, int]]:
        """
        Extract weighted centroids from a labeled mask with a room id
        """
        weighted_centroids = []
        tiles_per_split = 4 # any cluster with a multiple of this will be split into multiple targets

        for i in range(1, num_regions_raw + 1):
            coords = np.argwhere(labeled_mask == i)
            
            if len(coords) == 0:
                continue
            num_targets = max(1, len(coords) // tiles_per_split)
            raw_eig = len(coords) / num_targets 
            # clip the info gain to the split, such that larger clusters don't get disproportionately high weights
            expected_info_gain = min(float(tiles_per_split), raw_eig)

            if num_targets == 1:
                # Standard mean for small regions
                mean_coord = coords.mean(axis=0)
                distances = cdist([mean_coord], coords)
                best_coord = coords[np.argmin(distances)]
                room_id = i if use_room_ids else -1
                weighted_centroids.append((int(best_coord[1]), int(best_coord[0]), expected_info_gain, room_id))
            else:
                cluster_centers, _ = kmeans(coords.astype(float), num_targets)
                distances = cdist(cluster_centers, coords)
                
                closest_indices = np.argmin(distances, axis=1)
                
                for idx in np.unique(closest_indices):
                    best_coord = coords[idx]
                    room_id = i if use_room_ids else -1
                    weighted_centroids.append((int(best_coord[1]), int(best_coord[0]), expected_info_gain, room_id))
        return weighted_centroids
    
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
        
        weighted_centroids = self._extract_weighted_centroids(labeled_mask, num_regions_raw, use_room_ids=False)
        # self._visualize_frontiers(frontier_mask, weighted_centroids)
        return weighted_centroids
    
    def _create_voronoi_partition(self) -> list[tuple[int, int, float, int]]:
        explored = np.array(self.global_explored, dtype=bool)
        traversable = (self.shared_belief.traversability.matrix == TraversabilityLevel.TRAVERSABLE)
        
        known_free_space = (explored & traversable).astype(int)

        distance_transform = np.asarray(distance_transform_edt(known_free_space))
        skeleton = np.asarray(skeletonize(known_free_space))

        # try and find doorways in the skeleton by looking for points that are close to both sides of a wall
        doorway_threshold = 2.0 
        critical_points = skeleton & (distance_transform <= doorway_threshold)

        thick_barriers = binary_dilation(critical_points, iterations=int(doorway_threshold))

        segmented_space = known_free_space.copy()
        segmented_space[thick_barriers] = 0

        grouping_structure = np.ones((3, 3), dtype=int)
        labeled_rooms, num_rooms = cast(
            tuple[np.ndarray, int],
            label(segmented_space, structure=grouping_structure)
        )

        unexplored_and_traversable = (~explored) & traversable        
        cross_structure = np.array([[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]])
        
        adjacent_to_unexplored = binary_dilation(unexplored_and_traversable, structure=cross_structure)
        frontier_mask = explored & traversable & adjacent_to_unexplored

        frontier_coords = np.argwhere(frontier_mask)
        if len(frontier_coords) == 0:
            return []

        room_coords = np.argwhere(labeled_rooms > 0)
        
        # if not enough space for voronoi partitioning, just return the frontier centroids
        if len(room_coords) == 0:
            labeled_mask, num_regions_raw = cast(
                tuple[np.ndarray, int],
                label(frontier_mask, structure=grouping_structure)
            )
            return self._extract_weighted_centroids(labeled_mask, num_regions_raw, use_room_ids=False)

        distances = cdist(frontier_coords, room_coords)
        closest_room_indices = np.argmin(distances, axis=1)
        
        frontier_room_assignments = labeled_rooms[room_coords[closest_room_indices][:, 0], 
                                                room_coords[closest_room_indices][:, 1]]

        room_labeled_mask = np.zeros_like(explored, dtype=int)
        for idx, (y, x) in enumerate(frontier_coords):
            room_labeled_mask[y, x] = frontier_room_assignments[idx]

        weighted_centroids = self._extract_weighted_centroids(room_labeled_mask, num_rooms, use_room_ids=True)
        self._visualize_frontiers(frontier_mask, weighted_centroids)
        # self._visualize_segmentation(known_free_space, distance_transform, skeleton, critical_points, labeled_rooms)
        return weighted_centroids

    def _calculate_path(self, start_x: int, start_y: int, target_x: int, target_y: int) -> list[tuple[int, int]]:
        """
        Calculates a path from the agent's current position to the target using BFS.

        Parameters:
        start_x, start_y: The start coordinates of the path
        target_x, target_y: The target coordinates of the path

        Returns: a list of (x, y) coordinates representing the path.
        """
        path = fast_bfs_array(start_x, start_y,
                             target_x, target_y,
                             self.shared_belief.traversability.matrix,
                             self.shared_belief.victims.matrix,
                             self.shared_belief.fire.matrix,
                             TraversabilityLevel.UNTRAVERSABLE.value, VictimPresence.PRESENT.value, 
                             FireLevel.BURNING.value, AgentPresence.PRESENT.value)
        return [(int(pt[0]), int(pt[1])) for pt in path]

    def _visualize_frontiers(self, frontier_mask: np.ndarray, centroids: list[tuple[int, int, float, int]]) -> None:        
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        fig, ax = plt.subplots(figsize=(8, 8))
        display = np.zeros((*frontier_mask.shape, 3), dtype=np.uint8)
        display[self.global_explored] = [200, 200, 200]      # explored = light grey
        display[frontier_mask] = [50, 200, 50]               # frontier cells = green
        ax.imshow(display, origin='upper')
        for col, row, weight, room in centroids:
            ax.plot(col, row, 'r*', markersize=12)
            ax.annotate(f"({col},{row})\nw={weight:.1f}\nr={room}", (col, row), textcoords="offset points",
                        xytext=(4, 4), color='red', fontsize=8)
        ax.set_title(f"Frontiers ({len(centroids)} clusters)")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.tight_layout()
        plt.show()
    
    import matplotlib.pyplot as plt

    def _visualize_gvg(self, known_free_space, skeleton, intersections, labeled_corridors):
        """Temporary visualization to debug Tunnel/GVG segmentation."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        
        axes[0].imshow(known_free_space, cmap='gray')
        axes[0].set_title('1. Known Tunnel Space')

        # Combine skeleton and nodes
        skel_vis = np.zeros_like(known_free_space, dtype=float)
        skel_vis[skeleton == 1] = 1.0  # Tunnels in orange
        skel_vis[intersections] = 2.0  # Intersections in white
        axes[1].imshow(skel_vis, cmap='hot')
        axes[1].set_title('2. Skeleton & Intersections (Nodes)')

        # Display the labeled segments
        axes[2].imshow(labeled_corridors, cmap='tab20')
        axes[2].set_title('3. Labeled Tunnel Segments (Edges)')

        plt.tight_layout()
        plt.show()
    
    def _visualize_segmentation(self, known_free_space, distance_transform, skeleton, critical_points, labeled_rooms):
        import matplotlib.pyplot as plt
        """Temporary visualization to debug room segmentation."""
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        
        axes[0, 0].imshow(known_free_space, cmap='gray')
        axes[0, 0].set_title('1. Known Free Space')

        axes[0, 1].imshow(distance_transform, cmap='viridis')
        axes[0, 1].set_title('2. Distance Transform')

        # Combine skeleton and critical points
        skel_vis = np.zeros_like(known_free_space, dtype=float)
        skel_vis[skeleton] = 1.0  # Skeleton in orange
        skel_vis[critical_points] = 2.0  # Critical points in white
        axes[1, 0].imshow(skel_vis, cmap='hot')
        axes[1, 0].set_title('3. Skeleton & Critical Pts (Doorways)')

        axes[1, 1].imshow(labeled_rooms, cmap='tab20')
        axes[1, 1].set_title('4. Labeled Rooms (If all one color, it failed)')

        plt.tight_layout()
        plt.show()