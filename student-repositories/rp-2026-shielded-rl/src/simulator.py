import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from .grid import Grid
from .state import State
from .agent import Agent
from .fire_manager import FireManager
from .constants import *
from .metric import Metric

logger = logging.getLogger(__name__)

class Simulator:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.agents: list[Agent] = []
        self.ground_truth = State(width, height)
        self.fire_manager: FireManager = FireManager(width, height, 0.0, 0) #? Start with a non-spreading fire manager, wondering if it is the right way
        self.metrics = Metric()
        self.tripping = True

    def add_agent(self, agent: Agent) -> None:
        """
        Adds an agent to the simulation and updates the ground truth

        Parameters:
        agent: The agent to add
        """
        self.agents.append(agent)
        self.ground_truth.agents[agent.y][agent.x] = 1
        self.metrics.register_agent(agent)

    def step(self) -> list[State]:
        """
        Performs a single step of the simulation.
        This includes agent actions, environment actions, and updating the ground truth.

        Returns:
        A list with elements:
        list[0]: the ground truth state after the step
        list[1:]: the states of the agents after the step
        """
        if not self.metrics.history:
            self.metrics.record_snapshot()  # capture initial metric state

        intents = self._collect_intents()
        self._resolve_agent_conflicts(intents)
        self._commit_moves(intents)

        if self.tripping:
            self._perform_trips()

        self._apply_vulnerability_damage()
        
        # Perform environment actions (firespread, etc.)
        self.fire_manager.spread_fire(self.ground_truth)

        for agent in self.agents:
            agent.scan(self.ground_truth)
            
        self.metrics.steps_taken += 1
        self._update_victim_metrics()
        self._update_area_explored()
        self._update_infeasible_states()
        self.metrics.record_snapshot()

        res = [self.ground_truth.copy()]
        for agent in self.agents:
            res.append(agent.perception.copy())

        return res

    def run(self, steps: int) -> list[list[State]]:
        """
        Performs a number of steps of the simulation.

        Parameters:
        steps: The maximum number of steps to perform

        Returns:
        A list of size (#agents + 1) where
        list[0] are the ground truth states
        list[1:] are the states of the agents
        First element of every list is the initial state before running the simulation
        """
        record: list[list[State]] = []
        # Record the initial states
        record.append([self.ground_truth.copy()])

        for agent in self.agents:
            agent.scan(self.ground_truth)
            record.append([agent.perception.copy()])

        for _ in range(steps):
            # Record steps
            logger.info("Step %d =========================", _)
            step_result = self.step()
            for i in range(len(step_result)):
                record[i].append(step_result[i])

            if self.metrics.outcome != RunOutcome.IN_PROGRESS:
                break
        if self.metrics.outcome == RunOutcome.IN_PROGRESS:
            self.metrics.outcome = RunOutcome.TIMEOUT
            if self.metrics.history:
                # make final snapshot reflect the timeout outcome
                self.metrics.history[-1] = self.metrics.snapshot()
        return record

    def _collect_intents(self) -> dict[Agent, tuple[int, int]]:
        """
        Each agent picks an action. WAIT and collisions are recorded immediately and the
        agent's intent becomes its current position. Otherwise the intent is
        the target cell.
        """
        intents: dict[Agent, tuple[int, int]] = {}
        for agent in self.agents:
            action = agent.get_action()
            if action == AgentAction.WAIT:
                self.metrics.record_wait(agent)
                intents[agent] = (agent.x, agent.y)
                continue

            tx, ty = agent._target_cell(action)
            if not (0 <= tx < self.width and 0 <= ty < self.height):
                self.metrics.record_terrain_collision(agent)
                intents[agent] = (agent.x, agent.y)
            elif self.ground_truth.traversability[ty][tx] == TraversabilityLevel.UNTRAVERSIBLE:
                self.metrics.record_terrain_collision(agent)
                intents[agent] = (agent.x, agent.y)
            elif self.ground_truth.victims[ty][tx] == VictimPresence.PRESENT:
                self.metrics.record_victim_collision(agent)
                intents[agent] = (agent.x, agent.y)
            else:
                intents[agent] = (tx, ty)
        return intents

    def _resolve_agent_conflicts(self, intents: dict[Agent, tuple[int, int]]) -> None:
        """
        Iteratively force conflicted movers to stay until no conflicts remain.
        Three conflict types: same-target (multiple movers want one cell),
        mover-into-stayer (a mover targets a cell whose occupant is staying),
        and swap pairs.
        """
        while True:
            changed = False

            # all agents contesting one cell collide, non get to move.
            target_groups: dict[tuple[int, int], list[Agent]] = {}
            for agent in self.agents:
                if intents[agent] == (agent.x, agent.y):
                    continue
                target_groups.setdefault(intents[agent], []).append(agent)
            for movers in target_groups.values():
                if len(movers) > 1:
                    for a in movers:
                        self.metrics.record_inter_agent_collision(a)
                        intents[a] = (a.x, a.y)
                    changed = True
            if changed:
                continue

            # target cell is occupied by an agent that isn't moving away. Both agents collide.
            stayers: dict[tuple[int, int], Agent] = {
                (a.x, a.y): a for a in self.agents
                if intents[a] == (a.x, a.y)
            }
            for mover in [a for a in self.agents if intents[a] != (a.x, a.y)]:
                if intents[mover] in stayers:
                    stayer = stayers[intents[mover]]
                    self.metrics.record_inter_agent_collision(mover)
                    self.metrics.record_inter_agent_collision(stayer)
                    intents[mover] = (mover.x, mover.y)
                    changed = True
            if changed:
                continue

            # Two movers want each other's current cells.
            movers = [a for a in self.agents if intents[a] != (a.x, a.y)]
            for i, a in enumerate(movers):
                for b in movers[i+1:]:
                    if intents[a] == (b.x, b.y) and intents[b] == (a.x, a.y):
                        self.metrics.record_inter_agent_collision(a)
                        self.metrics.record_inter_agent_collision(b)
                        intents[a] = (a.x, a.y)
                        intents[b] = (b.x, b.y)
                        changed = True

            if not changed:
                break

    def _commit_moves(self, intents: dict[Agent, tuple[int, int]]) -> None:
        """
        Commit the agent moves to the ground truth and update the agent positions.
        
        Parameters:
        intents: The agent intents
        """
        for agent, (tx, ty) in intents.items():
            if (tx, ty) == (agent.x, agent.y):
                continue
            self.ground_truth.agents[agent.y][agent.x] = 0
            self.ground_truth.agents[ty][tx] = 1
            agent.move_to(tx, ty)

    def _update_victim_metrics(self) -> None:
        """
        Update the found metrics based on the current ground truth state.
        """
        truth = (self.ground_truth.victims.matrix == VictimPresence.PRESENT)
        total = int(truth.sum())
        self.metrics.total_victims = total
        if total == 0:
            return

        union = np.zeros_like(truth, dtype=bool)
        for agent in self.agents:
            union |= (agent.perception.victims.matrix == VictimPresence.PRESENT)
        found = int((truth & union).sum())
        self.metrics.victims_found = found

        if found > 0 and self.metrics.time_to_first_found is None:
            self.metrics.time_to_first_found = self.metrics.steps_taken
        if found == total and self.metrics.time_to_all_found is None:
            self.metrics.time_to_all_found = self.metrics.steps_taken
            self.metrics.outcome = RunOutcome.SUCCESS

    def _update_area_explored(self) -> None:
        """
        Update each agent's fraction of traversable cells they have ever scanned.
        """
        traversable = (self.ground_truth.traversability.matrix == TraversabilityLevel.TRAVERSIBLE)
        total = int(traversable.sum())
        self.metrics.total_traversable = total
        if total == 0:
            for agent in self.agents:
                self.metrics.area_explored[agent] = 0.0
            return

        for agent in self.agents:
            explored = int(agent.explored[traversable].sum())
            self.metrics.area_explored[agent] = explored / total

    def _apply_vulnerability_damage(self) -> None:
        """
        Damage on each agent based on the vulnerability of the cell they currently occupy.
        """
        for agent in self.agents:
            vulnerability = float(self.ground_truth.vulnerability[agent.y][agent.x])
            self.metrics.record_vulnerable_collision(agent, vulnerability)

    def _update_infeasible_states(self) -> None:
        for agent in self.agents:
            self.metrics.infeasible_states[agent] = agent.infeasible_states

    def _perform_trips(self) -> None:
        """
        For agents on vulnerable terrain, will make them trip depending on vulnerability level
        """
        for agent in self.agents:
            tile_vulnerability = float(self.ground_truth.vulnerability[agent.y][agent.x])
            tile_confidence = agent.perception.confidence[agent.y][agent.x]

            confidence_mult = 0.7
            trip_prob = max(0.0, tile_vulnerability - confidence_mult * tile_confidence)
            if np.random.random() > trip_prob:
                continue # Don't trip

            # Trip
            direction = np.random.randint(0, 4) # [0:UP, 1:DOWN, 2:LEFT, 3:RIGHT]
            dy = [-1, 1, 0, 0]
            dx = [0, 0, -1, 1]
            intended_x, intended_y = agent.x + dx[direction], agent.y + dy[direction]
            
            # Check for collisions
            is_out_of_bounds = not(0 <= intended_x < self.width and 0 <= intended_y < self.height)
            if is_out_of_bounds:
                self.metrics.record_terrain_collision(agent)
                continue

            is_wall = self.ground_truth.traversability[intended_y][intended_x] == TraversabilityLevel.UNTRAVERSIBLE
            if is_wall:
                self.metrics.record_terrain_collision(agent)
                continue

            if self.ground_truth.agents[intended_y][intended_x] == 1:
                self.metrics.record_inter_agent_collision(agent)
                continue
            
            if self.ground_truth.victims[intended_y][intended_x] == VictimPresence.PRESENT:
                self.metrics.record_victim_collision(agent)
                continue

            # Move the agent
            self.ground_truth.agents[agent.y][agent.x] = 0
            self.ground_truth.agents[intended_y][intended_x] = 1
            agent.move_to(intended_x, intended_y)
            

    def generate_ground_truth(self, config: MapConfig | None = None, seed: int | None = None) -> None:
        """
        Generates the ground truth state of the world, including the traversability, vulnerability, 
        agent positions, and victim positions.
        Parameters:
        config: The configuration for the map generation, including parameters like number of rooms, room sizes, vulnerability probabilities, etc. If None, default parameters will be used.
        seed: The seed for the random number generator to ensure reproducibility. If None, a random seed will be generated and printed. 

        """
        if config is None:
            config = MapConfig()
        if seed is None:
            seed = (int) (np.random.random() * 1_000_000_000)
        logger.info(f"Using seed {seed} to generate the map.")
        np.random.seed(seed)
        # generates and sets the 2D grid with agent and victims, currently with preset values.
        self.ground_truth.traversability.matrix, rooms, tunnels = _generate_traversability_matrix(self.width, self.height, 
                                                                                        config.num_rooms, config.unconnected_probability, 
                                                                                        config.start_room_width, config.start_room_length,
                                                                                        config.min_room_width, config.max_room_width, 
                                                                                        config.min_room_length, config.max_room_length, 
                                                                                        config.min_tunnel_thickness, config.max_tunnel_thickness)
        self.ground_truth.vulnerability.matrix = _generate_vulnerability_matrix(self.width, self.height, rooms, tunnels,
                                                                                config.room_vulnerability_probability, config.tunnel_vulnerability_probability,
                                                                                config.room_vulnerability_severity, config.tunnel_vulnerability_severity)
        self.ground_truth.agents.matrix = _place_agents(self.width, self.height, config.num_agents, rooms, self.ground_truth.victims)
        self.ground_truth.victims.matrix = _place_victims(self.width, self.height, config.num_victims, rooms, self.ground_truth.agents)

        self.fire_manager = FireManager(self.width, self.height, config.fire_spread_rate, config.fire_duration)
        self.fire_manager.initialize_fire(self.ground_truth, config.initial_fire_points, rooms)

        self.ground_truth.confidence.matrix = np.ones((self.height, self.width))
        return

def _generate_traversability_matrix(
        x: int, y: int,
        n: int, u_p: float,
        s_width: int, s_length: int,
        w_min: int, w_max: int,
        l_min: int, l_max: int,
        t_min: int, t_max: int
        ) -> tuple[np.ndarray, list[dict], list[dict]]:
    """
    Generates 2D traversability matrix with rooms and connecting tunnels, and returns the room and tunnel bounds and the matrix.
    
    Parameters:
    x, y: Dimensions of the grid
    n: Number of rooms (outside of first room in top left)
    s_width, s_length: Dimensions of the first room in the top left
    u_p: (unconnected_probability) The probability a room is unconnected
    w_min, w_max: Min/Max width of the rooms
    l_min, l_max: Min/Max length of the rooms
    t_min, t_max: Min/Max thickness of the tunnels
    """

    if not (0 <= u_p <= 1):
        raise ValueError(f"probability of unconnected rooms should be between 0 and 1 but was: {u_p}")
    
    matrix = np.full((y, x), TraversabilityLevel.UNTRAVERSIBLE)
    room_seeds = np.zeros((2, n + 1), dtype=int)
    rooms = []
    tunnels = []

    tunnel_width = np.random.randint(t_min, t_max + 1)
    half_tunnel_width = tunnel_width // 2

    # create first room in top left corner
    c = s_width // 2
    f = s_length // 2
    room_seeds[0][0] = c
    room_seeds[1][0] = f
    x_start = 0
    x_end = s_width
    y_start = 0
    y_end = s_length
    matrix[y_start:y_end, x_start:x_end] = TraversabilityLevel.TRAVERSIBLE
    #? Should we store the first room in the rooms list?

    # create rooms and store their centers in room_seeds
    # Reject placements that touch an existing room (1-cell wall margin) 
    # so rooms stay DISTINCT rather than merging into blobs
    # keep each room fully inside the grid so it is not clipped into an odd shape. 
    # Fall back to the last candidate if no clear spot is found after a few tries, 
    # so we always end up with n rooms
    for p in range(1, n + 1):
        c = f = x_start = x_end = y_start = y_end = 0
        for _attempt in range(40):
            random_width = np.random.randint(w_min, w_max + 1)
            random_length = np.random.randint(l_min, l_max + 1)
            half_w, half_l = random_width // 2, random_length // 2
            c = np.random.randint(half_w, max(half_w + 1, x - half_w))
            f = np.random.randint(half_l, max(half_l + 1, y - half_l))
            x_start, x_end = max(0, c - half_w), min(x, c + half_w + 1)
            y_start, y_end = max(0, f - half_l), min(y, f + half_l + 1)
            my0, my1 = max(0, y_start - 1), min(y, y_end + 1)
            mx0, mx1 = max(0, x_start - 1), min(x, x_end + 1)
            if not np.any(matrix[my0:my1, mx0:mx1] == TraversabilityLevel.TRAVERSIBLE):
                break

        room_seeds[0][p] = c
        room_seeds[1][p] = f

        matrix[y_start:y_end, x_start:x_end] = TraversabilityLevel.TRAVERSIBLE
        rooms.append({
            'x_range': (x_start, x_end),
            'y_range': (y_start, y_end),
            'center': (c, f)
        })
    
    # connect rooms with tunnels, skipping some based on unconnected_probability
    for q in range(n):
        connect = np.random.random() >= u_p

        if (connect):
            a_x, a_y = room_seeds[0][q], room_seeds[1][q]
            b_x, b_y = room_seeds[0][q + 1], room_seeds[1][q + 1]

            def get_bounds(start, end) -> tuple[int, int]:
                return (min(start, end), max(start, end) + 1)
            
            direction = np.random.randint(0, 2)
            if (direction == 1): # horizontal first, then vertical # TODO probably should also return the tunnels same as with rooms
                hx_bounds = get_bounds(a_x, b_x)
                hy_bounds = (max(0, a_y - half_tunnel_width), min(y, a_y + half_tunnel_width + 1))
                matrix[hy_bounds[0]:hy_bounds[1], hx_bounds[0]:hx_bounds[1]] = TraversabilityLevel.TRAVERSIBLE
                
                vx_bounds = (max(0, b_x - half_tunnel_width), min(x, b_x + half_tunnel_width + 1))
                vy_bounds = get_bounds(a_y, b_y)
                matrix[vy_bounds[0]:vy_bounds[1], vx_bounds[0]:vx_bounds[1]] = TraversabilityLevel.TRAVERSIBLE
                
                tunnels.append({
                    'horizontal_x_range': hx_bounds,
                    'horizontal_y_range': hy_bounds,
                    'vertical_x_bounds': vx_bounds,
                    'vertical_y_bounds': vy_bounds
                })
            else: # vertical first, then horizontal
                vx_bounds = (max(0, a_x - half_tunnel_width), min(x, a_x + half_tunnel_width + 1))
                vy_bounds = get_bounds(a_y, b_y)
                matrix[vy_bounds[0]:vy_bounds[1], vx_bounds[0]:vx_bounds[1]] = TraversabilityLevel.TRAVERSIBLE
                
                hx_bounds = get_bounds(a_x, b_x)
                hy_bounds = (max(0, b_y - half_tunnel_width), min(y, b_y + half_tunnel_width + 1))
                matrix[hy_bounds[0]:hy_bounds[1], hx_bounds[0]:hx_bounds[1]] = TraversabilityLevel.TRAVERSIBLE

                tunnels.append({
                    'horizontal_x_range': hx_bounds,
                    'horizontal_y_range': hy_bounds,
                    'vertical_x_bounds': vx_bounds,
                    'vertical_y_bounds': vy_bounds
                })

    return matrix, rooms, tunnels

def _generate_vulnerability_matrix(
        x: int, y: int, 
        rooms: list[dict],
        tunnels: list[dict],
        r_v_p: float, t_v_p: float,
        r_v_s: float, t_v_s: float
        ) -> np.ndarray:
    """
    Generates 2D vulnerability matrix for rooms and tunnels based on the provided probabilities.
    
    Parameters:
    x, y: Dimensions of the grid
    rooms, tunnels: the rooms and tunnels generated by _generate_traversability_matrix
    r_v_p, t_v_p: room and tunnel vulnerability probability, respectively. Dictates how probable it is a room will be either high risk or vulnerable.
    r_v_s, t_v_s: room and tunnel vulnerability severity, respectively. A higher value results in a higher chance of high risk.
    """
    if not (0 <= r_v_p <= 1 and 0 <= t_v_p <= 1 and 0 <= r_v_s <= 1 and 0 <= t_v_s <= 1):
        raise ValueError("One of the vulnerability probabilities is not between 0 and 1.")

    vulnerability = np.full((y, x), VulnerabilityLevel.SAFE)
    for tunnel in tunnels:
        vulnerable = np.random.random() <= t_v_p
        if not (vulnerable):
            continue
        vulnerability_level = VulnerabilityLevel.VULNERABLE
        if (np.random.random() <= t_v_s):
            vulnerability_level = VulnerabilityLevel.HIGH_RISK
        
        hx = tunnel['horizontal_x_range']
        hy = tunnel['horizontal_y_range']
        vulnerability[hy[0]:hy[1], hx[0]:hx[1]] = vulnerability_level
        
        vx = tunnel['vertical_x_bounds']
        vy = tunnel['vertical_y_bounds']
        vulnerability[vy[0]:vy[1], vx[0]:vx[1]] = vulnerability_level

    for room in rooms:
        vulnerable = np.random.random() <= r_v_p
        if not (vulnerable):
            continue
        vulnerability_level = VulnerabilityLevel.VULNERABLE
        if (np.random.random() <= r_v_s):
            vulnerability_level = VulnerabilityLevel.HIGH_RISK
        vulnerability[room['y_range'][0]:room['y_range'][1], room['x_range'][0]:room['x_range'][1]] = vulnerability_level

    return vulnerability

def _place_agents(
        x: int, y: int, 
        n: int, 
        rooms: list[dict], 
        victims: Grid
        ) -> np.ndarray:
    """
    Places agents in rooms.

    Parameters:
    x, y: Dimensions of the grid
    n: Number of agents
    rooms: A list of rooms with center and bounds
    victims: A matrix indicating the presence of victims (to avoid placing agents on top of victims)
    """

    agents = np.full((y, x), AgentPresence.NOT_PRESENT)

    room_n = len(rooms)

    for _ in range(n):
        random_room = rooms[np.random.randint(0, room_n)]
        x_range = random_room['x_range']
        y_range = random_room['y_range']
        random_x = np.random.randint(x_range[0], x_range[1])
        random_y = np.random.randint(y_range[0], y_range[1])
        while (agents[random_y, random_x] == AgentPresence.PRESENT or victims[random_y, random_x] == VictimPresence.PRESENT):
            random_x = np.random.randint(x_range[0], x_range[1])
            random_y = np.random.randint(y_range[0], y_range[1])
        agents[random_y, random_x] = AgentPresence.PRESENT
    
    return agents

def _place_victims(
        x: int, y: int,
        k: int, 
        rooms: list[dict], 
        agents: Grid) -> np.ndarray:
    """
    Places victims in rooms.

    Parameters:
    x, y: Dimensions of the grid
    k: Number of victims
    rooms: A list of rooms with center and bounds
    agents: A matrix indicating the presence of agents (to avoid placing victims on top of agents)
    """

    victims = np.full((y, x), VictimPresence.NOT_PRESENT)
    room_n = len(rooms)

    for _ in range(k):
        random_room = rooms[np.random.randint(0, room_n)]
        x_range = random_room['x_range']
        y_range = random_room['y_range']
        random_x = np.random.randint(x_range[0], x_range[1])
        random_y = np.random.randint(y_range[0], y_range[1])
        while (victims[random_y, random_x] == VictimPresence.PRESENT or agents[random_y, random_x] == AgentPresence.PRESENT):
            random_x = np.random.randint(x_range[0], x_range[1])
            random_y = np.random.randint(y_range[0], y_range[1])
        victims[random_y, random_x] = VictimPresence.PRESENT

    return victims


def visualize_grid_gen(traversability: Grid, agents: Grid, victims: Grid, vulnerability: Grid, fire: Grid) -> None:
    """
    Visualizes the map, vulnerability, fire, agents, and victims in a single plot.
    """
    plt.figure(figsize=(7, 7))
    
    plt.imshow(traversability.matrix, cmap='binary', interpolation='nearest')

    victim_mask = np.where(victims.matrix == VictimPresence.PRESENT, 1, np.nan)
    plt.imshow(victim_mask, cmap='autumn', interpolation='nearest', alpha=1.0)
    
    agent_mask = np.where(agents.matrix == AgentPresence.PRESENT, 1, np.nan)
    plt.imshow(agent_mask, cmap='winter', interpolation='nearest', alpha=1.0)

    vulnerability_mask = np.where(vulnerability.matrix > VulnerabilityLevel.SAFE.value, vulnerability.matrix, np.nan)
    plt.imshow(vulnerability_mask, cmap='autumn_r', interpolation='nearest', alpha=0.3, vmin=0.5, vmax=1)
    vuln_cmap = plt.cm.autumn_r
    
    flammable_mask = np.where(fire.matrix == FireLevel.FLAMMABLE.value, 1, np.nan)
    burning_mask = np.where(fire.matrix == FireLevel.BURNING.value, 1, np.nan)
    burnt_mask = np.where(fire.matrix == FireLevel.BURNT.value, 1, np.nan)

    plt.imshow(flammable_mask, cmap=mcolors.ListedColormap(['gold']), interpolation='nearest', alpha=0.3)
    plt.imshow(burning_mask, cmap=mcolors.ListedColormap(['darkorange']), interpolation='nearest', alpha=0.85)
    plt.imshow(burnt_mask, cmap=mcolors.ListedColormap(['dimgray']), interpolation='nearest', alpha=0.9)

    legend_elements = [
        mpatches.Patch(facecolor='black', edgecolor='black', label='Untraversable (Wall)'),
        mpatches.Patch(facecolor='white', edgecolor='gray', label='Traversable (Floor)'),
        mpatches.Patch(facecolor='blue', label='Agent'),
        mpatches.Patch(facecolor='red', label='Victim'),
        mpatches.Patch(facecolor=vuln_cmap(0.6), alpha=0.6, edgecolor='gray', label='Vulnerable'),
        mpatches.Patch(facecolor=vuln_cmap(1.0), alpha=0.6, edgecolor='gray', label='High Risk'),
        mpatches.Patch(facecolor='gold', alpha=0.5, edgecolor='gray', label='Flammable'),
        mpatches.Patch(facecolor='darkorange', alpha=0.85, edgecolor='darkred', label='Burning'),
        mpatches.Patch(facecolor='dimgray', alpha=0.9, edgecolor='black', label='Burnt')
    ]

    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    rows, cols = traversability.matrix.shape
    plt.xticks(np.arange(-0.5, cols, 1), [])
    plt.yticks(np.arange(-0.5, rows, 1), [])
    plt.grid(color='gray', linewidth=0.1)   
    plt.tick_params(bottom=False, left=False)

    plt.title("Search & Rescue Map")
    plt.tight_layout()
    plt.show()