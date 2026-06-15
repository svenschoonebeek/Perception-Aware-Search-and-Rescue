import atexit
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from .grid import Grid
from .state import State
from .agent import Agent
from .agents.victim import Victim, Personality, AgeGroup
from .fire_manager import FireManager
from .constants import *
from .metric import Metric
from .supervisor import Supervisor

logger = logging.getLogger(__name__)

class Simulator:
    def __init__(self, width: int, height: int, stopping_condition: StoppingCondition = StoppingCondition.ALL_VICTIMS_FOUND):
        self.width = width
        self.height = height
        self.agents: list[Agent] = []
        self.victims: list[Victim] = []
        self.ground_truth = State(width, height)
        self.fire_manager: FireManager | None = None
        self.metrics = Metric()
        self.supervisor: Supervisor | None = None
        self.tripping = True
        self.stopping_condition: StoppingCondition = stopping_condition
        self.static_victims = False # whether victims are static or move around
        self.rng = np.random.default_rng()

    def set_supervisor(self, supervisor: Supervisor) -> None:
        self.supervisor = supervisor
    
    def set_fire_manager(self, fire_manager: FireManager) -> None:
        self.fire_manager = fire_manager
    
    def add_agent(self, agent: Agent) -> None:
        """
        Adds an agent to the simulation and updates the ground truth

        Parameters:
        agent: The agent to add
        """
        if any(a.name == agent.name for a in self.agents):
            raise ValueError(f"Agent with name {agent.name} already exists in the simulation.")
        
        self.agents.append(agent)
        self.ground_truth.agents[agent.y][agent.x] = AgentPresence.PRESENT
        self.metrics.register_agent(agent)

    def step(self) -> list[State]:
        """
        Performs a single step of the simulation.
        This includes agent actions, environment actions, and updating the ground truth.

        Returns:
        A list with elements:
        list[0]: the ground truth state after the step
        list[1:#agents + 1]: the states of the agents after the step
        list[#agents + 1]: the supervisor state after the step if supervisor is present
        """
        if not self.metrics.history:
            self.metrics.record_snapshot()  # capture initial metric state

        intents = self._collect_intents()
        logger.debug("Intents: %s", {agent.name: (int(intent[0]), int(intent[1])) for agent, intent in intents.items()})
        self._resolve_agent_conflicts(intents)
        self._commit_moves(intents)
        self._step_victims()
        self._rescue_victims_at_exit()
        self._kill_victims_in_fire()

        if self.tripping:
            self._perform_trips()

        self._apply_vulnerability_damage()
        
        # Perform environment actions (firespread, etc.)
        if self.fire_manager is not None:
            self.fire_manager.spread_fire(self.ground_truth)

        # Agents scan and update the shared supervisor belief state
        agent_scans: list[Scan] = []
        for agent in self.agents:
            agent_scans.append(agent.scan(self.ground_truth))
            agent.check_status(self.supervisor, self.ground_truth)
            if (task := agent.get_current_task()) is not None:
                logger.debug(f"Agent {agent.name} has task {task.type} with target {task.x}, {task.y} at position ({agent.x}, {agent.y})")
                logger.debug(f"has path: {agent.get_current_path()}")
        self._refresh_agent_presence()
        
        if self.supervisor is not None:
            self.supervisor.update_belief(agent_scans, self.victims)
            self.supervisor.check_for_impatience()
            self.supervisor.check_explore_tasks()

            if self.supervisor.requires_intervention():
                self.supervisor.resolve_events(self.agents, self.victims) # victims are passed to be passed to rescue agent

        for victim in self.victims:
            victim.scan(self.ground_truth)

        # Perform environment actions (firespread, etc.)
        if self.fire_manager is not None:
            self.fire_manager.spread_fire(self.ground_truth)
        
        self.metrics.steps_taken += 1
        self._update_victim_metrics()
        self._update_area_explored()
        self._update_infeasible_states()
        self._update_inter_agent_distance()
        self._update_spatial_conflict_count()
        self.metrics.record_snapshot()

        res = [self.ground_truth.copy()]
        for agent in self.agents:
            res.append(agent.perception.copy())
        if self.supervisor is not None:
            res.append(self.supervisor.shared_belief.copy())

        return res

    def run(self, steps: int) -> list[list[State]]:
        """
        Performs a number of steps of the simulation.

        Parameters:
        steps: The maximum number of steps to perform

        Returns:
        A list of size (#agents + 2) with supervisor, (#agents + 1) without where
        list[0] are the ground truth states
        list[1:#agents + 1] are the states of the agents
        list[#agents + 1] are the supervisor states if supervisor is present
        First element of every list is the initial state before running the simulation
        """
        record: list[list[State]] = []
        # Record the initial states
        record.append([self.ground_truth.copy()])

        for agent in self.agents:
            agent.scan(self.ground_truth)
            record.append([agent.perception.copy()])
        if self.supervisor is not None:
            record.append([self.supervisor.shared_belief.copy()])        
        for _ in range(steps):
            # Record steps
            logger.debug("Step %d =========================", _)
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
            elif self.ground_truth.traversability[ty][tx] == TraversabilityLevel.UNTRAVERSABLE:
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
        # pass 1: clear old positions
        for agent, (tx, ty) in intents.items():
            if self.ground_truth.fire[ty, tx] == FireLevel.BURNING:
                self.metrics.record_fire_collision(agent)
            if (tx, ty) == (agent.x, agent.y):
                continue
            self.ground_truth.agents[agent.y][agent.x] = 0
        
        # pas 2: update new positions
        for agent, (tx, ty) in intents.items():
            if (tx, ty) == (agent.x, agent.y):
                continue
            self.ground_truth.agents[ty][tx] = AgentPresence.PRESENT
            agent.move_to(tx, ty)

    def _refresh_agent_presence(self) -> None:
        """
        Rewrites each agent's cell in the ground truth with the correct presence value
        based on their current task. Called every step after check_status so that task
        transitions (explore → rescuing) are immediately visible in the agents grid.
        """
        for agent in self.agents:
            task = agent.get_current_task()
            is_rescuing = (
                task is not None
                and task.type == TaskType.RESCUE
                and task.search_status == SearchStatus.RESCUING
            )
            presence = AgentPresence.RESCUE_AGENT if is_rescuing else AgentPresence.PRESENT
            self.ground_truth.agents[agent.y][agent.x] = presence

    def _step_victims(self) -> None:
        """
        Move victims with personality-based conflict resolution.

        Phase 1 – collect intents (physical validity only: bounds, walls, rescue agents).
        Phase 2 – block movers targeting a stayer's cell; for same-target contests pick a
                  winner by priority (SELFISH > EGOIST > ALTRUIST, coinflip on ties) and
                  mutate any EGOIST that shared a contest cell with a SELFISH.
        Phase 3 – commit all moves atomically.
        """
        _PRIORITY = {Personality.SELFISH: 2, Personality.EGOIST: 1, Personality.ALTRUIST: 0}

        # Phase 1: intents (physical validity: bounds, walls, agents, occupied victim cells)
        intents: dict[Victim, tuple[int, int]] = {}
        for victim in self.victims:
            action = victim.get_action()
            if action != AgentAction.WAIT:
                tx, ty = victim._target_cell(action)
                valid = (
                    0 <= tx < self.width and 0 <= ty < self.height
                    and self.ground_truth.traversability[ty][tx] != TraversabilityLevel.UNTRAVERSABLE
                    and self.ground_truth.agents[ty][tx] == AgentPresence.NOT_PRESENT
                    and self.ground_truth.victims[ty][tx] != VictimPresence.PRESENT
                )
                intents[victim] = (tx, ty) if valid else (victim.x, victim.y)
            else:
                intents[victim] = (victim.x, victim.y)

        # Phase 2: conflict resolution
        stayer_cells = {(v.x, v.y) for v in self.victims if intents[v] == (v.x, v.y)}

        # Block movers heading into an occupied stayer cell
        for victim in self.victims:
            if intents[victim] != (victim.x, victim.y) and intents[victim] in stayer_cells:
                intents[victim] = (victim.x, victim.y)

        # Group movers by target and resolve ties
        target_groups: dict[tuple[int, int], list[Victim]] = {}
        for victim in self.victims:
            if intents[victim] != (victim.x, victim.y):
                target_groups.setdefault(intents[victim], []).append(victim)

        for contestants in target_groups.values():
            if len(contestants) <= 1:
                continue

            # Winner determined by original personality
            max_prio = max(_PRIORITY[v.personality] for v in contestants)
            top_tier = [v for v in contestants if _PRIORITY[v.personality] == max_prio]
            winner_index = self.rng.integers(len(top_tier)) if len(top_tier) > 1 else 0
            winner = top_tier[winner_index]

            # Mutation: any EGOIST that contested against a SELFISH becomes SELFISH
            if any(v.personality == Personality.SELFISH for v in contestants):
                for v in contestants:
                    if v.personality == Personality.EGOIST:
                        v.personality = Personality.SELFISH

            for v in contestants:
                if v is not winner:
                    intents[v] = (v.x, v.y)

        # Phase 3: commit atomically — clear all old positions before setting new ones
        # to avoid one victim's clear wiping another's freshly written cell.
        moves = [(v, tx, ty) for v, (tx, ty) in intents.items() if (tx, ty) != (v.x, v.y)]
        for victim, _, _ in moves:
            self.ground_truth.victims[victim.y][victim.x] = VictimPresence.NOT_PRESENT
        for victim, tx, ty in moves:
            self.ground_truth.victims[ty][tx] = VictimPresence.PRESENT
            victim.move_to(tx, ty)


    def _rescue_victims_at_exit(self) -> None:
        """
        Remove any victim that has reached the exit (0,0) and record them as saved.
        The escorting RescueAgent stays in place at the exit; the supervisory layer
        will reassign it once implemented.
        """
        saved = [v for v in self.victims if v.x < 3 and v.y < 3]
        for victim in saved:
            self.ground_truth.victims[victim.y][victim.x] = VictimPresence.NOT_PRESENT
            self.victims.remove(victim)
            self.metrics.victims_saved += 1
            if self.metrics.time_to_first_saved is None:
                self.metrics.time_to_first_saved = self.metrics.steps_taken
           
    def _kill_victims_in_fire(self) -> None:
        """Remove any victim standing on a burning cell and count them as dead."""
        dead = [v for v in self.victims if self.ground_truth.fire[v.y, v.x] >= FireLevel.BURNING]
        for victim in dead:
            self.ground_truth.victims[victim.y][victim.x] = VictimPresence.NOT_PRESENT
            self.victims.remove(victim)
            self.metrics.victims_dead += 1

    def _update_victim_metrics(self) -> None:
        """
        Update the found metrics based on the current ground truth state.
        """
        truth = (self.ground_truth.victims.matrix == VictimPresence.PRESENT)
        total = int(truth.sum())
        self.metrics.total_victims = total
        if total == 0:
            if self.stopping_condition == StoppingCondition.ALL_VICTIMS_RESCUED:
             self.metrics.outcome = RunOutcome.SUCCESS
            return

        union = np.zeros_like(truth, dtype=bool)
        for agent in self.agents:
            union |= (agent.perception.victims.matrix == VictimPresence.PRESENT)
        found = int((truth & union).sum())
        if found > self.metrics.victims_found:
            self.metrics.individual_detection_times.append(self.metrics.steps_taken)
        self.metrics.victims_found = found

        if found > 0 and self.metrics.time_to_first_found is None:
            self.metrics.time_to_first_found = self.metrics.steps_taken
        if found == total and self.metrics.time_to_all_found is None:
            self.metrics.time_to_all_found = self.metrics.steps_taken
            if self.stopping_condition == StoppingCondition.ALL_VICTIMS_FOUND:
                self.metrics.outcome = RunOutcome.SUCCESS

    def _update_area_explored(self) -> None:
        """
        Update each agent's fraction, and the total of traversable cells they have ever scanned.
        """
        traversable = (self.ground_truth.traversability.matrix == TraversabilityLevel.TRAVERSABLE)
        total = int(traversable.sum())
        self.metrics.total_traversable = total
        if total == 0:
            for agent in self.agents:
                self.metrics.area_explored[agent] = 0.0
            return

        global_explored = np.zeros_like(traversable, dtype=bool)
        
        for agent in self.agents:
            global_explored |= (agent.explored)
            explored = int(agent.explored[traversable].sum())
            self.metrics.area_explored[agent] = explored / total
        
        total_explored_count = int(global_explored[traversable].sum())
        self.metrics.total_explored = total_explored_count / total

    def _apply_vulnerability_damage(self) -> None:
        """
        Damage on each agent based on the vulnerability of the cell they currently occupy.
        """
        for agent in self.agents:
            vulnerability = float(self.ground_truth.vulnerability[agent.y][agent.x])
            self.metrics.record_vulnerable_collision(agent, vulnerability)

    def _update_spatial_conflict_count(self) -> None:
        self.metrics.total_spatial_conflicts = 0
        for _, a in enumerate(self.agents):
            self.metrics.total_spatial_conflicts += a.get_spatial_conflicts()

    def _update_inter_agent_distance(self) -> None:
        if len(self.agents) < 2:
            self.metrics.average_inter_agent_distance = 0.0
            return
        
        total_distance = 0.0
        count = 0
        for i, agent_a in enumerate(self.agents):
            for agent_b in self.agents[i+1:]:
                dist = np.sqrt((agent_a.x - agent_b.x) ** 2 + (agent_a.y - agent_b.y) ** 2)
                total_distance += dist
                count += 1
        
        self.metrics.average_inter_agent_distance = total_distance / count if count > 0 else 0.0

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

            trip_prob = max(0.0, tile_vulnerability - tile_confidence) ** 2
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

            is_wall = self.ground_truth.traversability[intended_y][intended_x] == TraversabilityLevel.UNTRAVERSABLE
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
        np.random.seed(seed)
        
        def log_seed():
            logger.info(f"Using seed {seed} to generate the map.")
        log_seed()
        atexit.register(log_seed)
        
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
        self.ground_truth.victims.matrix = _place_victims(self.width, self.height, config.num_victims, rooms, self.ground_truth.agents)
        self.victims = []
        personalities = list(Personality)
        age_groups = list(AgeGroup)
        ys, xs = np.nonzero(self.ground_truth.victims.matrix == VictimPresence.PRESENT)
        for i, (vy, vx) in enumerate(zip(ys, xs)):
            personality = personalities[np.random.randint(len(personalities))]
            age_group = age_groups[np.random.randint(len(age_groups))]
            v = Victim(f"victim_{i}", int(vx), int(vy), self.width, self.height, personality, age_group, is_static=self.static_victims, seed=seed)
            v.scan(self.ground_truth)
            self.victims.append(v)

        self.set_fire_manager(FireManager(self.width, self.height, config.fire_spread_rate, config.fire_duration, seed))
        if self.fire_manager is not None:
            self.fire_manager.initialize_fire(self.ground_truth, config.initial_fire_points, rooms)

        self.ground_truth.confidence.matrix = np.ones((self.height, self.width))
        self.rng = np.random.default_rng(seed)
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
    
    matrix = np.full((y, x), TraversabilityLevel.UNTRAVERSABLE)
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
    matrix[y_start:y_end, x_start:x_end] = TraversabilityLevel.TRAVERSABLE
    #? Should we store the first room in the rooms list?

    # create rooms and store their centers in room_seeds
    for p in range(1, n + 1):
        c = np.random.randint(0,x)
        f = np.random.randint(0,y)

        room_seeds[0][p] = c
        room_seeds[1][p] = f

        random_width = np.random.randint(w_min, w_max)
        half_random_width = random_width // 2

        random_length = np.random.randint(l_min, l_max)
        half_random_length = random_length // 2

        x_start = max(0, c - half_random_width)
        x_end = min(x, c + half_random_width + 1)
        y_start = max(0, f - half_random_length)
        y_end = min(y, f + half_random_length + 1)

        matrix[y_start:y_end, x_start:x_end] = TraversabilityLevel.TRAVERSABLE
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
                matrix[hy_bounds[0]:hy_bounds[1], hx_bounds[0]:hx_bounds[1]] = TraversabilityLevel.TRAVERSABLE
                
                vx_bounds = (max(0, b_x - half_tunnel_width), min(x, b_x + half_tunnel_width + 1))
                vy_bounds = get_bounds(a_y, b_y)
                matrix[vy_bounds[0]:vy_bounds[1], vx_bounds[0]:vx_bounds[1]] = TraversabilityLevel.TRAVERSABLE
                
                tunnels.append({
                    'horizontal_x_range': hx_bounds,
                    'horizontal_y_range': hy_bounds,
                    'vertical_x_bounds': vx_bounds,
                    'vertical_y_bounds': vy_bounds
                })
            else: # vertical first, then horizontal
                vx_bounds = (max(0, a_x - half_tunnel_width), min(x, a_x + half_tunnel_width + 1))
                vy_bounds = get_bounds(a_y, b_y)
                matrix[vy_bounds[0]:vy_bounds[1], vx_bounds[0]:vx_bounds[1]] = TraversabilityLevel.TRAVERSABLE
                
                hx_bounds = get_bounds(a_x, b_x)
                hy_bounds = (max(0, b_y - half_tunnel_width), min(y, b_y + half_tunnel_width + 1))
                matrix[hy_bounds[0]:hy_bounds[1], hx_bounds[0]:hx_bounds[1]] = TraversabilityLevel.TRAVERSABLE

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
    vuln_cmap = plt.get_cmap('autumn_r')
    
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