from dataclasses import dataclass
from enum import Enum, IntEnum

class RunOutcome(Enum):
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    TIMEOUT = "timeout"

class AgentAction(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4

class FireLevel(IntEnum):
    SAFE = 0
    FLAMMABLE = 1
    BURNING = 2
    BURNT = 3

class VulnerabilityLevel(float, Enum): #? Maybe create a separate TUNNEL vulnerability level?
    SAFE = 0.0
    VULNERABLE = 0.6
    HIGH_RISK = 1.0

class TraversabilityLevel(IntEnum):
    TRAVERSIBLE = 0
    UNTRAVERSIBLE = 1

class AgentPresence(IntEnum):
    NOT_PRESENT = 0
    PRESENT = 1

class VictimPresence(IntEnum):
    NOT_PRESENT = 0
    PRESENT = 1

@dataclass
class BaseCommand:
    target_x: int
    target_y: int

@dataclass
class RescueCommand(BaseCommand): #TODO current variables are just placeholders, need to be defined properly after we have a better idea of the MPC implementation
    urgency_weight: float = 2.0
    target_tolerance: int = 0

@dataclass
class DiscoverCommand(BaseCommand): #TODO same as above
    coverage_radius: int = 3
    avoid_fire_weight: float = 5.0

class TaskType(Enum):
    RESCUE = "rescue"
    DISCOVER = "discover"

@dataclass
class Task:
    task_id: int
    x: int
    y: int
    task_type: TaskType
    task_reward: int

    def generate_command(self) -> BaseCommand:
        """
        Translates task into command to send to MPC agent.
        """
        if self.task_type == TaskType.RESCUE:
            # dynamically set command parameters based on the task's reward/priority
            urgency = 2.0 if self.task_reward > 50 else 1.0
            return RescueCommand(
                target_x=self.x, 
                target_y=self.y, 
                urgency_weight=urgency
            )
            
        elif self.task_type == TaskType.DISCOVER:
            return DiscoverCommand(
                target_x=self.x, 
                target_y=self.y
            )
            
        else:
            raise ValueError(f"Unsupported TaskType: {self.task_type}")

@dataclass
class AgentTelemetry:
    agent_id: int
    current_x: float
    current_y: float
    current_status: str
    cost_to_go: float   # The MPC's mathematical estimate 
    local_discoveries: list

@dataclass
class MapConfig:
    num_rooms: int = 4
    unconnected_probability: float = 0.0
    room_vulnerability_probability: float = 0.3
    room_vulnerability_severity: float = 0.4
    tunnel_vulnerability_probability: float = 0.3
    tunnel_vulnerability_severity: float = 0.4
    initial_fire_points: int = 1
    fire_spread_rate: float = 0.3
    fire_duration: int = 8  # -1 means infinite duration
    start_room_width: int = 3
    start_room_length: int = 3
    min_room_width: int = 6
    max_room_width: int = 12
    min_room_length: int = 6
    max_room_length: int = 12
    min_tunnel_thickness: int = 1
    max_tunnel_thickness: int = 3
    num_agents: int = 0
    num_victims: int = 2