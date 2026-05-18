from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any
from numpy.typing import NDArray

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
    TRAVERSABLE = 0
    UNTRAVERSABLE = 1

class AgentPresence(IntEnum):
    NOT_PRESENT = 0
    PRESENT = 1

class VictimPresence(IntEnum):
    NOT_PRESENT = 0
    PRESENT = 1

class EventPriority(IntEnum): # TODO to be extended with possible future events
    """Lower number = Higher Priority"""
    CRITICAL_VICTIM_FOUND = 1
    PATH_BLOCKED = 2
    SPATIAL_CONFLICT = 3
    LOW_PROGRESS = 4
    TASK_COMPLETED = 5

@dataclass
class InterventionEvent:
    priority: EventPriority
    description: str
    data: dict[str, Any] = field(default_factory=dict)
    
class TaskType(Enum):
    IDLE = "idle"
    RESCUE = "rescue"
    EXPLORE = "explore"
    
@dataclass
class Task:
    task_type: TaskType
    x: int
    y: int

@dataclass
class AgentInfo:
    x: int 
    y: int
    task: Task
    impatience: int

@dataclass
class Scan:
    # metadata:
    agent_name: str
    agent_x: int 
    agent_y: int

    # data:
    xs: NDArray
    ys: NDArray
    traversability: NDArray
    vulnerability: NDArray
    victims: NDArray
    agents: NDArray
    fire: NDArray
    confidence: NDArray

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