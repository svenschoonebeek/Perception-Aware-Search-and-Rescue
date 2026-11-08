"""Shared SaR map configs for the training runner against all agents"""
from src.constants import MapConfig


def sar_config(size: int, num_victims: int = 10, corridor: int = 1) -> MapConfig:
    rooms = {20: 7, 25: 10, 30: 14}.get(size, max(6, size // 3))
    return MapConfig(
        num_rooms=rooms,
        num_victims=num_victims,
        num_agents=0,                  # the RL env will place the agent at the reset
        unconnected_probability=0.0,   # everything reachable to make the map solvable
        min_room_width=3, max_room_width=5,
        min_room_length=3, max_room_length=5,
        min_tunnel_thickness=corridor, max_tunnel_thickness=corridor,
        initial_fire_points=1, fire_spread_rate=0.03, fire_duration=8,
        room_vulnerability_probability=0.12, room_vulnerability_severity=0.3,
        tunnel_vulnerability_probability=0.08, tunnel_vulnerability_severity=0.3,
    )
