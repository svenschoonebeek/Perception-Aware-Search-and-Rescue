"""Unified held-out comparison: drive every agent through the SAME Simulator on the
SAME held-out maps, collect identical metrics, aggregate to mean +/- 95% CI.

Agents (all on identical seeded maps, same entrance, same sensing, same metrics):
  rl     - the trained RL policy, acting in the sim
  mpc    - exhaustive-search MPC
  tmpc   - tube / SDD-MPC
  shield - shielded RL: take the RL action if its slip-tube is safe,
           otherwise defer to the SDD-MPC backup.

Usage:
  uv run python -m src.rl.run_compare --agents mpc,tmpc --size 20 --victims 6 --horizon 300 --episodes 30
  uv run python -m src.rl.run_compare --agents rl,shield --rl-model runs/lstm_*/L1_g0999_h1000_rs1 --lstm \
      --size 20 --victims 6 --horizon 1000 --episodes 100
"""
import argparse
import contextlib
import math
import os
import time
from pathlib import Path

import numpy as np

from src.simulator import Simulator
from src.constants import RunOutcome, AgentAction, TraversabilityLevel, FireLevel, MapConfig
from src.agent import Agent
from src.agents.mpc import MpcAgent
from src.agents.tmpc import TmpcAgent
from src.rl.env import build_observation, N_CHANNELS, N_STACK
from src.rl.configs import sar_config

SCAN = dict(decay=0.01, scan_accuracy=0.9, scan_radius=3, scan_falloff=True)
START = (0, 0)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HAZARD_VUL = 0.5  # cells with belief-vulnerability >= this are treated as unsafe


# ---------------------------------------------------------------------------
# RL policy acting inside the simulator (builds obs from belief, runs the model)
# ---------------------------------------------------------------------------
class PolicyController:
    """Wraps a trained SB3 model so it can act on an Agent's belief each step."""

    def __init__(self, model, lstm, size, deterministic):
        self.model = model
        self.lstm = lstm
        self.det = deterministic
        self.size = size
        self.lstm_state = None
        self.ep_start = True
        # frame-stack buffer (oldest first, newest last), matching VecFrameStack.
        self.buffer = None if lstm else np.zeros((N_STACK, N_CHANNELS, size, size), dtype=np.float32)

    def act(self, agent) -> int:
        obs = build_observation(agent)  # (N_CHANNELS, H, W)
        if self.lstm:
            x = obs[None]
            action, self.lstm_state = self.model.predict(
                x, state=self.lstm_state, episode_start=np.array([self.ep_start]),
                deterministic=self.det)
            self.ep_start = False
        else:
            self.buffer = np.roll(self.buffer, -1, axis=0)
            self.buffer[-1] = obs
            x = self.buffer.reshape(N_STACK * N_CHANNELS, self.size, self.size)[None]
            action, _ = self.model.predict(x, deterministic=self.det)
        return int(np.asarray(action).reshape(-1)[0])


class PolicyAgent(Agent):
    """Pure-RL agent: always executes the policy's action (clean RL baseline in-sim)."""

    def __init__(self, name, x, y, size, controller):
        super().__init__(name, x, y, size, size, **SCAN)
        self.controller = controller

    def get_action(self) -> AgentAction:
        return AgentAction(self.controller.act(self))


class ShieldedAgent(TmpcAgent):
    """Run the RL action if its slip-tube is safe, else defer to the SDD-MPC backup."""

    def __init__(self, name, x, y, size, controller, stuck_patience=0):
        super().__init__(name, x, y, size, size, **SCAN)
        self.controller = controller
        self.n_decisions = 0
        self.n_override = 0
        # After this many steps with no new exploration, hand off to the MPC (0 = off).
        self.stuck_patience = stuck_patience
        self._best_explored = 0
        self._stale = 0
        self._mpc_latch = False

    def _in_safe_set(self, x, y) -> bool:
        """Safe = in-grid, traversable, no victim, not burning, not a high-risk hazard cell."""
        if not (0 <= x < self.world_width and 0 <= y < self.world_height):
            return False
        if self.perception.traversability[y][x] != TraversabilityLevel.TRAVERSIBLE:
            return False
        if self.perception.victims[y][x] == 1:
            return False
        if float(self.perception.vulnerability[y][x]) >= HAZARD_VUL:
            return False
        return self.perception.fire[y][x] != FireLevel.BURNING

    def _shield_safe(self, action: AgentAction) -> bool:
        """Safe if the target and every traversable neighbour it could slip into are safe."""
        tx, ty = self._target_cell(action)
        if not self._in_safe_set(tx, ty):
            return False
        for dx, dy in N4:
            nx, ny = tx + dx, ty + dy
            if (0 <= nx < self.world_width and 0 <= ny < self.world_height
                    and self.perception.traversability[ny][nx] == TraversabilityLevel.TRAVERSIBLE
                    and not self._in_safe_set(nx, ny)):
                return False
        return True

    def get_action(self) -> AgentAction:
        self.n_decisions += 1
        # Detect a stall (no newly explored cell for stuck_patience steps).
        if self.stuck_patience > 0 and not self._mpc_latch:
            explored = int(self.explored.sum())
            if explored > self._best_explored:
                self._best_explored, self._stale = explored, 0
            else:
                self._stale += 1
                if self._stale >= self.stuck_patience:
                    self._mpc_latch = True  # hand off to the MPC to finish
        if self._mpc_latch:
            self.n_override += 1
            return super().get_action()
        rl_action = AgentAction(self.controller.act(self))
        if self._shield_safe(rl_action):
            return rl_action
        self.n_override += 1
        return super().get_action()  # MPC backup


def make_agent(kind, size, model, lstm, deterministic, completion, start=START):
    x, y = start
    if kind == "mpc":
        return MpcAgent("mpc", x, y, size, size, **SCAN)
    if kind == "tmpc":
        return TmpcAgent("tmpc", x, y, size, size, **SCAN)
    if kind in ("rl", "shield", "shieldc"):
        ctrl = PolicyController(model, lstm, size, deterministic)
        if kind == "rl":
            return PolicyAgent("rl", x, y, size, ctrl)
        patience = completion if kind == "shieldc" else 0  # shield=safety-only, shieldc=+completion
        return ShieldedAgent(kind, x, y, size, ctrl, stuck_patience=patience)
    raise ValueError(f"unknown agent kind: {kind}")


def open_config(size, victims):
    """Big-room, wide-tunnel map: hazards in the rooms but clear tunnels, fire on.
    Low room severity keeps most hazard scannable so the planners don't stall."""
    return MapConfig(
        num_rooms=max(6, size // 3), num_victims=victims, num_agents=0,
        unconnected_probability=0.0,
        min_room_width=3, max_room_width=6, min_room_length=3, max_room_length=6,
        min_tunnel_thickness=3, max_tunnel_thickness=3,
        initial_fire_points=2, fire_spread_rate=0.03, fire_duration=8,
        room_vulnerability_probability=0.35, room_vulnerability_severity=0.25,
        tunnel_vulnerability_probability=0.0, tunnel_vulnerability_severity=0.0,
    )


def run_episode(kind, size, victims, horizon, seed, model, lstm, deterministic, corridor, completion, start, open_map):
    sim = Simulator(size, size)
    cfg = open_config(size, victims) if open_map else sar_config(size, num_victims=victims, corridor=corridor)
    sim.generate_ground_truth(cfg, seed=seed)
    agent = make_agent(kind, size, model, lstm, deterministic, completion, start)
    sim.add_agent(agent)

    # Decision time = just the get_action call (policy or MPC), excluding scan and physics.
    # Wrap get_action so the timer brackets exactly that call, averaged over all steps.
    agent._decide_s = 0.0
    agent._decide_n = 0
    _orig_get = agent.get_action

    def _timed_get(_orig=_orig_get, _a=agent):
        _t = time.perf_counter()
        act = _orig()
        _a._decide_s += time.perf_counter() - _t
        _a._decide_n += 1
        return act

    agent.get_action = _timed_get

    with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn):
        sim.run(horizon)  # silence the MPC agents' debug prints

    curr = sim.metrics.history[-1]
    total = int(curr.total_victims) or 1
    steps = int(getattr(curr, "step", sim.metrics.steps_taken)) or 1
    return dict(
        coverage=float(curr.area_explored.get(agent, 0.0)),
        victims=float(curr.victims_found) / total,
        success=1.0 if curr.outcome == RunOutcome.SUCCESS else 0.0,
        terr_coll=int(curr.terrain_collisions.get(agent, 0)),
        vict_coll=int(curr.victim_collisions.get(agent, 0)),
        damage=float(curr.damage.get(agent, 0.0)),
        infeasible=int(getattr(agent, "infeasible_states", 0)),  # MPC stalls (0 for pure RL)
        steps=steps,
        decision_ms=1000.0 * agent._decide_s / max(1, agent._decide_n),
        override_frac=(agent.n_override / max(1, agent.n_decisions)) if kind in ("shield", "shieldc") else 0.0,
    )


def ci95(xs):
    a = np.asarray(xs, dtype=float)
    m = float(a.mean())
    half = 1.96 * float(a.std(ddof=1)) / math.sqrt(len(a)) if len(a) > 1 else 0.0
    return m, half


METRICS = ["coverage", "victims", "success", "terr_coll", "vict_coll", "damage",
           "infeasible", "steps", "decision_ms", "override_frac"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="mpc,tmpc")
    ap.add_argument("--rl-model", default=None, help="run dir or .zip (for rl/shield)")
    ap.add_argument("--lstm", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--victims", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=777_000)
    ap.add_argument("--corridor", type=int, default=1, help="tunnel width (>=2 lets the SDD-MPC navigate)")
    ap.add_argument("--completion", type=int, default=0,
                    help="shield: hand off to SDD-MPC after K steps with no new exploration (0=safety-only)")
    ap.add_argument("--start", default="0,0", help="start cell 'x,y' or 'center' (corner traps the tube MPC)")
    ap.add_argument("--open", action="store_true", help="big-room hazard-free maps where the SDD-MPC navigates")
    ap.add_argument("--outdir", default="runs/compare", help="output dir (set per-shard for parallel runs)")
    args = ap.parse_args()
    start = (args.size // 2, args.size // 2) if args.start == "center" else tuple(int(v) for v in args.start.split(","))

    kinds = [k.strip() for k in args.agents.split(",")]
    model = None
    if any(k in ("rl", "shield", "shieldc") for k in kinds):
        mp = Path(args.rl_model)
        if mp.suffix != ".zip":
            mp = next((mp / n for n in ("best_model.zip", "rppo_model.zip", "ppo_model.zip")
                       if (mp / n).exists()), mp)
        if args.lstm:
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO.load(mp, device="auto")
        else:
            from stable_baselines3 import PPO
            model = PPO.load(mp, device="auto")
        print(f"[compare] loaded {'LSTM' if args.lstm else 'PPO'} model: {mp}", flush=True)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    # Record the exact settings used, for reproducibility / the report.
    (out / "run_config.txt").write_text(repr(dict(
        agents=args.agents, size=args.size, victims=args.victims, horizon=args.horizon,
        episodes=args.episodes, seed=args.seed, open_map=bool(args.open), corridor=args.corridor,
        completion=args.completion, start=start, rl_model=args.rl_model, lstm=bool(args.lstm),
        deterministic=bool(args.deterministic), scan=SCAN, hazard_vul=HAZARD_VUL)))
    summary = out / "summary.csv"
    header = ",".join(["agent", "N"] + [f"{m}_mean" for m in METRICS] + [f"{m}_ci95" for m in METRICS])
    if not summary.exists():
        summary.write_text(header + "\n")

    for kind in kinds:
        print(f"\n[compare] {kind}: {args.episodes} held-out maps "
              f"({args.size}x{args.size}, {args.victims} victims, horizon {args.horizon})", flush=True)
        rows = []
        for i in range(args.episodes):
            r = run_episode(kind, args.size, args.victims, args.horizon, args.seed + i,
                            model, args.lstm, args.deterministic, args.corridor, args.completion, start, args.open)
            rows.append(r)
            print(f"  ep {i+1}/{args.episodes} seed={args.seed+i}: cov={r['coverage']:.3f} "
                  f"vic={r['victims']:.3f} succ={int(r['success'])} "
                  f"coll={r['terr_coll']+r['vict_coll']} dmg={r['damage']:.1f} inf={r['infeasible']}"
                  + (f" ovr={r['override_frac']:.2f}" if kind in ("shield", "shieldc") else ""), flush=True)

        agg = {m: ci95([r[m] for r in rows]) for m in METRICS}
        per = [",".join(METRICS)] + [",".join(str(round(r[m], 4)) for m in METRICS) for r in rows]
        (out / f"{kind}_episodes.csv").write_text("\n".join(per))
        line = ",".join([kind, str(len(rows))] +
                        [f"{agg[m][0]:.4f}" for m in METRICS] + [f"{agg[m][1]:.4f}" for m in METRICS])
        with summary.open("a") as f:
            f.write(line + "\n")
        print(f"  -> {kind}: " + "  ".join(f"{m}={agg[m][0]:.3f}" for m in METRICS), flush=True)

    print(f"\n[compare] summary -> {summary}", flush=True)


if __name__ == "__main__":
    main()
