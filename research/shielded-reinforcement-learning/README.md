# MPC-Shielded Reinforcement Learning for Safe Autonomous Exploration in Search and Rescue

A reinforcement-learning agent learns to explore an unknown disaster environment and find
victims quickly. A lightweight model-predictive shield keeps every move inside a safe set
certified against the agent's current belief. If the policy proposes an unsafe action the
shield hands control to a tube SDD-MPC backup. Otherwise the policy acts freely.

This repository is part of the [Research Project 2026](https://github.com/TU-Delft-CSE/Research-Project)
of TU Delft.

## Agents

All agents run through the same simulator on the same held-out maps with identical sensing
and metrics.

| name      | description                                          |
|-----------|------------------------------------------------------|
| `rl`      | the trained policy acting alone                      |
| `mpc`     | exhaustive-search MPC                                |
| `tmpc`    | tube SDD-MPC                                          |
| `shield`  | RL with the slip-tube safety shield                  |
| `shieldc` | shield plus an MPC takeover when the policy stalls   |

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

Train the recurrent policy.

```bash
SCALE=1 uv run python -m src.rl.run_lstm
```

Evaluate every agent on the same held-out maps.

```bash
uv run python -m src.rl.run_compare_parallel \
    --agents rl,mpc,tmpc,shield,shieldc \
    --rl-model runs/lstm_XXXX/scale_30x30 --lstm --deterministic \
    --size 30 --victims 10 --horizon 500 --episodes 250 --open --start center
```

Run the paired analysis over the results.

```bash
uv run python -m src.rl.analyze_compare --dir runs/compare
```

## Author

Jokūbas Dimša. BSc Computer Science and Engineering at TU Delft. Supervised by
A. Jamshidnejad and S. Schoonebeek. The simulator was built by the project group.
