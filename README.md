
---

# Coordination-Aware Task Allocation in Multi-Agent Search and Rescue

This repository contains the simulation environment and control architecture for the bachelor's thesis: **"Coordination-Aware Task Allocation Via a Lightweight Supervisory Layer: An Event-Driven Approach to Multi-Agent Search and Rescue"**.

> **Academic Context:** This project was developed by T.R. Schutgens as part of the [TU Delft CSE3000 Research Project (2026)](https://cse3000-research-project.github.io/) for the EEMCS Faculty. An electronic version of the associated thesis will be made available at the [TU Delft Education Repository](http://repository.tudelft.nl/).

## Overview

In multi-robot Search and Rescue (SaR) missions, autonomous agents must operate in highly volatile environments with strict safety constraints. This project implements a hybrid task-allocation architecture that bridges the gap between centralized planning and decentralized execution.

The system relies on local Model Predictive Control (MPC) for safe, autonomous navigation and obstacle avoidance, while a centralized, event-driven supervisor manages high-level task allocation via a Sequential Single-Item (SSI) auction. The supervisor remains dormant to save computational overhead and only intervenes upon discrete triggers:

* **Spatial Conflicts**

* **Agent Stagnation (Impatience)**

* **Victim Discovery**


## Features

* **Procedural Map Generation:** A parameterized room-and-tunnel generator simulating structured indoor environments with dynamic fire spread and structural vulnerabilities.


* **Heterogeneous Fleet Support:** Support for multiple types of agents, to be added in the `agents` directory.


* **Game-Theoretic Victim Behavior:** Victims are modeled using a GT-BDI architecture, simulating realistic panic, wandering, and trust-building proximity mechanics.


* **Performance Metrics:** Built-in extraction of key mission metrics.



## Getting Started

### Prerequisites

* Python 3.12+


* Standard scientific libraries (`numpy`, `pandas`, `scipy`)

### Running a Simulation

To run a single simulation with the visualizer enabled, execute the main script:

```bash
python -m src.main --debug
```

To run a headless batch simulation for statistical data gathering (comparing Supervised, Unsupervised, and Shared Map architectures):

```bash
python -m src.runner
```

## Data & Evaluation

The `runner.py` script automatically exports mission results to a CSV file in the `sim_data/` directory and prints a detailed console evaluation covering exploration milestones, computation times, and safety metrics (e.g., collision avoidance).

## Team & Collaborators

This project was developed in collaboration with the following team members and academic supervisors from TU Delft:

* **Tigo Schutgens**
* **Jacob Dimša**
* **Nicolas Laplagne**
* **Filip Keerberg**
* **Maria Grouev**
* **Sven Schoonebeek** – *Supervisor*
* **Prof. Dr. Anahita Jamshidnejad** – *Responsible Professor*
