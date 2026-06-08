# Koopman-GNN Neural Path Planner

A neural path planning framework that combines **Extended Dynamic Mode Decomposition (EDMD)** for Koopman operator learning with **Graph Attention Networks (GAT)** for obstacle-aware motion planning in 2D environments.

## Overview

The planner represents the robot's configuration space as a KNN motion graph and trains a hybrid model that:

- Learns **globally linearized dynamics** via Koopman operator projection (EDMD with RBF observables)
- Captures **local nonlinear relationships** between graph nodes using multi-head GATs
- Plans paths with a **neural Dijkstra** that weighs edge distances, obstacle proximity, and model-predicted criticality scores

Four distinct obstacle environments are used to benchmark the system:

| Pattern | Challenge |
|---|---|
| Spiral | Smooth curved trajectory following |
| Narrow Corridors | Precision navigation in constrained spaces |
| Clustered Obstacles | Local minima avoidance |
| Funnel | Adaptive planning under progressive space reduction |

## Project Structure

```
koopman-gnn-planner/
├── main.py                  # Entry point — runs full benchmark
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── edmd.py              # EDMD / Koopman operator fitting
│   ├── obstacles.py         # Four obstacle pattern generators
│   ├── graph.py             # KNN motion graph builder + collision checking
│   ├── models.py            # KoopmanGAT and NeuralPlanner nn.Module
│   ├── training.py          # Multi-objective loss function
│   ├── visualization.py     # Matplotlib planning & comparison plots
│   └── utils.py             # Differential-drive kinematics, seed helper
└── analysis_results/        # Output CSVs and plots (git-ignored)
```

## Installation

```bash
pip install -r requirements.txt
```

> `torch-geometric` may require platform-specific wheels — see the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Usage

```bash
python main.py
```

The script will:
1. Generate EDMD training trajectories for each pattern
2. Build a KNN motion graph over the free configuration space
3. Train `NeuralPlanner` for 60 epochs per trial (3 trials × 4 patterns)
4. Evaluate success rate, average path length, and planning time
5. Save results to `analysis_results/comprehensive_results.csv`
6. Save a comparison bar-chart to `analysis_results/comparison_plots.png`

## Module Reference

### `src.edmd.EDMD`
Fits a Koopman operator `K` via least-squares regression over RBF observable snapshots.

```python
edmd = EDMD(obs_dim=50)
edmd.fit(X_states, Y_next_states)   # numpy arrays or torch tensors
psi_next = edmd.predict(x)
```

### `src.obstacles.AdvancedObstacleGenerator`
Returns a `torch.Tensor` binary occupancy grid for one of four patterns.

```python
gen = AdvancedObstacleGenerator(map_size=10, resolution=0.35)
occ = gen.spiral_pattern()   # or corridor_pattern / clustered_pattern / funnel_pattern
```

### `src.graph.KNNMotionGraph`
Builds a collision-free `torch_geometric.data.Data` graph with node features
`[x, y, θ, dist_to_goal, obstacle_proximity]`.

```python
mg = KNNMotionGraph(map_size=10, resolution=0.35, k_neighbors=8,
                    robot_radius_map_units=0.2, pattern_type='spiral')
graph = mg.build_graph(num_nodes=150, start=start_tensor, goal=goal_tensor)
```

### `src.models.NeuralPlanner`
PyTorch `nn.Module` combining Koopman projection + 2-layer GAT + neural Dijkstra.

```python
model = NeuralPlanner(node_dim=5, hidden_dim=64, koopman_dim=50, edmd=edmd)
path_scores, critical_probs, psi, psi_next, path = model(graph, start_idx, goal_idx)
```

### `src.training.compute_loss`
Five-term loss: path regression · Koopman consistency · collision · path length · smoothness.

```python
loss, details = compute_loss(model, graph, goal_idx, path_indices, map_size=10.0)
```

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `MAP_SIZE` | 10.0 | Environment size (m) |
| `RESOLUTION` | 0.35 | Grid cell size (m) |
| `NUM_GRAPH_NODES` | 150 | Sampled roadmap nodes |
| `K_NEIGHBORS` | 8 | KNN edges per node |
| `KOOPMAN_OBS_DIM` | 50 | RBF observable / Koopman state dimension |
| `HIDDEN_DIM` | 64 | GAT hidden width |
| `EPOCHS` | 60 | Training epochs per trial |
| `ROBOT_RADIUS` | 0.2 | Collision checking radius (m) |

## License

MIT
