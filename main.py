import os
import time
import random
import numpy as np
import torch
import pandas as pd

from src import (
    EDMD, KNNMotionGraph, NeuralPlanner,
    compute_loss, visualize_planning, create_comparison_plots,
    differential_drive_next_state, set_seeds,
)

PATTERNS = [
    {
        'name': 'Spiral Navigation',
        'type': 'spiral',
        'description': 'Curved spiral obstacles testing smooth trajectory following',
        'expected_challenge': 'Curved path navigation and trajectory smoothness',
    },
    {
        'name': 'Narrow Corridors',
        'type': 'corridor',
        'description': 'Tight passages testing precision navigation',
        'expected_challenge': 'Precision navigation in constrained spaces',
    },
    {
        'name': 'Clustered Obstacles',
        'type': 'clustered',
        'description': 'Grouped obstacles testing local minima avoidance',
        'expected_challenge': 'Local minima avoidance and cluster navigation',
    },
    {
        'name': 'Funnel Navigation',
        'type': 'funnel',
        'description': 'Narrowing passages testing progressive constraint handling',
        'expected_challenge': 'Adaptive planning with progressive space reduction',
    },
]

# ── Hyper-parameters ──────────────────────────────────────────────────────────
MAP_SIZE = 10.0
RESOLUTION = 0.35
NUM_GRAPH_NODES = 150
K_NEIGHBORS = 8
KOOPMAN_OBS_DIM = 50
HIDDEN_DIM = 64
NODE_FEATURE_DIM = 5   # [x, y, theta, dist_to_goal, obstacle_proximity]
LEARNING_RATE = 0.001
EPOCHS = 60
ROBOT_RADIUS = 0.2
NUM_EDMD_SAMPLES = 2000
NUM_TRIALS = 3
VIZ_EPOCHS = 40
# ─────────────────────────────────────────────────────────────────────────────


def _build_edmd():
    current = torch.tensor([MAP_SIZE / 2, MAP_SIZE / 2, 0.0], dtype=torch.float32)
    X, Y = [], []
    for _ in range(NUM_EDMD_SAMPLES):
        X.append(current.clone())
        v = random.uniform(0.1, 0.8)
        w = random.uniform(-0.5, 0.5)
        nxt = differential_drive_next_state(current, v=v, w=w, dt=0.2)
        Y.append(nxt.clone())
        current = nxt
    edmd = EDMD(obs_dim=KOOPMAN_OBS_DIM)
    edmd.fit(torch.stack(X), torch.stack(Y))
    return edmd


def _path_length(graph, path_indices):
    total = 0.0
    for i in range(len(path_indices) - 1):
        src, dst = path_indices[i], path_indices[i + 1]
        mask = (
            ((graph.edge_index[0] == src) & (graph.edge_index[1] == dst)) |
            ((graph.edge_index[0] == dst) & (graph.edge_index[1] == src))
        )
        if mask.any():
            total += graph.edge_attr[mask].min().item()
    return total


def _train_model(model, graph, start_idx, goal_idx, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    best_length, best_path = float('inf'), []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        _, _, _, _, path_indices = model(graph, start_idx, goal_idx)
        loss, details = compute_loss(model, graph, goal_idx, path_indices, MAP_SIZE)
        loss.backward()
        optimizer.step()

        if len(path_indices) > 1:
            length = _path_length(graph, path_indices)
            if 0 < length < best_length:
                best_length, best_path = length, path_indices.copy()

        history.append({'epoch': epoch, **details, 'path_nodes': len(path_indices)})
        if (epoch + 1) % 15 == 0:
            print(f"  Epoch {epoch+1:2d}: Loss={details['total_loss']:.4f}, "
                  f"Nodes={len(path_indices):3d}, Best_Length={best_length:.2f}")

    return history, best_length, best_path


def run_comprehensive_analysis():
    os.makedirs("analysis_results", exist_ok=True)
    print("Neural Path Planning with Advanced Obstacle Patterns")
    print("=" * 70)

    all_results = []

    for idx, cfg in enumerate(PATTERNS):
        print(f"\nPattern {idx+1}/4: {cfg['name']}")
        print(f"Description: {cfg['description']}")
        print(f"Challenge:   {cfg['expected_challenge']}")
        print("-" * 50)

        print("Generating EDMD training data...")
        edmd = _build_edmd()
        print("EDMD training completed")

        graph_builder = KNNMotionGraph(
            map_size=MAP_SIZE, resolution=RESOLUTION,
            k_neighbors=K_NEIGHBORS, robot_radius_map_units=ROBOT_RADIUS,
            pattern_type=cfg['type'],
        )

        start_pos = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)
        goal_pos = torch.tensor([8.0, 8.0, np.pi / 2], dtype=torch.float32)

        if graph_builder.is_collision(start_pos[:2].tolist()):
            print(f"Start position collision in {cfg['name']}, skipping.")
            continue
        if graph_builder.is_collision(goal_pos[:2].tolist()):
            print(f"Goal position collision in {cfg['name']}, skipping.")
            continue

        print("Building motion graph...")
        graph = graph_builder.build_graph(NUM_GRAPH_NODES, start=start_pos, goal=goal_pos)
        if graph is None:
            print(f"Failed to build graph for {cfg['name']}, skipping.")
            continue

        start_idx, goal_idx = 0, graph.num_nodes - 1
        if not graph_builder._is_connected(graph, start_idx, goal_idx):
            print(f"Graph disconnected in {cfg['name']}, skipping.")
            continue

        obstacle_density = (torch.sum(graph_builder.obstacle_map).item()
                            / (graph_builder.grid_width * graph_builder.grid_height))
        print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges  "
              f"| Obstacle density: {obstacle_density:.3f}")

        pattern_results = []
        for trial in range(NUM_TRIALS):
            print(f"\nTrial {trial+1}/{NUM_TRIALS}")
            model = NeuralPlanner(NODE_FEATURE_DIM, HIDDEN_DIM, KOOPMAN_OBS_DIM, edmd)
            _train_model(model, graph, start_idx, goal_idx, EPOCHS)

            model.eval()
            success_count, path_lengths, planning_times = 0, [], []
            for _ in range(5):
                t0 = time.time()
                with torch.no_grad():
                    _, _, _, _, final_path = model(graph, start_idx, goal_idx)
                planning_times.append(time.time() - t0)
                if len(final_path) > 1:
                    success_count += 1
                    path_lengths.append(_path_length(graph, final_path))
                else:
                    path_lengths.append(0)

            sr = success_count / 5
            avg_len = np.mean([l for l in path_lengths if l > 0]) if any(path_lengths) else 0
            avg_time = np.mean(planning_times)
            print(f"  Success: {sr:.2f}, Avg Length: {avg_len:.2f}, Time: {avg_time:.4f}s")

            pattern_results.append({
                'pattern': cfg['name'], 'trial': trial,
                'success_rate': sr, 'avg_path_length': avg_len,
                'avg_planning_time': avg_time, 'obstacle_density': obstacle_density,
            })

        avg_success = np.mean([r['success_rate'] for r in pattern_results])
        valid_lengths = [r['avg_path_length'] for r in pattern_results if r['avg_path_length'] > 0]
        avg_length = np.mean(valid_lengths) if valid_lengths else 0
        avg_time = np.mean([r['avg_planning_time'] for r in pattern_results])

        all_results.append({
            'pattern_name': cfg['name'], 'pattern_type': cfg['type'],
            'description': cfg['description'],
            'avg_success_rate': avg_success,
            'avg_path_length': avg_length if not np.isnan(avg_length) else 0,
            'avg_planning_time': avg_time,
            'obstacle_density': obstacle_density,
            'num_trials': NUM_TRIALS,
        })

        if avg_success > 0:
            print(f"Creating visualization for {cfg['name']}...")
            viz_model = NeuralPlanner(NODE_FEATURE_DIM, HIDDEN_DIM, KOOPMAN_OBS_DIM, edmd)
            _train_model(viz_model, graph, start_idx, goal_idx, VIZ_EPOCHS)
            viz_model.eval()
            with torch.no_grad():
                _, _, _, _, viz_path = viz_model(graph, start_idx, goal_idx)
            visualize_planning(
                graph, viz_path, graph_builder.obstacle_map,
                RESOLUTION, ROBOT_RADIUS, MAP_SIZE,
                title=f"{cfg['name']} — Success Rate: {avg_success:.2f}",
            )

        print(f"{cfg['name']} completed: Success={avg_success:.2f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 70)

    if all_results:
        print(f"{'Pattern':<22} {'Success':>8} {'Path Len':>10} {'Time (s)':>10} {'Density':>8}")
        print("-" * 65)
        for r in all_results:
            print(f"{r['pattern_name']:<22} {r['avg_success_rate']:>8.2f} "
                  f"{r['avg_path_length']:>10.2f} {r['avg_planning_time']:>10.4f} "
                  f"{r['obstacle_density']:>8.3f}")

        df = pd.DataFrame(all_results)
        df.to_csv('analysis_results/comprehensive_results.csv', index=False)
        print("\nResults saved to 'analysis_results/comprehensive_results.csv'")
        create_comparison_plots(all_results)
    else:
        print("No successful results to analyze.")

    return all_results


if __name__ == "__main__":
    set_seeds(42)
    print("Neural Path Planning with Advanced Obstacle Patterns")
    print("Patterns: Spiral | Narrow Corridors | Clustered Obstacles | Funnel\n")
    results = run_comprehensive_analysis()
    if results:
        print(f"\nAnalysis completed — {len(results)} patterns evaluated.")
        print("Results and visualizations saved to 'analysis_results/'")
    else:
        print("\nAnalysis failed — no successful results obtained.")
