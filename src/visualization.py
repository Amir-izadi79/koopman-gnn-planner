import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_planning(graph, path_indices, obstacle_map, resolution,
                       robot_radius, map_size, title=""):
    plt.figure(figsize=(12, 10))

    for i in range(obstacle_map.shape[0]):
        for j in range(obstacle_map.shape[1]):
            if obstacle_map[i, j] == 1:
                rect = patches.Rectangle(
                    (i * resolution, j * resolution), resolution, resolution,
                    linewidth=0.5, edgecolor='black', facecolor='darkred', alpha=0.8
                )
                plt.gca().add_patch(rect)

    pos = graph.x[:, :2].numpy()
    plt.scatter(pos[:, 0], pos[:, 1], c='lightblue', s=15, alpha=0.6, zorder=2)

    for i in range(graph.edge_index.shape[1]):
        s, d = graph.edge_index[:, i].tolist()
        plt.plot([pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
                 'gray', linewidth=0.3, alpha=0.4, zorder=1)

    if len(path_indices) > 1:
        pp = graph.x[path_indices, :2].numpy()
        plt.plot(pp[:, 0], pp[:, 1], 'lime', linewidth=3,
                 label=f'Neural Path ({len(path_indices)} nodes)', zorder=3)
        plt.scatter(pp[:, 0], pp[:, 1], c='green', s=30, zorder=4)

    start_pos = graph.x[0, :2].numpy()
    goal_pos = graph.x[-1, :2].numpy()

    plt.gca().add_patch(plt.Circle(start_pos, robot_radius, color='blue', alpha=0.3))
    plt.gca().add_patch(plt.Circle(goal_pos, robot_radius, color='red', alpha=0.3))
    plt.scatter(*start_pos, c='blue', s=150, marker='s', label='Start', zorder=5,
                edgecolors='darkblue', linewidth=2)
    plt.scatter(*goal_pos, c='red', s=150, marker='*', label='Goal', zorder=5,
                edgecolors='darkred', linewidth=2)

    plt.xlim(0, map_size)
    plt.ylim(0, map_size)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('X Position', fontsize=12)
    plt.ylabel('Y Position', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()


def create_comparison_plots(results, save_path='analysis_results/comparison_plots.png'):
    if not results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    patterns = [r['pattern_name'] for r in results]

    def _annotate_bars(ax, bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h,
                    f'{h:.2f}', ha='center', va='bottom')

    success_rates = [r['avg_success_rate'] for r in results]
    bars = axes[0, 0].bar(patterns, success_rates, color=colors)
    axes[0, 0].set_title('Success Rate by Pattern', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_ylim(0, 1)
    _annotate_bars(axes[0, 0], bars)

    path_lengths = [r['avg_path_length'] for r in results]
    bars = axes[0, 1].bar(patterns, path_lengths, color=colors)
    axes[0, 1].set_title('Average Path Length by Pattern', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Path Length')
    _annotate_bars(axes[0, 1], bars)

    times_ms = [r['avg_planning_time'] * 1000 for r in results]
    bars = axes[1, 0].bar(patterns, times_ms, color=colors)
    axes[1, 0].set_title('Average Planning Time by Pattern', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Planning Time (ms)')
    _annotate_bars(axes[1, 0], bars)

    densities = [r['obstacle_density'] for r in results]
    for i, (d, s, p) in enumerate(zip(densities, success_rates, patterns)):
        axes[1, 1].scatter(d, s, c=colors[i], s=200, alpha=0.7, label=p)
    axes[1, 1].set_title('Success Rate vs Obstacle Density', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Obstacle Density')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comparison plots saved to '{save_path}'")
