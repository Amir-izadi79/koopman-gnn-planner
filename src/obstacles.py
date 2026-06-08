import random
import numpy as np
import torch


class AdvancedObstacleGenerator:
    """Generates four distinct obstacle patterns for path planning benchmarks."""

    def __init__(self, map_size=10, resolution=0.35):
        self.map_size = map_size
        self.resolution = resolution
        self.grid_size = int(map_size / resolution)

    def spiral_pattern(self):
        """Spiral obstacles — tests curved trajectory following."""
        obstacle_map = torch.zeros(self.grid_size, self.grid_size)
        cx, cy = self.grid_size // 2, self.grid_size // 2
        max_radius = min(cx, cy) - 5

        for radius in range(3, max_radius, 3):
            for angle in np.linspace(0, 3 * np.pi, int(radius * 4)):
                x = int(cx + radius * np.cos(angle))
                y = int(cy + radius * np.sin(angle))
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                                obstacle_map[nx, ny] = 1

        return self._finalize_map(obstacle_map, "Spiral Pattern")

    def corridor_pattern(self):
        """Narrow corridor obstacles — tests constrained-space precision."""
        obstacle_map = torch.zeros(self.grid_size, self.grid_size)
        corridor_width = 4
        num_corridors = 3

        for i in range(num_corridors):
            y_center = (i + 1) * (self.grid_size // (num_corridors + 1))
            for y in range(self.grid_size):
                if abs(y - y_center) > corridor_width // 2:
                    for x in range(self.grid_size):
                        obstacle_map[x, y] = 1
            if i < num_corridors - 1:
                for x in range(5, self.grid_size - 5, 8):
                    obstacle_map[x, y_center] = 1
                    obstacle_map[x + 1, y_center] = 1

        for i in range(num_corridors - 1):
            x_connector = int((i + 1.5) * (self.grid_size // num_corridors))
            y1 = (i + 1) * (self.grid_size // (num_corridors + 1))
            y2 = (i + 2) * (self.grid_size // (num_corridors + 1))
            for y in range(min(y1, y2), max(y1, y2) + 1):
                for x in range(max(0, x_connector - 2), min(self.grid_size, x_connector + 3)):
                    obstacle_map[x, y] = 0

        return self._finalize_map(obstacle_map, "Corridor Pattern")

    def clustered_pattern(self):
        """Clustered rectangular obstacles — tests local-minima avoidance."""
        obstacle_map = torch.zeros(self.grid_size, self.grid_size)
        num_clusters = 5
        cluster_centers = []

        while len(cluster_centers) < num_clusters:
            cx = random.randint(6, self.grid_size - 6)
            cy = random.randint(6, self.grid_size - 6)
            too_close = any(
                np.sqrt((cx - ex) ** 2 + (cy - ey) ** 2) < 6
                for ex, ey in cluster_centers
            )
            if not too_close:
                cluster_centers.append((cx, cy))

        for cx, cy in cluster_centers:
            for _ in range(random.randint(4, 7)):
                radius = random.randint(1, 4)
                angle = random.uniform(0, 2 * np.pi)
                x = int(cx + radius * np.cos(angle))
                y = int(cy + radius * np.sin(angle))
                w, h = random.randint(2, 4), random.randint(2, 4)
                for dx in range(w):
                    for dy in range(h):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            obstacle_map[nx, ny] = 1

        return self._finalize_map(obstacle_map, "Clustered Pattern")

    def funnel_pattern(self):
        """Progressively narrowing funnel — tests adaptive planning under tightening constraints."""
        obstacle_map = torch.zeros(self.grid_size, self.grid_size)
        center_y = self.grid_size // 2

        for x in range(self.grid_size):
            progress = x / self.grid_size
            funnel_width = int((1 - progress * 0.7) * (self.grid_size // 2 - 3)) + 3
            for y in range(self.grid_size):
                if abs(y - center_y) > funnel_width:
                    obstacle_map[x, y] = 1

        for x in range(5, self.grid_size - 5, 7):
            progress = x / self.grid_size
            funnel_width = int((1 - progress * 0.7) * (self.grid_size // 2 - 3)) + 3
            if funnel_width > 4:
                sign = 1 if x % 14 == 5 else -1
                oy = center_y + sign * (funnel_width // 2 - 1)
                if 0 <= oy < self.grid_size:
                    obstacle_map[x, oy] = 1
                    next_oy = oy + sign
                    if 0 <= next_oy < self.grid_size:
                        obstacle_map[x, next_oy] = 1

        return self._finalize_map(obstacle_map, "Funnel Pattern")

    def _finalize_map(self, obstacle_map, pattern_name):
        obstacle_map[0, :] = 1
        obstacle_map[-1, :] = 1
        obstacle_map[:, 0] = 1
        obstacle_map[:, -1] = 1

        start_grid = (int(1.0 / self.resolution), int(1.0 / self.resolution))
        goal_grid = (int(8.0 / self.resolution), int(8.0 / self.resolution))
        clear_radius = 3

        for dx in range(-clear_radius, clear_radius + 1):
            for dy in range(-clear_radius, clear_radius + 1):
                sx, sy = start_grid[0] + dx, start_grid[1] + dy
                if 0 <= sx < self.grid_size and 0 <= sy < self.grid_size:
                    obstacle_map[sx, sy] = 0
                gx, gy = goal_grid[0] + dx, goal_grid[1] + dy
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    obstacle_map[gx, gy] = 0

        density = torch.sum(obstacle_map).item() / (self.grid_size * self.grid_size)
        print(f"Generated {pattern_name} with density: {density:.3f}")
        return obstacle_map
