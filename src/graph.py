import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from collections import deque

from .obstacles import AdvancedObstacleGenerator


PATTERN_METHODS = ['spiral', 'corridor', 'clustered', 'funnel']


class KNNMotionGraph:
    """Builds a KNN-based motion graph over a 2D map with collision checking."""

    def __init__(self, map_size=10, resolution=0.5, k_neighbors=4,
                 robot_radius_map_units=0.2, pattern_type='spiral'):
        self.map_size = map_size
        self.resolution = resolution
        self.k_neighbors = k_neighbors
        self.robot_radius_map_units = robot_radius_map_units
        self.robot_radius_grid_units = robot_radius_map_units / resolution
        self.pattern_type = pattern_type
        self.obstacle_generator = AdvancedObstacleGenerator(map_size, resolution)
        self.obstacle_map = self._generate_obstacles()
        self.grid_width = int(self.map_size / self.resolution)
        self.grid_height = int(self.map_size / self.resolution)

    def _generate_obstacles(self):
        methods = {
            'spiral': self.obstacle_generator.spiral_pattern,
            'corridor': self.obstacle_generator.corridor_pattern,
            'clustered': self.obstacle_generator.clustered_pattern,
            'funnel': self.obstacle_generator.funnel_pattern,
        }
        if self.pattern_type not in methods:
            print(f"Unknown pattern '{self.pattern_type}', falling back to spiral.")
            return self.obstacle_generator.spiral_pattern()
        return methods[self.pattern_type]()

    def _compute_obstacle_proximity(self, positions):
        proximity = []
        for x, y in positions:
            grid_x, grid_y = int(x / self.resolution), int(y / self.resolution)
            near = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = grid_x + dx, grid_y + dy
                    if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height
                            and self.obstacle_map[nx, ny] == 1):
                        near = 1
                        break
                if near:
                    break
            proximity.append(near)
        return torch.tensor(proximity, dtype=torch.float32).unsqueeze(1)

    def build_graph(self, num_nodes=150, start=None, goal=None):
        nodes = torch.rand(num_nodes, 3) * torch.tensor([self.map_size, self.map_size, 2 * np.pi])

        if start is not None:
            nodes = torch.cat([start.unsqueeze(0), nodes])
        if goal is not None:
            nodes = torch.cat([nodes, goal.unsqueeze(0)])

        free_mask = torch.tensor([
            not self.is_collision((x.item(), y.item())) for x, y in nodes[:, :2]
        ])
        nodes = nodes[free_mask]

        if len(nodes) < 2:
            return None

        coords = nodes[:, :2].numpy()
        knn = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(nodes) - 1))
        knn.fit(coords)
        _, indices = knn.kneighbors(coords)

        edge_index, edge_weights = [], []
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i != j and self._valid_edge(nodes[i], nodes[j]):
                    dist = torch.norm(nodes[i, :2] - nodes[j, :2])
                    edge_index += [[i, j], [j, i]]
                    edge_weights += [dist, dist]

        if not edge_index:
            return None

        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

        proximity = self._compute_obstacle_proximity(nodes[:, :2])
        goal_idx = len(nodes) - 1 if goal is not None else None
        if goal_idx is not None:
            dist_to_goal = torch.norm(nodes[:, :2] - nodes[goal_idx, :2], dim=1, keepdim=True)
        else:
            dist_to_goal = torch.zeros(len(nodes), 1)

        node_features = torch.cat([nodes, dist_to_goal, proximity], dim=1)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights)

    def _valid_edge(self, n1, n2) -> bool:
        dx, dy = n2[0] - n1[0], n2[1] - n1[1]
        dist = torch.sqrt(dx ** 2 + dy ** 2)
        if dist > 3.0:
            return False
        steps = max(5, int(dist / (self.resolution / 4)))
        for t in torch.linspace(0, 1, steps):
            if self.is_collision(((n1[0] + t * dx).item(), (n1[1] + t * dy).item())):
                return False
        return True

    def is_collision(self, pos) -> bool:
        x_map, y_map = pos
        if not (0 <= x_map / self.resolution < self.grid_width
                and 0 <= y_map / self.resolution < self.grid_height):
            return True

        r = self.robot_radius_map_units
        min_gx = max(0, int(np.floor((x_map - r) / self.resolution)))
        max_gx = min(self.grid_width, int(np.ceil((x_map + r) / self.resolution)))
        min_gy = max(0, int(np.floor((y_map - r) / self.resolution)))
        max_gy = min(self.grid_height, int(np.ceil((y_map + r) / self.resolution)))

        for gx in range(min_gx, max_gx):
            for gy in range(min_gy, max_gy):
                if not (0 <= gx < self.grid_width and 0 <= gy < self.grid_height):
                    continue
                if self.obstacle_map[gx, gy] == 1:
                    cx = max(gx * self.resolution, min(x_map, (gx + 1) * self.resolution))
                    cy = max(gy * self.resolution, min(y_map, (gy + 1) * self.resolution))
                    if np.sqrt((x_map - cx) ** 2 + (y_map - cy) ** 2) < r:
                        return True
        return False

    def _is_connected(self, graph, start_node, end_node) -> bool:
        if start_node == end_node:
            return True
        adj = {i: [] for i in range(graph.num_nodes)}
        for i in range(graph.edge_index.shape[1]):
            u, v = graph.edge_index[:, i].tolist()
            adj[u].append(v)

        visited = [False] * graph.num_nodes
        queue = deque([start_node])
        visited[start_node] = True
        while queue:
            u = queue.popleft()
            if u == end_node:
                return True
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return False
