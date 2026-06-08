import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class KoopmanGAT(nn.Module):
    """Hybrid model: linearized global Koopman dynamics + local GAT nonlinear relationships."""

    def __init__(self, node_dim, hidden_dim, koopman_dim, edmd):
        super().__init__()
        self.edmd = edmd

        self.koopman_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, koopman_dim),
        )
        self.K = nn.Parameter(torch.tensor(edmd.K, dtype=torch.float32))

        self.gat1 = GATConv(node_dim, hidden_dim, edge_dim=1, heads=4, dropout=0.2)
        self.gat2 = GATConv(hidden_dim * 4 + koopman_dim, hidden_dim, edge_dim=1,
                            heads=1, concat=False, dropout=0.2)

        self.path_head = nn.Linear(hidden_dim, 1)
        self.critical_head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr
        norm_weights = (edge_weights - edge_weights.mean()) / (edge_weights.std() + 1e-6)

        psi = self.koopman_proj(x[:, :3])
        psi_next = torch.mm(psi, self.K)

        h_gat = F.elu(self.gat1(x, edge_index, edge_attr=norm_weights))
        h_combined = torch.cat([h_gat, psi], dim=1)
        h_out = self.gat2(h_combined, edge_index, edge_attr=norm_weights)

        path_scores = self.path_head(h_out)
        critical_probs = torch.sigmoid(self.critical_head(h_out))
        return path_scores, critical_probs, psi, psi_next


class NeuralPlanner(KoopmanGAT):
    """Extends KoopmanGAT with Dijkstra-based neural path planning."""

    def __init__(self, node_dim, hidden_dim, koopman_dim, edmd):
        super().__init__(node_dim, hidden_dim, koopman_dim, edmd)

    def neural_plan(self, graph, start_idx, goal_idx):
        self.eval()
        with torch.no_grad():
            path_scores, critical_probs, _, _ = super().forward(graph)

        dist = {i: float('inf') for i in range(graph.num_nodes)}
        prev = {i: None for i in range(graph.num_nodes)}
        dist[start_idx] = 0
        pq = [(0, start_idx)]

        while pq:
            cost, u = heapq.heappop(pq)
            if cost > dist[u]:
                continue
            if u == goal_idx:
                path = []
                while u is not None:
                    path.insert(0, u)
                    u = prev[u]
                return path

            mask = graph.edge_index[0] == u
            for i, v_tensor in enumerate(graph.edge_index[1, mask]):
                v = v_tensor.item()
                edge_cost = graph.edge_attr[mask][i].item()
                near_obs = graph.x[v, -1].item()
                obs_penalty = 1000.0 if near_obs == 1 else critical_probs[v].item() * 50.0
                goal_cost = path_scores[v].item() * 0.5
                new_cost = cost + edge_cost + obs_penalty + goal_cost
                if new_cost < dist[v]:
                    dist[v] = new_cost
                    prev[v] = u
                    heapq.heappush(pq, (new_cost, v))
        return []

    def forward(self, data, start_idx=None, goal_idx=None):
        data.x = data.x.float()
        path_scores, critical_probs, psi, psi_next = super().forward(data)
        if start_idx is not None and goal_idx is not None:
            path_indices = self.neural_plan(data, start_idx, goal_idx)
            return path_scores, critical_probs, psi, psi_next, path_indices
        return path_scores, critical_probs, psi, psi_next
