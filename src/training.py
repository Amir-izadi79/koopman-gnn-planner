import torch
import torch.nn.functional as F

from .utils import differential_drive_next_state


def compute_loss(model, graph, goal_idx, path_indices, map_size):
    model.train()
    path_scores, critical_probs, psi, psi_next = model(graph)

    path_loss = F.mse_loss(path_scores.squeeze(), graph.x[:, -2])

    true_next_states = torch.stack([
        differential_drive_next_state(s[:3]) for s in graph.x
    ])
    koopman_loss = F.mse_loss(psi_next, model.koopman_proj(true_next_states))
    collision_loss = torch.mean(graph.x[:, -1] * critical_probs.squeeze())

    path_length_tensor = torch.tensor(0.0, device=graph.x.device)
    valid_segments = 0
    if len(path_indices) > 1:
        for i in range(len(path_indices) - 1):
            src, dst = path_indices[i], path_indices[i + 1]
            mask = (
                ((graph.edge_index[0] == src) & (graph.edge_index[1] == dst)) |
                ((graph.edge_index[0] == dst) & (graph.edge_index[1] == src))
            )
            if mask.any():
                path_length_tensor += graph.edge_attr[mask].min()
                valid_segments += 1
        if valid_segments > 0:
            path_length_tensor = path_length_tensor / valid_segments
        else:
            path_length_tensor = torch.tensor(map_size * 2, dtype=torch.float32,
                                              device=graph.x.device)

    smoothness_loss = torch.tensor(0.0, device=graph.x.device)
    if len(path_indices) >= 3:
        positions = graph.x[path_indices, :2]
        v1 = positions[1:-1] - positions[:-2]
        v2 = positions[2:] - positions[1:-1]
        v1_n = v1 / (torch.norm(v1, dim=1, keepdim=True) + 1e-6)
        v2_n = v2 / (torch.norm(v2, dim=1, keepdim=True) + 1e-6)
        cos_theta = torch.clamp(torch.sum(v1_n * v2_n, dim=1), -1.0, 1.0)
        smoothness_loss = torch.mean(torch.acos(cos_theta) ** 2)

    total_loss = (
        path_loss * 1.0
        + koopman_loss * 0.7
        + collision_loss * 0.6
        + path_length_tensor * 0.6
        + smoothness_loss * 1.0
    )

    return total_loss, {
        'total_loss': total_loss.item(),
        'path_loss': path_loss.item(),
        'koopman_loss': koopman_loss.item(),
        'collision_loss': collision_loss.item(),
        'path_length': path_length_tensor.item(),
        'smoothness_loss': smoothness_loss.item(),
    }
