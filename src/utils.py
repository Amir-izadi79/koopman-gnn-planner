import random
import numpy as np
import torch


def differential_drive_next_state(state, v=0.5, w=0.2, dt=0.1):
    x, y, theta = state
    x_new = x + v * torch.cos(theta) * dt
    y_new = y + v * torch.sin(theta) * dt
    theta_new = theta + w * dt
    theta_new = torch.atan2(torch.sin(theta_new), torch.cos(theta_new))
    return torch.tensor([x_new, y_new, theta_new])


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
