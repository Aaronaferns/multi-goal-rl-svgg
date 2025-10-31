import numpy as np
import torch
import random
import torch.nn as nn
import os
import wandb



def wandb_log(key, value, step, log_frequency=1):
    """Lightweight scalar logging."""
    if step % log_frequency == 0:
        if torch.is_tensor(value):
            value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().numpy()
        wandb.log({key: value}, step=step)


def log_param_wandb(key, layer, step, log_frequency=100):
    """Logs histograms for weights, biases, and their gradients (detached safely)."""
    if step % log_frequency != 0:
        return

    # Always detach and move to CPU before converting to numpy
    if hasattr(layer, 'weight') and layer.weight is not None:
        wandb.log({
            f"{key}/weight": wandb.Histogram(layer.weight.detach().cpu().numpy())
        }, step=step)

        if layer.weight.grad is not None:
            wandb.log({
                f"{key}/weight_grad": wandb.Histogram(layer.weight.grad.detach().cpu().numpy())
            }, step=step)

    if hasattr(layer, 'bias') and layer.bias is not None:
        wandb.log({
            f"{key}/bias": wandb.Histogram(layer.bias.detach().cpu().numpy())
        }, step=step)

        if layer.bias.grad is not None:
            wandb.log({
                f"{key}/bias_grad": wandb.Histogram(layer.bias.grad.detach().cpu().numpy())
            }, step=step)

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()



