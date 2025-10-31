import torch
import torch.nn as nn
from utils import *
import wandb

class AnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32], log_frequency = 1):
        super().__init__()
        # Encoder
        self.encoder = nn.ModuleList()
        in_features = input_dim
        for h in hidden_layers:
            self.encoder.append(nn.Linear(in_features, h))
            in_features = h
        
        # Decoder
        reversed_layers = list(reversed(hidden_layers))
        self.decoder = nn.ModuleList()
        in_features = reversed_layers[0]
        for h in reversed_layers[1:]:
            self.decoder.append(nn.Linear(in_features, h))
            in_features = h
        
        # Final decoder layer (to reconstruct input)
        self.decoder.append(nn.Linear(in_features, input_dim))
        self.activation = nn.ReLU()
        self.log_frequency =  log_frequency
        
    def forward(self, x):
        # Encoder
        for layer in self.encoder:
            x = self.activation(layer(x))
        latent = x
        
        # Decoder
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i < len(self.decoder) - 1:
                x = self.activation(x)
        return x
    
    def log(self,step, prefix="anomaly"):
        """
        Logs encoder/decoder layer weights, biases, and gradients using the existing log_param_wandb function.
        """
        # Log encoder layers
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Linear):
                log_param_wandb(f"{prefix}/encoder_fc{i}", layer, step, self.log_frequency)

        # Log decoder layers
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.Linear):
                log_param_wandb(f"{prefix}/decoder_fc{i}", layer, step, self.log_frequency)

        

def reconstruction_score(goals_tensor, model):
    reconstructed = model(goals_tensor)
    
    return torch.mean((goals_tensor - reconstructed) ** 2, dim=1)  


def log_p_valid(goals, model, step, temperature=0.1, prefix="anomaly"):
    
    score = reconstruction_score(goals, model)
    if step:
        wandb.log({f"{prefix}/reconstruction_loss": score.mean()}, step=step)
        wandb.log({f"{prefix}/reconstruction_loss_hist": wandb.Histogram(score.detach().cpu().numpy())}, step=step)
    return -score / temperature

