import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import *
class SkillsPredictorModel(nn.Module):
    def __init__(self, nG, layers=[64, 64], log_frequency = 1):
        super(SkillsPredictorModel, self).__init__()
        self.layers = nn.ModuleList()  
        self.log_frequency = log_frequency
        
        input_dim = nG
        for layer_dim in layers:
            self.layers.append(nn.Linear(input_dim, layer_dim))
            input_dim = layer_dim

        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = F.gelu(layer(x))   
        x = self.output_layer(x)
        return torch.sigmoid(x)

    def log(self,step, prefix="skills_predictor"):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                log_param_wandb(f"{prefix}/fc{i}", layer, step, self.log_frequency)

            # Log output layer
            if isinstance(self.output_layer, nn.Linear):
                log_param_wandb(f"{prefix}/output_layer", self.output_layer, step, self.log_frequency)       
