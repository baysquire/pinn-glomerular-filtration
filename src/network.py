import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers):
        """
        2D Physics-Informed Neural Network Architecture.
        Args:
            layers: List of neurons per layer (e.g., [3, 40, 40, 40, 40, 1] for x, y, t inputs)
        """
        super(PINN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layers) - 2):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i+1]))
            
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        
        # Tanh is critical for 2nd order differential equations
        self.activation = nn.Tanh()

    def forward(self, x, y, t):
        """
        Inputs: 
            x: Length along capillary
            y: Width across capillary
            t: Time
        Output: C (Concentration)
        """
        inputs = torch.cat([x, y, t], dim=1)
        for layer in self.hidden_layers:
            inputs = self.activation(layer(inputs))
        C = self.output_layer(inputs)
        return C
   