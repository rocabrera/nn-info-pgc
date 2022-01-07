import torch
import torch.nn as nn
from pathlib import Path


class MLP(nn.Module):

    def __init__(self, model_arch):
        super(MLP, self).__init__()
        self.camadas = nn.ModuleList([nn.Linear(before, after)
                                      for before, after in model_arch.parse_architecture()])  # achar nomes melhores
        self.activation_function = model_arch.activation_function

        Path("outputs").mkdir(parents=True, exist_ok=True)
        self.layers_output = None

    def get_layers_output(self):
        return self.layers_output

    def forward(self, X):

        self.layers_output = []

        out = self.camadas[0](X)
        with torch.no_grad():
            self.layers_output.append(out.cpu().numpy())
        for camada in self.camadas[1:]:
            out = self.activation_function(out)
            out = camada(out)
            with torch.no_grad():
                self.layers_output.append(out.cpu().numpy())

        return out
