import math
from typing import Optional
import torch
import torch.nn as nn


class NoiseLayer(nn.Module):

    def __init__(self, mean = 0, variance = 0.1) -> None:
        super().__init__()
        self.mean     = mean
        self.variance = variance

        if self.variance == 0:
            raise ValueError("Variance should not be 0")

        # Precalculate
        self.variance_multiplier = math.sqrt(variance)

    # TODO Maybe change the part with `output = torch.zeros_like(x)`
    def forward(self, x: torch.tensor, where: Optional[torch.tensor] = None):
        # When we are not training we do not add any noise
        if not self.training:
            return x
            
        output = torch.zeros_like(x)
        if where is not None:
            noise = self.mean + self.variance_multiplier * torch.randn((torch.count_nonzero(where), *list(x.shape)[len(list(where.shape)):])).to(x.device)

            noise.to(x.device)
            output[where] += noise
        else:
            noise = self.mean + self.variance_multiplier * torch.randn(x.shape)
            noise.to(x.device)
            output += noise

        return output + x

