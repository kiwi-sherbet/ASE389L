import numpy as np
import torch
from torch import nn
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

class mlp(nn.Module):

    def __init__(
        self,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()

        self.device = device
        self.input_dim = 15
        self.layer_sizes = (self.input_dim, 256, 256)
        self.activation = (nn.ReLU(), nn.ReLU())

        model = []
        for in_dim, out_dim, activ in zip(self.layer_sizes[:-1], self.layer_sizes[1:], self.activation):
            model += [nn.Linear(in_dim, out_dim), activ]

        self.output_dim = self.layer_sizes[-1]
        self.model = nn.Sequential(*model)


    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:

        s = torch.as_tensor(s,
                            device=self.device, dtype=torch.float32,)
        logits = self.model(s.flatten(1))

        return logits, state

