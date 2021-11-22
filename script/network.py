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

        s = torch.as_tensor(s, device=self.device, dtype=torch.float32,)

        logits = self.model(s.flatten(1))

        return logits, state



class cnn_mlp(nn.Module):
    def __init__(
            self,
            device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()

        self.device = device

        ## CNN Model ##

        self.cnn_input_dim = (20, 20)
        self.cnn_input_size = np.prod(self.cnn_input_dim)
        self.cnn_input_channel = 1
        self.cnn_channel_sizes = (self.cnn_input_channel, 32, 32)
        self.cnn_kernel_sizes = (4, 4)
        self.cnn_activation = (nn.ReLU(), nn.ReLU())
        self.cnn_output_dim = 12

        size_cnn = np.array(self.cnn_input_dim)

        cnn_model = []
        for idx in range(len(self.cnn_channel_sizes) -1):
            in_dim = self.cnn_channel_sizes[idx]
            out_dim = self.cnn_channel_sizes[idx+1]
            kernel_dim = self.cnn_kernel_sizes[idx]
            activation = self.cnn_activation[idx]

            cnn_model += [nn.Conv2d(in_dim, out_dim, kernel_size=kernel_dim, padding=1), activation]

            size_cnn += 2 - kernel_dim 

            if idx < len(self.cnn_channel_sizes) - 2:
                cnn_model += [nn.MaxPool2d(2)]
                size_cnn = np.floor(size_cnn/2) + 1
            else:
                cnn_model += [nn.Softmax2d()]

        cnn_out_dim = int(np.prod(size_cnn)) * out_dim
        self.cnn_model = nn.Sequential(*cnn_model)
        self.proj = nn.Linear(cnn_out_dim, self.cnn_output_dim)

        ## MLP Model ##

        self.mlp_input_dim = 15 + self.cnn_output_dim
        self.mlp_layer_sizes = (self.mlp_input_dim, 256, 256)
        self.mlp_activation = (nn.ReLU(), nn.ReLU())

        mlp_model = []
        for in_dim, out_dim, activ in zip(self.mlp_layer_sizes[:-1], self.mlp_layer_sizes[1:], self.mlp_activation):
            mlp_model += [nn.Linear(in_dim, out_dim), activ]

        self.output_dim = self.mlp_layer_sizes[-1]
        self.mlp_model = nn.Sequential(*mlp_model)



    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:

        s = torch.as_tensor(s, device=self.device, dtype=torch.float32,)
        o1 = s.narrow(1, 0, 15)

        t1 = s.narrow(1, 15, self.cnn_input_size)
        t1 = t1.reshape((-1, 1, self.cnn_input_dim[0], self.cnn_input_dim[1]))
        t2 = self.cnn_model(t1)

        o2 = self.proj(t2.flatten(1))

        o = torch.cat((o1, o2), dim=1)

        out = self.mlp_model(o)

        return out, state
