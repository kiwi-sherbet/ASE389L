import torch
import numpy as np
from torch import nn
from typing import Any, Dict, Tuple, Union, Optional, Sequence

from tianshou.utils.net.common import MLP, ModuleType


class AdaptationEncoder(nn.Module):
    def __init__(self,  numChannelInput: int = 62,
                        numChannelEmbedding: int = 32,
                        lenHistoryInput: int = 50,
                        sizeOutput: int = 8, 
                        device: Union[str, int, torch.device] = "cpu",) -> None:
        super().__init__()

        self.device = device

        self.numChannelInput = numChannelInput
        self.numChannelEmbedding = numChannelEmbedding
        self.lenHistoryInput = lenHistoryInput
        self.sizeOutput = sizeOutput

        sizeProj = self.lenHistoryInput
        raConv = []
        for input, output, kernel, stride in ((self.numChannelEmbedding, self.numChannelEmbedding, 8, 4),
                                              (self.numChannelEmbedding, self.numChannelEmbedding, 5, 1),
                                              (self.numChannelEmbedding, self.numChannelEmbedding, 5, 1)):

            sizeProj = int(np.floor((sizeProj-kernel)/stride + 1))
            raConv += [nn.Conv1d(self.numChannelEmbedding, self.numChannelEmbedding, kernel_size=kernel, stride=stride)]
            # raConv += [nn.BatchNorm1d(output), nn.ReLU(inplace=True)]
            raConv += [nn.ReLU(inplace=True)]

        self.embd = nn.Sequential(nn.Linear(self.numChannelInput, self.numChannelEmbedding),  nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*raConv)
        self.proj = nn.Linear(sizeProj*self.numChannelEmbedding, self.sizeOutput)


    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)  # type: ignore

        if type(x) is type(torch.Tensor([])):
            assert not torch.isnan(x).any()
        elif type(x) is type(np.array([])):
            assert not np.isnan(x).any()

        h0 = x.reshape((-1, self.lenHistoryInput, self.numChannelInput))

        if type(h0) is type(torch.Tensor([])):
            assert not torch.isnan(h0).any()
        elif type(h0) is type(np.array([])):
            assert not np.isnan(h0).any()

        h1 = self.embd(h0)

        if type(h1) is type(torch.Tensor([])):
            assert not torch.isnan(h1).any()
        elif type(h1) is type(np.array([])):
            assert not np.isnan(h1).any()

        h2 = self.conv(h1.transpose(1,2))

        if type(h2) is type(torch.Tensor([])):
            assert not torch.isnan(h2).any()
        elif type(h2) is type(np.array([])):
            assert not np.isnan(h2).any()

        out = self.proj(h2.flatten(1))

        return out


class AdaptationNet(nn.Module):

    def __init__(
        self,
        sizeState: int,
        sizeAction: int = 0,
        lenHistory: int = 0,
        sizeEmbedding: int = 32,
        sizeFeature: int = 8,
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        num_atoms: int = 1,
    ) -> None:
        super().__init__()

        self.device = device

        self.sizeState = sizeState
        self.sizeAction = sizeAction

        self.output_dim = self.sizeAction

        self.modelEncoder = AdaptationEncoder(numChannelInput = sizeState + sizeAction,
                                              numChannelEmbedding = sizeEmbedding,
                                              lenHistoryInput = lenHistory,
                                              sizeOutput = sizeFeature,
                                              device= device)

        self.modelPolicy = MLP(sizeFeature + sizeState + sizeAction, sizeAction, (256, 128),
                               norm_layer, activation, device)


    def forward(
        self, s: Union[np.ndarray, torch.Tensor], state: Any = None, info: Dict[str, Any] = {}, ) -> Tuple[torch.Tensor, Any]:

        s = torch.as_tensor(s, device=self.device, dtype=torch.float32)

        if type(s) is type(torch.Tensor([])):
            assert not torch.isnan(s).any()
        elif type(s) is type(np.array([])):
            assert not np.isnan(s).any()

        o1 = s.narrow(1, 0, self.sizeState + self.sizeAction)
        o2 = self.modelEncoder(s)

        if type(o2) is type(torch.Tensor([])):
            assert not torch.isnan(o2).any()
        elif type(o2) is type(np.array([])):
            assert not np.isnan(o2).any()

        o = torch.cat((o1, o2), dim=1)

        # action = self.modelPolicy(o1)
        action = self.modelPolicy(o)

        if type(action) is type(torch.Tensor([])):
            assert not torch.isnan(action).any()
        elif type(action) is type(np.array([])):
            assert not np.isnan(action).any()

        return action, state

