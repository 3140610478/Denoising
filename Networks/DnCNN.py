import os
import sys
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config


class ConvBlock(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)
        self.norm = nn.BatchNorm2d(self.out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = super().forward(input)
        y = self.norm(y)
        return F.relu(y)


class DnCNN(nn.Sequential):
    def __init__(self, in_channels=3, layers=config.layers) -> None:
        super().__init__(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(),
            *(ConvBlock(64, 64, 3, 1, 1, bias=False) for _ in range(layers-2)),
            nn.Conv2d(64, in_channels, 3, 1, 1, bias=False),
        )


if __name__ == "__main__":
    dncnn = DnCNN(3, config.layers).to("cuda")
    print(str(dncnn))
    a = torch.zeros((16, 3, 120, 90)).to("cuda")
    while True:
        print(a.shape, dncnn.forward(a).shape)
    pass
