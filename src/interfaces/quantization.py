from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from torch import nn
import torch


@dataclass
class Quantization(ABC):
    model: nn.Module
    dataloader: torch.utils.data.DataLoader


    @abstractmethod
    def quantize(self):
        pass
