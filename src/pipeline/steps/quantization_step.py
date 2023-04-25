from torch.utils.data import DataLoader
from src.interfaces.step import Step
from dataclasses import dataclass, field
import torch.nn as nn


@dataclass
class Quantization(Step):
    model: nn.Module = field(default=None)
    dataloader: DataLoader = field(default=None)

    def run_step(self):
       pass