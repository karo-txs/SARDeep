from src.interfaces.quantization import Quantization
from dataclasses import dataclass
import torch


@dataclass
class PytorchDynamicQuantization(Quantization):

    def __post_init__(self):
        self.base_path = "pytorch/dynamic"
        self.make_dirs()

    def quantize(self):
        self.quantized_model = torch.quantization.quantize_dynamic(self.model,
                                                                   qconfig_spec={torch.nn.Linear},
                                                                   dtype=torch.qint8)

        torch.save(self.quantized_model, f"{self.model_path}/quantization/{self.base_path}/quantized_model.pt")
        self.quantized_model = torch.load(f"{self.model_path}/quantization/{self.base_path}/quantized_model.pt")
        self.upload_config()
