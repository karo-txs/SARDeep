from src.interfaces.quantization import Quantization
import torch.ao.nn.quantized.dynamic as nnqd
from dataclasses import dataclass
import torch


@dataclass
class PytorchDynamicQuantization(Quantization):

    def __post_init__(self):
        self.base_path = "pytorch/dynamic"
        self.quantized_path = f"{self.model_path}/quantization/{self.base_path}/"
        self.model_save_name = "quantized_model.pt"
        self.make_dirs()

    def quantize(self):
        print(torch.__version__)
        self.quantized_model = torch.quantization.quantize_dynamic(self.model,
                                                                   qconfig_spec={torch.nn.Linear, },
                                                                   dtype=torch.qint8,
                                                                   mapping={torch.nn.Linear: nnqd.Linear, })
        torch.save(self.quantized_model, f"{self.quantized_path}{self.model_save_name}")
        self.upload_config()
