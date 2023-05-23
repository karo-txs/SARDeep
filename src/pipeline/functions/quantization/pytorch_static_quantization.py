from src.interfaces.quantization import Quantization
from dataclasses import dataclass
import torch


@dataclass
class PytorchStaticQuantization(Quantization):

    def __post_init__(self):
        self.base_path = "pytorch/static"
        self.quantized_path = f"{self.model_path}/quantization/{self.base_path}/"
        self.model_save_name = "quantized_model.pt"
        self.make_dirs()

    def quantize(self):
        self.quantized_model = self.model

        self.quantized_model.qconfig = torch.ao.quantization.default_qconfig
        torch.ao.quantization.prepare(self.quantized_model, inplace=True)

        self.test_quantized_model(eval_types=["voc"], export_results=False)

        torch.ao.quantization.convert(self.quantized_model, inplace=True)

        torch.save(self.quantized_model, f"{self.quantized_path}{self.model_save_name}")
        self.upload_config()
