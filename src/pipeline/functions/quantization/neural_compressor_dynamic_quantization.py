from neural_compressor.config import PostTrainingQuantConfig
from src.interfaces.quantization import Quantization
from neural_compressor import quantization
from dataclasses import dataclass
import torch


@dataclass
class NeuralCompressorDynamicQuantization(Quantization):

    def __post_init__(self):
        self.base_path = "neural_compressor/dynamic"
        self.make_dirs()

    def quantize(self):
        conf = PostTrainingQuantConfig(approach="dynamic")
        self.quantized_model = quantization.fit(model=self.model,
                                                conf=conf,
                                                calib_dataloader=self.dataloader,
                                                eval_dataloader=self.dataloader)
        torch.save(self.quantized_model, f"{self.model_path}/quantization/{self.base_path}/quantized_model.pt")
        self.upload_config()
