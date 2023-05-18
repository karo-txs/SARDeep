from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor import quantization
from src.interfaces.quantization import Quantization
from dataclasses import dataclass


@dataclass
class NeuralCompressorStaticQuantization(Quantization):

    def quantize(self):
        conf = PostTrainingQuantConfig(approach="dynamic")
        q_model = quantization.fit(model=self.model,
                                   conf=conf,
                                   calib_dataloader=self.dataloader)
        q_model.save('./output')
