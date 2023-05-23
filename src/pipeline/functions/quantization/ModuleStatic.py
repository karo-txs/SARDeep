from torch.ao.quantization import QuantStub, DeQuantStub
import torch
from torch.testing._internal.common_quantization import ConvBNReLU


class ModuleStatic(torch.nn.Module):
    def __init__(self, model=torch.nn.Module):
        super(ModuleStatic, self).__init__()
        self.model = model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, imgs, img_metas, proposals):
        imgs = self.quant(imgs)
        x = self.model(return_loss=False, rescale=True, imgs=imgs, img_metas=img_metas, proposals=proposals)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.ao.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)