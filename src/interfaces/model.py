from tools.deployment.pytorch2onnx import parse_normalize_cfg, pytorch2onnx
from mmdet.core.export.model_wrappers import ONNXRuntimeDetector
from mmcv.onnx.symbolic import register_extra_symbolics
from dataclasses import dataclass, field
from mmdet.utils import build_dp
from mmcv import Config


@dataclass
class Model:
    model: any
    cfg: Config
    is_quantized: bool = field(default=False)
    backend: str = field(default="torch")
    output_file: str = field(default=None)

    def __post_init__(self):
        if self.backend == "torch":
            self.model = build_dp(self.model, self.cfg.device, device_ids=self.cfg.gpu_ids)
            self.model.eval()

        elif self.backend == "onnx":
            register_extra_symbolics(11)
            img_scale = self.cfg.test_pipeline[1]['img_scale']
            input_shape = (1, 3, img_scale[1], img_scale[0])
            normalize_cfg = parse_normalize_cfg(self.cfg.test_pipeline)

            pytorch2onnx(
                self.model,
                "mmdetection/demo/demo.jpg",
                input_shape,
                normalize_cfg,
                opset_version=11,
                show=True,
                output_file=self.output_file,
                verify=True,
                test_img=None,
                do_simplify=False,
                dynamic_export=None,
                skip_postprocess=False)

            self.model = ONNXRuntimeDetector(self.output_file, class_names=self.model.CLASSES, device_id=0)
            self.model.eval()

    def __call__(self, return_loss=False, rescale=True, **data):
        if self.is_quantized:
            self.model = build_dp(self.model, self.cfg.device, device_ids=self.cfg.gpu_ids)
            self.model.eval()

        return self.model(return_loss=return_loss, rescale=rescale, **data)

    def parameters(self):
        return self.model.parameters()

    def show_results(self, img_show, result, i, palette_bbox, palette_text, show, out_file, show_score_thr):
        if not self.is_quantized:
            self.model.module.show_result(
                img_show,
                result[i],
                bbox_color=palette_bbox,
                text_color=palette_text,
                mask_color=palette_bbox,
                show=show,
                out_file=out_file,
                score_thr=show_score_thr)
