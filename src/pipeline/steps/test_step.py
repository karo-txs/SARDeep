from src.infra.configs.config import Configuration
from mmdet.core import encode_mask_results
from dataclasses import dataclass, field
from src.pipeline.functions import *
from src.interfaces.step import Step
from mmdet.utils import build_dp
from mmcv import tensor2imgs
import os.path as osp
import shutil
import torch
import mmcv
import os


@dataclass
class Test(Step):
    model: dict = field(default=None)
    eval_metrics: list = field(default_factory=lambda: ["voc"])
    show: bool = field(default=True)
    show_score_thr: float = field(default=0.3)

    def run_step(self):

        config = Configuration(self.model)
        for eval_type in self.eval_metrics:
            cfg = config.load_config_for_test(eval_type)

            data_test = config.base_file["datasets"]["paths"][config.base_file["datasets"]["dataset_type"]]["test"][
                "name"]
            show_dir = f"""{cfg.work_dir}/test/{data_test}"""
            out = f"""{cfg.work_dir}/test/{data_test}/results_{eval_type}.pkl"""

            loader = Loader(cfg)
            dataset, data_loader = loader.load_dataset()

            mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

            model = loader.load_model()
            model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
            outputs = self.single_gpu_test(model, data_loader, self.show, show_dir, self.show_score_thr)

            print(f'\nwriting results to {out}')
            mmcv.dump(outputs, out)

            json_file = osp.join(f"{cfg.work_dir}/test/{data_test}", f'eval_{eval_type}.json')

            eval_kwargs = cfg.get('evaluation', {}).copy()

            for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)

            if eval_type == "coco":
                self.move_images(show_dir)

                eval_kwargs.update(dict(metric="bbox",
                                        iou_thrs=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                                        metric_items=['mAP', 'mAP_50', 'mAP_75',
                                                      'mAP_s', 'mAP_m', 'mAP_l']))
            else:
                eval_kwargs.update(dict(metric="mAP"))

            metric = dataset.evaluate(outputs, **eval_kwargs)
            metric_dict = dict(config=config.config_file, metric=metric)
            mmcv.dump(metric_dict, json_file)

    def move_images(self, show_dir: str):
        if not os.path.exists(f"{show_dir}/JPEGImagesCOCO"):
            os.makedirs(f"{show_dir}/JPEGImagesCOCO")

        source = os.listdir(show_dir)
        for files in source:
            if files.endswith('.jpg'):
                shutil.move(f"{show_dir}/{files}", f"{show_dir}/JPEGImagesCOCO/{files}")

    def single_gpu_test(self, model,
                        data_loader,
                        show=False,
                        out_dir=None,
                        show_score_thr=0.3):
        PALETTE_BBOX = ((2, 144, 240))
        PALETTE_TEXT = ((255, 255, 255))
        model.eval()
        results = []
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

            batch_size = len(result)
            if show or out_dir:
                if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        bbox_color=PALETTE_BBOX,
                        text_color=PALETTE_TEXT,
                        mask_color=PALETTE_BBOX,
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)

            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (bbox_results,
                                                encode_mask_results(mask_results))

            results.extend(result)

            for _ in range(batch_size):
                prog_bar.update()
        return results
