from src.infra.configs.config import Configuration
from src.pipeline.utils.loader import Loader
from src.pipeline.utils.timer import Timer
from mmdet.core import encode_mask_results
from dataclasses import dataclass, field
from src.interfaces.step import Step
from mmdet.utils import build_dp
from mmcv import tensor2imgs, Config
import os.path as osp
import shutil
import torch
import json
import mmcv
import os


@dataclass
class Test(Step):
    model: dict = field(default=None)
    load_epoch: str = field(default="latest")
    eval_metrics: list = field(default_factory=lambda: ["voc"])
    show: bool = field(default=True)
    show_score_thr: float = field(default=0.3)
    timer: Timer = field(default=Timer())

    def run_step(self):

        config = Configuration(self.model)
        for eval_type in self.eval_metrics:
            cfg = config.load_config_for_test(eval_type)

            data_test = config.base_file["datasets"]["paths"][config.base_file["datasets"]["dataset_type"]]["test"][
                "name"]
            show_dir = f"""{cfg.work_dir}/test_{config.device}/{data_test}"""
            out = f"""{cfg.work_dir}/test_{config.device}/{data_test}/results_{eval_type}.pkl"""

            loader = Loader(cfg)
            dataset, data_loader = loader.load_dataset()

            mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

            model = loader.load_model(self.load_epoch)
            self.test_model(model, cfg, config, data_loader, dataset, show_dir, out, data_test, eval_type)

            config_info = dict(is_quantized=False, approach=None)
            with open(f"{show_dir}/config.json", "w") as jsonFile:
                json.dump(config_info, jsonFile)

    def test_model(self, model: torch.nn.Module,
                   cfg: Config,
                   config: Configuration,
                   data_loader: any,
                   dataset: any,
                   show_dir: str,
                   out: str,
                   data_test: str,
                   eval_type: str,
                   export_results: bool = True):
        self.timer.device = config.device
        self.timer.batch_size = config.batch_size

        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

        print("\nTest Results")
        outputs, timer_result = self.test_results(model, data_loader, True, show_dir, self.show_score_thr)

        if export_results:
            mmcv.mkdir_or_exist(osp.abspath(show_dir))

            print(f'\nwriting results to {out}')
            mmcv.dump(outputs, out)

        json_file = osp.join(show_dir, f'eval_{eval_type}.json')
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        eval_kwargs = cfg.get('evaluation', {}).copy()

        for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule', 'dynamic_intervals'
        ]:
            eval_kwargs.pop(key, None)
        metric = {}
        if eval_type == "coco":
            self.move_images(show_dir)

            eval_kwargs.update(dict(metric="bbox",
                                    iou_thrs=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                                    metric_items=['mAP', 'mAP_50', 'mAP_75',
                                                  'mAP_s', 'mAP_m', 'mAP_l']))

            metric = dataset.evaluate(outputs, **eval_kwargs)
        else:
            iou_thrs = [0.25, 0.50, 0.75]

            for iou_thr in iou_thrs:
                eval_kwargs.update(dict(metric="mAP", iou_thr=iou_thr))
                metric[f"iou_thr_{iou_thr}"] = dataset.evaluate(outputs, **eval_kwargs)

        metric_dict = dict(config=config.config_file, metric=metric, timer=timer_result, params=params)
        if export_results:
            mmcv.dump(metric_dict, json_file)

    def move_images(self, show_dir: str):
        if not os.path.exists(f"{show_dir}/JPEGImagesCOCO"):
            os.makedirs(f"{show_dir}/JPEGImagesCOCO")

        source = os.listdir(show_dir)
        for files in source:
            if files.endswith('.jpg'):
                shutil.move(f"{show_dir}/{files}", f"{show_dir}/JPEGImagesCOCO/{files}")

    def test_results(self, model,
                     data_loader,
                     show=False,
                     out_dir=None,
                     show_score_thr=0.3):
        PALETTE_BBOX = ((2, 144, 240))
        PALETTE_TEXT = ((255, 255, 255))
        model.eval()
        results = []
        dataset = data_loader.dataset
        count = 0

        # GPU-WARM-UP
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                _ = model(return_loss=False, rescale=True, **data)
                count += 1
                if count == 10:
                    break

        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                self.timer.start()
                result = model(return_loss=False, rescale=True, **data)
                self.timer.finalize()

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
        self.timer.calculate()
        return results, self.timer.result
