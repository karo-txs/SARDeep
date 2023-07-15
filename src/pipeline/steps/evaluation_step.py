from src.infra.configs.config import Configuration
from dataclasses import dataclass, field
from src.interfaces.step import Step
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import json
import os

load_dotenv()


@dataclass
class Evaluation(Step):
    model: dict = field(default=None)
    load_epoch: str = field(default="latest")
    results_dir: list = field(default_factory=lambda: [])
    models_dir: list = field(default_factory=lambda: [])
    device: str = field(default="cpu")

    def __post_init__(self):
        config = Configuration(self.model)
        self.device = config.device
        self.work_dir = config.cfg.work_dir

        data_test = config.base_file["datasets"]["paths"][config.base_file["datasets"]["dataset_type"]]["test"]["name"]

        base_results = f"""{config.cfg.work_dir}/test_{self.device}/{data_test}"""
        self.results_dir.append(base_results)

        model_pth = ""
        for i in os.listdir(config.cfg.work_dir):
            if i.endswith(".pth"):
                model_pth = f"{config.cfg.work_dir}/{i}"
                break

        self.models_dir.append(model_pth)

        if os.path.isdir(f"{base_results}/quantization"):
            if os.path.isdir(f"{base_results}/quantization/pytorch/dynamic"):
                self.results_dir.append(f"{base_results}/quantization/pytorch/dynamic")
                self.models_dir.append(f"""{config.cfg.work_dir}/quantization/pytorch/dynamic/quantized_model.pt""")

    def run_step(self):

        for result_dir, model_dir in zip(self.results_dir, self.models_dir):
            output_dir = f"""{os.getenv("WORK_DIR")}/evaluate"""

            with open(f"{result_dir}/eval_voc.json", "r") as jsonFile:
                eval_voc = json.load(jsonFile)

            with open(f"{result_dir}/eval_coco.json", "r") as jsonFile:
                eval_coco = json.load(jsonFile)

            with open(f"{result_dir}/config.json", "r") as jsonFile:
                config = json.load(jsonFile)

            voc_metrics = dict(AP25=eval_voc["metric"]["iou_thr_0.25"]["AP25"],
                               AP50=eval_voc["metric"]["iou_thr_0.5"]["AP50"],
                               AP75=eval_voc["metric"]["iou_thr_0.75"]["AP75"],
                               params=eval_voc["params"])

            eval_dict = self.merge_dicts(voc_metrics, eval_coco["metric"])
            eval_dict = self.merge_dicts(eval_dict, eval_voc["timer"])
            eval_dict = self.merge_dicts(eval_dict, config)

            eval_dict["model"] = f"""{self.model["name"]}{self.model["version"]}"""
            eval_dict["device"] = self.device
            eval_dict["dataset_train"] = self.model["datasets"]["name"]
            eval_dict["dataset_train_fold"] = int(self.model["datasets"]["fold"].replace("fold", ""))
            eval_dict["dataset_test"] = self.model["datasets"]["paths"]["voc"]["test"]["name"]
            eval_dict["size"] = Path(model_dir).stat().st_size / 1000000
            eval_dct = {k: [v] for k, v in eval_dict.items()}

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            df = pd.DataFrame.from_dict(eval_dct)
            if os.path.isfile(f"{output_dir}/metric_results.csv"):
                df.to_csv(f"{output_dir}/metric_results.csv", mode='a', index=False, header=False)
            else:
                df.to_csv(f"{output_dir}/metric_results.csv", mode='a', index=False, header=True)

            self.calculate_means()

    def merge_dicts(self, *dict_args):
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    def calculate_means(self):
        output_dir = f"""{os.getenv("WORK_DIR")}/evaluate"""
        df = pd.read_csv(f"{output_dir}/metric_results.csv")

        models = df["model"].unique().tolist()

        df_general = {"model": [], "iteration": [], "AP25": [], "AP50": [], "AP75": [],
                      "APs": [], "APm": [], "APl": [], "mean_syn": [],
                      "std_syn": [], "throughput": [], "is_quantized": [],
                      "approach": [], "size": [], "device": [], "dataset_train": [],
                      "dataset_test": []}

        devices = ["cpu", "cuda"]
        is_quantized_op = [True, False]
        approachs = ["pytorch/dynamic"]
        datasets = ["sard", "heridal"]
        iterations = df["iteration"].unique().tolist()

        for i in iterations:
            for model in models:
                for device in devices:
                    for dataset in datasets:
                        for is_quantized in is_quantized_op:
                            try:
                                if is_quantized is False:
                                    model_map = df.loc[(df.model == model) & (df.device == device)
                                                       & (df.dataset_test == dataset)
                                                       & (df.iteration == i)].query('is_quantized == False')
                                    if not model_map.empty:
                                        self.append_values(df_general, model, model_map, is_quantized=False,
                                                           approach=None,
                                                           device=device, dataset_test=dataset, dataset_train=dataset,
                                                           iteration=i)
                                else:
                                    for approach in approachs:
                                        model_map = df.loc[(df.model == model) & (df.device == device)
                                                           & (df.dataset_test == dataset)
                                                           & (df.approach == approach)
                                                           & (df.iteration == i)].query('is_quantized == True')
                                        if not model_map.empty:
                                            self.append_values(df_general, model, model_map, is_quantized=True,
                                                               approach=approach,
                                                               device=device, dataset_test=dataset,
                                                               dataset_train=dataset, iteration=i)
                            except:
                                print(f"Results Not Found:", model, device, dataset, is_quantized)

        df = pd.DataFrame.from_dict(df_general)

        if os.path.isfile(f"{output_dir}/general_results.csv"):
            os.remove(f"{output_dir}/general_results.csv")

        df.to_csv(f"{output_dir}/general_results.csv", mode='a', index=False, header=True)

    def append_values(self, dict_general: dict, model: str, df: pd.DataFrame, is_quantized: bool, approach: any,
                      device: str, dataset_train: str, dataset_test: str, iteration: int):
        dict_general["model"].append(model)
        dict_general["iteration"].append(iteration)
        dict_general["APs"].append(df.loc[:, 'bbox_mAP_s'].mean())
        dict_general["APm"].append(df.loc[:, 'bbox_mAP_m'].mean())
        dict_general["APl"].append(df.loc[:, 'bbox_mAP_l'].mean())
        dict_general["AP25"].append(df.loc[:, 'AP25'].mean())
        dict_general["AP50"].append(df.loc[:, 'AP50'].mean())
        dict_general["AP75"].append(df.loc[:, 'AP75'].mean())
        dict_general["mean_syn"].append(df.loc[:, 'mean_syn'].mean())
        dict_general["std_syn"].append(df.loc[:, 'std_syn'].mean())
        dict_general["throughput"].append(df.loc[:, 'throughput'].mean())
        dict_general["is_quantized"].append(is_quantized)
        dict_general["approach"].append(approach)
        dict_general["size"].append(df.loc[:, 'size'].mean())
        dict_general["device"].append(device)
        dict_general["dataset_train"].append(dataset_train)
        dict_general["dataset_test"].append(dataset_test)
