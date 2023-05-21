from src.infra.configs.config import Configuration
from dataclasses import dataclass, field
from src.interfaces.step import Step
from dotenv import load_dotenv
import pandas as pd
import json
import ast
import os

load_dotenv()


@dataclass
class Evaluation(Step):
    model: dict = field(default=None)
    results_dir: list = field(default_factory=lambda: [])
    device: str = field(default="cpu")

    def __post_init__(self):
        config = Configuration(self.model)
        self.device = config.device
        self.work_dir = config.cfg.work_dir

        data_test = config.base_file["datasets"]["paths"][config.base_file["datasets"]["dataset_type"]]["test"][
            "name"]

        base_results = f"""{config.cfg.work_dir}/test_{self.device}/{data_test}"""
        self.results_dir.append(base_results)

        if os.path.isdir(f"{base_results}/quantization"):
            if os.path.isdir(f"{base_results}/quantization/pytorch/dynamic"):
                self.results_dir.append(f"{base_results}/quantization/pytorch/dynamic")

            if os.path.isdir(f"{base_results}/quantization/neural_compressor/dynamic"):
                self.results_dir.append(f"{base_results}/quantization/neural_compressor/dynamic")

    def run_step(self):

        for result_dir in self.results_dir:

            output_dir = f"""{os.getenv("WORK_DIR")}/evaluate"""

            with open(f"{result_dir}/eval_voc.json", "r") as jsonFile:
                eval_voc = json.load(jsonFile)

            with open(f"{result_dir}/eval_coco.json", "r") as jsonFile:
                eval_coco = json.load(jsonFile)

            with open(f"{result_dir}/config.json", "r") as jsonFile:
                config = json.load(jsonFile)

            eval_dict = self.merge_dicts(eval_voc["metric"], eval_coco["metric"])
            eval_dict = self.merge_dicts(eval_dict, eval_voc["timer"])
            eval_dict = self.merge_dicts(eval_dict, config)

            eval_dict["model"] = f"""{self.model["name"]}{self.model["version"]}"""
            eval_dict["device"] = self.device
            eval_dict["dataset_train"] = self.model["datasets"]["name"]
            eval_dict["dataset_train_fold"] = int(self.model["datasets"]["fold"].replace("fold", ""))
            eval_dict["dataset_test"] = self.model["datasets"]["paths"]["voc"]["test"]["name"]
            eval_dct = {k: [v] for k, v in eval_dict.items()}

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            df = pd.DataFrame.from_dict(eval_dct)
            if os.path.isfile(f"{output_dir}/metric_results.csv"):
                df.to_csv(f"{output_dir}/metric_results.csv", mode='a', index=False, header=False)
            else:
                df.to_csv(f"{output_dir}/metric_results.csv", mode='a', index=False, header=True)

            if "quantization" not in result_dir:
                train_epochs = ""
                for root, dirs, files in os.walk(f"{self.work_dir}"):
                    for file in files:
                        if "log" in file and file.endswith('.log'):
                            train_epochs = os.path.join(root, file)

                file = open(train_epochs, 'r')
                lines = file.read().splitlines()
                file.close()

                train_dict = dict(
                    model=self.model["name"],
                    dataset_train=self.model["datasets"]["name"],
                    dataset_train_fold=int(self.model["datasets"]["fold"].replace("fold", "")),
                    dataset_test=self.model["datasets"]["paths"]["voc"]["test"]["name"]
                )

                for line in lines:
                    if "mmdet" in line and "Epoch" in line and "val" not in line:
                        epoch_batch = line.split("\t")[0].split(" ")[-1]

                        result_line = line.split("\t")[1].split(", ")
                        del result_line[1]

                        for index, value in enumerate(result_line):
                            key_value = value.split(": ")

                            result_line[index] = f"\"{key_value[0]}\": {key_value[1]}"

                        result_line = ", ".join(result_line)

                        result_str = ''.join(('{', result_line, '}'))

                        results = ast.literal_eval(result_str)
                        results["epoch_batch"] = epoch_batch

                        results_final = dict(epoch_batch=results["epoch_batch"],
                                             lr=results["lr"],
                                             memory=results["memory"],
                                             loss=results["loss"])

                        final_dict = self.merge_dicts(train_dict, results_final)
                        final_dict = {k: [v] for k, v in final_dict.items()}
                        df = pd.DataFrame.from_dict(final_dict)

                        if os.path.isfile(f"{output_dir}/epoch_results.csv"):
                            df.to_csv(f"{output_dir}/epoch_results.csv", mode='a', index=False, header=False)
                        else:
                            df.to_csv(f"{output_dir}/epoch_results.csv", mode='a', index=False, header=True)

    def merge_dicts(self, *dict_args):
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result
