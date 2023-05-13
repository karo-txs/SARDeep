from sklearn.model_selection import KFold
from dataclasses import dataclass, field
from src.interfaces.step import Step
from dotenv import load_dotenv
import json
import os

load_dotenv()


@dataclass
class DatasetPreparation(Step):
    dataset_train: str = field(default="sard")
    dataset_test: str = field(default="sard")
    n_splits: int = field(default=5)

    def cross_validation(self):
        data_root = os.getenv("DATA_ROOT")
        resource_path = os.getenv("RESOURCE_PATH")

        with open(f"{resource_path}/datasets.json", "r") as jsonFile:
            data = json.load(jsonFile)

        data["datasets"] = []

        data_path = f"""{data_root}/{self.dataset_train}/VOC2012/ImageSets/Main"""

        train_file = open(f"{data_path}/train.txt", 'r')
        val_file = open(f"{data_path}/val.txt", 'r')

        train_data = [line for line in train_file.readlines()]
        val_data = [line for line in val_file.readlines()]
        train_data.extend(val_data)

        kf = KFold(n_splits=self.n_splits)
        splits = []

        for i, (train_index, test_index) in enumerate(kf.split(train_data)):
            val_fold = [(train_data[i] if "\n" in train_data[i] else f"{train_data[i]}\n") for i in test_index]
            train_fold = [(train_data[i] if "\n" in train_data[i] else f"{train_data[i]}\n") for i in train_index]

            splits.append(dict(split=(i + 1), val=val_fold, train=train_fold))

        for split in splits:
            with open(f"""{data_path}/val_fold{split["split"]}.txt""", 'w') as f:
                for value in split["val"]:
                    f.write(value)

            with open(f"""{data_path}/train_fold{split["split"]}.txt""", 'w') as f:
                for value in split["train"]:
                    f.write(value)

            data["datasets"].append(self.get_dataset_info(data_root, split["split"]))

        with open(f"{resource_path}/datasets.json", "w") as jsonFile:
            json.dump(data, jsonFile)

    def get_dataset_info(self, data_root: str, split: int) -> dict:
        data_path_train = f"""{data_root}/{self.dataset_train}/VOC2012"""
        data_path_test = f"""{data_root}/{self.dataset_test}/VOC2012"""

        return {
            "name": self.dataset_train,
            "fold": f"fold{split}",
            "paths": {
                "train": {
                    "name": self.dataset_train,
                    "ann_file": f"{data_path_train}/ImageSets/Main/train_fold{split}.txt",
                    "img_prefix": data_path_train
                },
                "val": {
                    "name": self.dataset_train,
                    "ann_file": f"{data_path_train}/ImageSets/Main/val_fold{split}.txt",
                    "img_prefix": data_path_train
                },
                "test": {
                    "name": self.dataset_test,
                    "ann_file": f"{data_path_test}/ImageSets/Main/test.txt",
                    "img_prefix": data_path_test
                }
            }
        }

    def _chunks(self, lst: list, n: int):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def run_step(self):
        self.cross_validation()
