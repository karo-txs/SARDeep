from src.infra.scripts.voc_to_coco import voc_to_coco
from sklearn.model_selection import KFold
from dataclasses import dataclass, field
from src.interfaces.step import Step
from dotenv import load_dotenv
import shutil
import json
import os

load_dotenv()


@dataclass
class DatasetPreparation(Step):
    dataset_train: str = field(default="sard")
    dataset_test: str = field(default="sard")
    dataset_type: str = field(default="voc")
    converter_coco: bool = field(default=True)
    labels: list = field(default_factory=lambda: ["person"])
    n_splits: int = field(default=5)

    def cross_validation(self):
        data_root = os.getenv("DATA_ROOT")
        resource_path = os.getenv("RESOURCE_PATH")

        with open(f"{resource_path}/datasets.json", "r") as jsonFile:
            data = json.load(jsonFile)

        data["datasets"] = []

        data_path = f"""{data_root}/{self.dataset_train}/VOC2012/ImageSets/Main"""

        # Copy to coco
        if not os.path.isdir(f"""{data_root }/{self.dataset_train}/coco/images""") and self.converter_coco:
            os.makedirs(f"""{data_root}/{self.dataset_train}/coco/annotations""")
            shutil.copytree(f"""{data_root}/{self.dataset_train}/VOC2012/JPEGImages/""", f"""{data_root
            }/{self.dataset_train}/coco/images""")

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

            # VOC to COCO Format
            if self.converter_coco:
                voc_to_coco(ann_dir=f"{data_root}/{self.dataset_train}/VOC2012/Annotations",
                            ann_ids=f"""{data_path}/train_fold{split["split"]}.txt""",
                            labels=self.labels,
                            output=f"""{data_root}/{self.dataset_train}/coco/annotations/instances_train_fold{split["split"]}.json""")

                voc_to_coco(ann_dir=f"{data_root}/{self.dataset_train}/VOC2012/Annotations",
                            ann_ids=f"""{data_path}/val_fold{split["split"]}.txt""",
                            labels=self.labels,
                            output=f"""{data_root}/{self.dataset_train}/coco/annotations/instances_val_fold{split["split"]}.json""")

        if self.converter_coco:
            if not os.path.isdir(f"""{data_root}/{self.dataset_test}/coco/images"""):
                shutil.copytree(f"""{data_root}/{self.dataset_test}/VOC2012/JPEGImages/""", f"""{data_root
                }/{self.dataset_test}/coco/images""")
                os.makedirs(f"""{data_root}/{self.dataset_test}/coco/annotations""")

            voc_to_coco(ann_dir=f"{data_root}/{self.dataset_test}/VOC2012/Annotations",
                        ann_ids=f"""{data_root}/{self.dataset_test}/VOC2012/ImageSets/Main/test.txt""",
                        labels=self.labels,
                        output=f"""{data_root}/{self.dataset_test}/coco/annotations/instances_test.json""")

        with open(f"{resource_path}/datasets.json", "w") as jsonFile:
            json.dump(data, jsonFile)

    def get_dataset_info(self, data_root: str, split: int) -> dict:
        data_path_train = f"""{data_root}/{self.dataset_train}"""
        data_path_test = f"""{data_root}/{self.dataset_test}"""

        return {
            "name": self.dataset_train,
            "fold": f"fold{split}",
            "dataset_type": self.dataset_type,
            "paths": {
                "voc": {
                    "train": {
                        "name": self.dataset_train,
                        "ann_file": f"{data_path_train}/VOC2012/ImageSets/Main/train_fold{split}.txt",
                        "img_prefix": f"{data_path_train}/VOC2012"
                    },
                    "val": {
                        "name": self.dataset_train,
                        "ann_file": f"{data_path_train}/VOC2012/ImageSets/Main/val_fold{split}.txt",
                        "img_prefix": f"{data_path_train}/VOC2012"
                    },
                    "test": {
                        "name": self.dataset_test,
                        "ann_file": f"{data_path_test}/VOC2012/ImageSets/Main/test.txt",
                        "img_prefix": f"{data_path_test}/VOC2012"
                    }
                },
                "coco": {
                    "train": {
                        "name": self.dataset_train,
                        "ann_file": f"{data_path_train}/coco/annotations/instances_train_fold{split}.json",
                        "img_prefix": f"{data_path_train}/coco/images"
                    },
                    "val": {
                        "name": self.dataset_train,
                        "ann_file": f"{data_path_train}/coco/annotations/instances_val_fold{split}.json",
                        "img_prefix": f"{data_path_train}/coco/images"
                    },
                    "test": {
                        "name": self.dataset_test,
                        "ann_file": f"{data_path_test}/coco/annotations/instances_test.json",
                        "img_prefix": f"{data_path_test}/coco/images"
                    }
                }
            }
        }

    def run_step(self):
        self.cross_validation()
