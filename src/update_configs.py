import argparse
import json
import os
import sys

absolute_path = os.path.abspath(__file__)
sys.path.append("/".join(os.path.dirname(absolute_path).split("/")[:-1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', default="faster_rcnn", type=str, help='Model name')
    parser.add_argument('-p', '--load-epoch', default="1", type=str, help='Load epoch number')
    parser.add_argument('-t', '--train-data', default="sard", type=str, help='Train Dataset name')
    parser.add_argument('-d', '--test-data', default="sard", type=str, help='Test Dataset name')
    parser.add_argument('-a', '--activate', default="false", type=str, help='Activate dataset preparation with folds')
    parser.add_argument('-f', '--fold', default=1, type=int, help='Number of fold')
    parser.add_argument('-r', '--iteration', default=1, type=int, help='Number of iteration')
    parser.add_argument('-e', '--max-epochs', default=200, type=int, help='Max epochs')
    parser.add_argument('-l', '--lr', default=1e-2, type=float, help='Learning Rate')
    parser.add_argument('-o', '--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('-i', '--interval', default=20, type=int, help='Interval of checkpoint')
    parser.add_argument('-w', '--workers', default=1, type=int, help='Number of workers')
    args = vars(parser.parse_args())

    with open("resource/pipeline.json", "r") as jsonFile:
        data = json.load(jsonFile)

    with open("resource/models.json", "r") as jsonFile:
        data_models = json.load(jsonFile)

    data["iteration"] = args["iteration"]
    data["models"] = args["model_name"].split(",")
    data["workers"] = args["workers"]
    data["load_epoch"] = [args["load-epoch"]]
    data["datasets"]["dataset_train"] = args["train_data"]
    data["datasets"]["dataset_test"] = args["test_data"]
    data["datasets"]["fold"] = f"""fold{args["fold"]}"""

    data["priority_steps"][0]["dataset_train"] = args["train_data"]
    data["priority_steps"][0]["dataset_test"] = args["test_data"]
    data["priority_steps"][0]["activate"] = True if args["activate"] == "true" else False

    for idx, models in enumerate(data_models["models"]):
        if f"""{models["name"]}{models["version"]}""" == args["model_name"]:
            data_models["models"][idx]["runner"]["max_epochs"] = args["max_epochs"]
            data_models["models"][idx]["optimizer"]["lr"] = args["lr"]
            data_models["models"][idx]["optimizer"]["momentum"] = args["momentum"]
            data_models["models"][idx]["optimizer"]["interval"] = args["interval"]

    with open("resource/pipeline.json", "w") as jsonFile:
        json.dump(data, jsonFile)

    with open("resource/models.json", "w") as jsonFile:
        json.dump(data_models, jsonFile)
