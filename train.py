# train.py
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pps.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_path = Path(cfg["dataset"]["path"])
    data_yaml = dataset_path / cfg["dataset"]["data_yaml"]

    train_cfg = cfg["train"]

    print(f"[info] Using dataset yaml: {data_yaml}")
    print(f"[info] Loading model: {train_cfg['model']}")

    model = YOLO(train_cfg["model"])

    model.train(
        data=str(data_yaml),
        imgsz=train_cfg["imgsz"],
        epochs=train_cfg["epochs"],
        batch=train_cfg["batch"],
        device=train_cfg["device"],
        close_mosaic=train_cfg["close_mosaic"],
        project=train_cfg["project"],
        name=train_cfg["name"],
        exist_ok=train_cfg["exist_ok"],
    )


if __name__ == "__main__":
    main()