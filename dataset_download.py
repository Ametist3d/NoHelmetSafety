# dataset_download.py
import argparse
import shutil
from pathlib import Path
import kagglehub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", required=True, help="Kaggle dataset slug, e.g. vodan37/yolo-helmethead")
    parser.add_argument("--out", required=True, help="Target folder, e.g. DS/datasets")
    args = parser.parse_args()

    print(f"[info] Downloading dataset: {args.ds}")
    downloaded_path = kagglehub.dataset_download(args.ds)

    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)

    print(f"[info] Copying dataset to: {target}")
    shutil.copytree(downloaded_path, target / Path(downloaded_path).name, dirs_exist_ok=True)

    print("[done] Dataset ready.")


if __name__ == "__main__":
    main()