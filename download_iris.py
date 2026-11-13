"""Download scikit-learn/iris from Hugging Face and save it locally."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from datasets import load_dataset


def main():
    ds = load_dataset("scikit-learn/iris")
    output = ROOT / "data"
    output.mkdir(exist_ok=True)
    ds.save_to_disk(str(output / "iris"))
    ds["train"].to_csv(str(output / "iris.csv"), index=False)
    print(ds)
    print(f"Saved dataset: {output / 'iris'}")
    print(f"Saved CSV: {output / 'iris.csv'}")


if __name__ == "__main__":
    main()
