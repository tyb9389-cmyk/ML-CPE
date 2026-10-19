from pathlib import Path

import torch


DEVICE = 0 if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parent
_small = ROOT / "dataset_small" / "data_small.yaml"

DATA_YAML = _small if _small.exists() else ROOT / "dataset" / "data.yaml"
LAB_DIR = ROOT / "runs" / "lab10"


FRACTION = 0.02


EXPERIMENTS = [
    {"name": "exp1_e5_img480_b8",  "epochs": 5,  "imgsz": 480, "batch": 8},
    {"name": "exp2_e10_img480_b8", "epochs": 10, "imgsz": 480, "batch": 8},
    {"name": "exp3_e20_img480_b8", "epochs": 20, "imgsz": 480, "batch": 8},
    {"name": "exp4_e10_img640_b8", "epochs": 10, "imgsz": 640, "batch": 8},
    {"name": "exp5_e10_img480_b16", "epochs": 10, "imgsz": 480, "batch": 16},
]


def weights_path(name: str, kind: str = "best") -> Path:
    return LAB_DIR / name / "weights" / f"{kind}.pt"