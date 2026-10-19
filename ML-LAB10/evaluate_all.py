import pandas as pd
from ultralytics import YOLO

from lab_config import ROOT, DATA_YAML, LAB_DIR, EXPERIMENTS, DEVICE, weights_path

OUT_CSV = LAB_DIR / "metrics_summary.csv"


def pick_split():
   
    labels = ROOT / "dataset" / "test" / "labels"
    if labels.exists() and any(labels.iterdir()):
        return "test"
    print("[WARN] ไม่พบ label ของ test set -> ใช้ val แทน")
    return "val"


def main():
    split = pick_split()
    rows = []

    for cfg in EXPERIMENTS:
        w = weights_path(cfg["name"])
        if not w.exists():
            print(f"[SKIP] {cfg['name']}: ไม่พบ {w}")
            continue

        print(f"\n===== Evaluating {cfg['name']} on {split} =====")
        model = YOLO(str(w))
        m = model.val(
            data=str(DATA_YAML),
            split=split,
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            device=DEVICE,
            workers=0,
            plots=True,
            project=str(LAB_DIR / "eval"),
            name=cfg["name"],
            exist_ok=True,
        )

        
        csv = LAB_DIR / cfg["name"] / "results.csv"
        last = {}
        if csv.exists():
            df = pd.read_csv(csv)
            df.columns = df.columns.str.strip()
            last = df.iloc[-1].to_dict()

        rows.append({
            "experiment": cfg["name"],
            "epochs": cfg["epochs"],
            "imgsz": cfg["imgsz"],
            "batch": cfg["batch"],
            "split": split,
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
            "mAP50": round(float(m.box.map50), 4),
            "mAP50-95": round(float(m.box.map), 4),
            "train_box_loss": round(last.get("train/box_loss", float("nan")), 4),
            "train_cls_loss": round(last.get("train/cls_loss", float("nan")), 4),
            "val_box_loss": round(last.get("val/box_loss", float("nan")), 4),
            "val_cls_loss": round(last.get("val/cls_loss", float("nan")), 4),
        })

    if not rows:
        print("ยังไม่มีโมเดลที่เทรนแล้ว: รัน train_experiments.py ก่อน")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("\n===== Summary =====")
    print(df.to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()