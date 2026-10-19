import sys
from ultralytics import YOLO

from lab_config import DATA_YAML, LAB_DIR, EXPERIMENTS, FRACTION, DEVICE, weights_path


def main():
    # ใช้งาน: python train_experiments.py            -> เทรนทุก experiment
    #         python train_experiments.py exp1 exp3  -> เทรนเฉพาะที่ชื่อขึ้นต้นด้วยคำที่ระบุ
    selected = sys.argv[1:]
    exps = [
        e for e in EXPERIMENTS
        if not selected or any(e["name"].startswith(s) for s in selected)
    ]

    for cfg in exps:
        print(f"\n===== Training {cfg['name']} =====")
        print(f"epochs={cfg['epochs']} imgsz={cfg['imgsz']} batch={cfg['batch']}")

        # โหลด pretrained ใหม่ทุกครั้ง เพื่อให้ทุก experiment เริ่มจากจุดเดียวกัน
        model = YOLO("yolov8n.pt")

        model.train(
            data=str(DATA_YAML),
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            fraction=FRACTION,
            device=DEVICE,
            workers=0,
            amp=True,
            cache=False,
            project=str(LAB_DIR),   # path เต็ม ป้องกันโฟลเดอร์ runs/detect ซ้อน
            name=cfg["name"],
            exist_ok=True,
            save=True,
            plots=True,
            seed=0,
        )

        print(f"Best Model: {weights_path(cfg['name'])}")
        print(f"Training CSV: {LAB_DIR / cfg['name'] / 'results.csv'}")

    print("\nAll training completed!")


if __name__ == "__main__":
    main()