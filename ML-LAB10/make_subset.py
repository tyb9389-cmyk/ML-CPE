
import random
import shutil

from lab_config import ROOT

SRC = ROOT / "dataset"
DST = ROOT / "dataset_small"
SIZES = {"val": 1000, "test": 500}   
random.seed(0)

EXTS = {".jpg", ".jpeg", ".png"}


def main():
    for split, n in SIZES.items():
        img_dir = SRC / split / "images"
        lbl_dir = SRC / split / "labels"
        if not img_dir.exists():
            print(f"[SKIP] ไม่พบ {img_dir}")
            continue

   
        pairs = [
            (p, lbl_dir / f"{p.stem}.txt")
            for p in img_dir.iterdir()
            if p.suffix.lower() in EXTS and (lbl_dir / f"{p.stem}.txt").exists()
        ]
        pairs = random.sample(pairs, min(n, len(pairs)))

        out_img = DST / split / "images"
        out_lbl = DST / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for img, lbl in pairs:
            shutil.copy2(img, out_img / img.name)
            shutil.copy2(lbl, out_lbl / lbl.name)
        print(f"{split}: copied {len(pairs)} images")

    yaml = f"""path: {DST.as_posix()}
train: {(SRC / 'train' / 'images').as_posix()}
val: val/images
test: test/images

nc: 10
names:
  - person
  - rider
  - car
  - bus
  - truck
  - bike
  - motor
  - traffic light
  - traffic sign
  - train
"""
    (DST / "data_small.yaml").write_text(yaml, encoding="utf-8")
    print(f"Saved: {DST / 'data_small.yaml'}")


if __name__ == "__main__":
    main()
