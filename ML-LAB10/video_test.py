import sys
import cv2
from ultralytics import YOLO

from lab_config import ROOT, LAB_DIR, DEVICE, weights_path

# เลือก experiment ที่ดีที่สุดจากผล evaluate_all.py
BEST_EXPERIMENT = "exp2_e10_img480_b8"
IMGSZ = 480


def main():
    # ใช้งาน: python video_test.py path/to/video.mp4
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python video_test.py <video_path>")

    video_path = sys.argv[1]
    weights = weights_path(BEST_EXPERIMENT)
    if not weights.exists():
        raise FileNotFoundError(f"Model not found: {weights}")

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = LAB_DIR / "video_result.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                source=frame, conf=0.25, imgsz=IMGSZ, device=DEVICE, verbose=False
            )
            # plot() วาดกรอบ + ชื่อ class + confidence
            annotated = results[0].plot()
            writer.write(annotated)

            cv2.imshow("YOLO Video Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()