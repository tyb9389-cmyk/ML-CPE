import matplotlib.pyplot as plt
import pandas as pd

from lab_config import LAB_DIR, EXPERIMENTS

OUT_DIR = LAB_DIR / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load(name):
    csv = LAB_DIR / name / "results.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    return df


def plot_curves(names, title, filename):
    """กราฟ train/val loss (box, cls, dfl) และ mAP ต่อ epoch ของหลาย experiment"""
    panels = [
        ("train/box_loss", "Train Box Loss"),
        ("val/box_loss", "Val Box Loss"),
        ("train/cls_loss", "Train Cls Loss"),
        ("val/cls_loss", "Val Cls Loss"),
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (col, label) in zip(axes.ravel(), panels):
        for n in names:
            df = load(n)
            if df is not None and col in df:
                ax.plot(df["epoch"], df[col], marker="o", label=n)
        ax.set_title(label)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / filename}")


def plot_metric_bars():
    csv = LAB_DIR / "metrics_summary.csv"
    if not csv.exists():
        print("ไม่พบ metrics_summary.csv: รัน evaluate_all.py ก่อน")
        return
    df = pd.read_csv(csv).set_index("experiment")
    ax = df[["precision", "recall", "mAP50", "mAP50-95"]].plot(
        kind="bar", figsize=(12, 6)
    )
    ax.set_title("Precision / Recall / mAP per experiment")
    ax.set_ylabel("score")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "metrics_bar.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR / 'metrics_bar.png'}")


def main():
    names = [e["name"] for e in EXPERIMENTS]
    by_prefix = lambda *p: [n for n in names if n.split("_")[0] in p]

    plot_curves(by_prefix("exp1", "exp2", "exp3"),
                "Effect of number of epochs (imgsz=480, batch=8)",
                "compare_epochs.png")
    plot_curves(by_prefix("exp2", "exp4"),
                "Effect of image size (epochs=10, batch=8)",
                "compare_imgsz.png")
    plot_curves(by_prefix("exp2", "exp5"),
                "Effect of batch size (epochs=10, imgsz=480)",
                "compare_batch.png")
    plot_metric_bars()


if __name__ == "__main__":
    main()
