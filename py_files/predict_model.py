from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import random
from pathlib import Path
from ultralytics import YOLO
import argparse

logger = logging.getLogger(__name__)

# Minimum confidence for a detection to be kept.
CONF_THRESHOLD = 0.3

# Scatter dot area = confidence * SCATTER_DOT_SCALE (matplotlib `s` units).
# At confidence 1.0 this gives 50pt², making dots clearly visible without
# overwhelming the plot at the expected 0.3–1.0 confidence range.
SCATTER_DOT_SCALE = 50

_Z_PATTERNS = [
    r'_z(\d+)_',
    r'_z(\d+)\.',
    r'z(\d+)_',
    r'slice_?(\d+)',
    r'_(\d+)_',
]


def extract_z(img_path: str | Path) -> int:
    """Return the z-slice index parsed from img_path's filename, or -1 if unparseable."""
    for pattern in _Z_PATTERNS:
        match = re.search(pattern, Path(img_path).name)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return -1


def predict_on_test(
        model: YOLO,
        test_dir: Path,
        predictions_dir: Path,
        label_dir: Path,
        num_tomos: int | None,
) -> None:
    '''
    Predicts and visualizes 3D flagellar motor detections from test slices.

    The function:
    - Groups images by tomogram ID parsed from filenames like 'tomoX_z123_y456_x789.jpg'
    - Randomly selects 'num_tomos' tomograms
    - Runs predictions on all slices for each selected tomogram
    - Filters out low-confidence detections (conf < CONF_THRESHOLD)
    - Saves per-slice prediction images and a combined CSV of 3D predictions
    - Generates a 3D scatter plot for each tomogram to visualize detection locations
    '''
    plot_dir = predictions_dir / "3d_visualizations"
    plot_dir.mkdir(parents=True, exist_ok=True)

    predictions_3d: list[dict] = []

    image_files = sorted(Path(test_dir).glob("*.jpg"))
    if not image_files:
        logger.warning("No test images found in: %s", test_dir)
        return

    grouped_by_tomo: dict[str, list[Path]] = {}
    for img_path in image_files:
        match = re.match(r'(.+?)_z\d+_', img_path.stem)
        if not match:
            logger.warning("Could not parse tomo_id from %s, skipping", img_path.name)
            continue
        grouped_by_tomo.setdefault(match.group(1), []).append(img_path)

    tomo_ids = list(grouped_by_tomo.keys())
    if num_tomos:
        tomo_ids = random.sample(tomo_ids, min(num_tomos, len(tomo_ids)))

    for tomo_id in tqdm(tomo_ids, desc="Processing tomograms"):
        for img_path in sorted(grouped_by_tomo[tomo_id]):
            results = model.predict(str(img_path), conf=CONF_THRESHOLD, verbose=False)
            result_img = results[0].plot()
            Image.fromarray(result_img).save(
                predictions_dir / f"pred_{img_path.name}")

            z = extract_z(img_path)
            if z == -1:
                logger.warning("Could not extract z-coordinate from %s, skipping slice", img_path.name)
                continue

            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue
                x, y = box.xywh[0][:2].tolist()
                w, h = box.xywh[0][2:4].tolist()
                cls = int(box.cls[0]) if len(box.cls) > 0 else -1
                predictions_3d.append({
                    "tomo_id": tomo_id,
                    "image": img_path.name,
                    "x": x, "y": y, "z": z,
                    "width": w, "height": h,
                    "confidence": conf, "class": cls
                })

    df_preds = pd.DataFrame(predictions_3d)
    df_preds.to_csv(
        predictions_dir /
        "3d_predictions_test.csv",
        index=False)
    logger.info("Saved test predictions to %s", predictions_dir / "3d_predictions_test.csv")

    if df_preds.empty:
        logger.warning("No detections above confidence threshold %.2f — skipping 3D plots", CONF_THRESHOLD)
        return

    img_width, img_height = Image.open(image_files[0]).size

    for tomo_id, group in df_preds.groupby("tomo_id"):

        # plot 1: only predictions
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(
            group['x'], group['y'], group['z'],
            c=group['confidence'], cmap='plasma', vmin=0, vmax=1,
            s=group['confidence'] * SCATTER_DOT_SCALE, alpha=0.7, label='Predicted'
        )
        ax.set_title(f"{len(group)} Detections in {tomo_id}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(0, img_width)
        ax.set_ylim(0, img_height)
        ax.legend()
        plt.colorbar(sc, label="Confidence", shrink=0.6)
        plt.savefig(plot_dir / f"3d_scatter_pred_{tomo_id}.png", dpi=300)
        plt.close()

        # plot 2: predictions with ground truth
        gt_coords: list[tuple[float, float, int]] = []
        for label_path in label_dir.glob(f"{tomo_id}_*.txt"):
            z_match = re.search(r'_z(\d+)_', label_path.name)
            if not z_match:
                continue
            z = int(z_match.group(1))

            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    _, x_rel, y_rel, w_rel, h_rel = map(float, parts)

                    x = x_rel * img_width
                    y = y_rel * img_height

                    gt_coords.append((x, y, z))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(
            group['x'], group['y'], group['z'],
            c=group['confidence'], cmap='plasma', vmin=0, vmax=1,
            s=group['confidence'] * SCATTER_DOT_SCALE, alpha=0.7, label='Predicted'
        )

        if gt_coords:
            gt_arr = np.array(gt_coords)
            ax.scatter(gt_arr[:, 0], gt_arr[:, 1], gt_arr[:, 2],
                       color='black', s=30, marker='x', label='Ground Truth', alpha=1)

        ax.set_title(f"{len(group)} Detections in {tomo_id} (with Ground Truth)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(0, img_width)
        ax.set_ylim(0, img_height)
        ax.legend()
        plt.colorbar(sc, label="Confidence", shrink=0.6)
        plt.savefig(plot_dir / f"3d_scatter_{tomo_id}.png", dpi=300)
        plt.close()

        logger.info("Saved both scatter plots for %s", tomo_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    np.random.seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    model_path = project_root / "results" / args.model_name / "yolo_train" / "weights" / "best.pt"
    test_dir = project_root / "datasets" / "yolo_data" / "images" / "test"
    label_dir = project_root / "datasets" / "yolo_data" / "labels" / "test"
    predictions_dir = project_root / "results" / args.model_name / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_dir}\n"
            f"Run training first: uv run python py_files/train_model.py {args.model_name} --data-dir <path>"
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            f"Run training first: uv run python py_files/train_model.py {args.model_name} --data-dir <path>"
        )
    model = YOLO(model_path)

    logger.info("Running 3D test predictions...")
    predict_on_test(model, test_dir, predictions_dir, label_dir, num_tomos=10)

    logger.info("All predictions completed.")
