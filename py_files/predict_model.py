import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import json
import random
from pathlib import Path
from ultralytics import YOLO
import argparse


def predict_on_test(model, test_dir, predictions_dir, num_tomos):
    '''
    Predicts and visualizes 3D flagellar motor detections from test slices.

    The function:
    - Groups images by tomogram ID parsed from filenames like 'tomoX_z123_y456_x789.jpg'
    - Randomly selects 'num_tomos' tomograms
    - Runs predictions on all slices for each selected tomogram
    - Filters out low-confidence detections (conf < 0.3)
    - Saves per-slice prediction images and a combined CSV of 3D predictions
    - Generates a 3D scatter plot for each tomogram to visualize detection locations
    '''
    plot_dir = predictions_dir / "3d_visualizations"
    plot_dir.mkdir(parents=True, exist_ok=True)

    predictions_3d = []
    z_patterns = [
        r'_z(\d+)_',
        r'_z(\d+)\.',
        r'z(\d+)_',
        r'slice_?(\d+)',
        r'_(\d+)_'
    ]

    image_files = sorted(Path(test_dir).glob("*.jpg"))
    if not image_files:
        print("No test images found in:", test_dir)
        return

    grouped_by_tomo = {}
    for img_path in image_files:
        match = re.match(r'(.+?)_z\d+_', img_path.stem)
        tomo_id = match.group(1) if match else "unknown"
        grouped_by_tomo.setdefault(tomo_id, []).append(img_path)

    tomo_ids = list(grouped_by_tomo.keys())
    if num_tomos:
        tomo_ids = random.sample(tomo_ids, min(num_tomos, len(tomo_ids)))

    for tomo_id in tqdm(tomo_ids, desc="Processing tomograms"):
        for img_path in sorted(grouped_by_tomo[tomo_id]):
            results = model.predict(str(img_path), conf=0.3, verbose=False)
            result_img = results[0].plot()
            Image.fromarray(result_img).save(
                predictions_dir / f"pred_{img_path.name}")

            # extracting z-index (depth)
            z = -1
            for pattern in z_patterns:
                match = re.search(pattern, img_path.name)
                if match:
                    try:
                        z = int(match.group(1))
                        break
                    except ValueError:
                        continue
            if z == -1:
                numbers = re.findall(r'\d+', img_path.stem)
                if numbers:
                    z = int(numbers[-1])

            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf < 0.3:
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
    print(
        f"Saved test predictions to {predictions_dir / '3d_predictions_test.csv'}")

    label_dir = Path("../datasets/yolo_data/labels/test")
    
    image_files = list(Path(test_dir).glob("*.jpg"))
    sample_img_path = random.choice(image_files)    
    img_width, img_height = Image.open(sample_img_path).size
    print(f"{img_width=}")
    print(f"{img_height=}")
        
    for tomo_id, group in df_preds.groupby("tomo_id"):
        
        # plot 1: only predictions
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(
            group['x'], group['y'], group['z'],
            c=group['confidence'], cmap='plasma', vmin=0, vmax=1,  
            s=group['confidence'] * 50, alpha=0.7, label='Predicted'
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
        gt_coords = []
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
            s=group['confidence'] * 50, alpha=0.7, label='Predicted'
        )                    

        if gt_coords:
            gt_coords = np.array(gt_coords)
            ax.scatter(gt_coords[:, 0], gt_coords[:, 1], gt_coords[:, 2],
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
        
        print(f"Saved both scatter plots for {tomo_id}")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    args = parser.parse_args()

    model_path = Path(f"../results/{args.model_name}/yolo_train/weights/best.pt")
    test_dir = Path("../datasets/yolo_data/images/test")
    predictions_dir = Path(f"../results/{args.model_name}/predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    print("\nRunning 3D test predictions...")
    predict_on_test(model, test_dir, predictions_dir, num_tomos=10)

    print("\nAll predictions completed.")
