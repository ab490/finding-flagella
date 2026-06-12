from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
import json
import yaml
import random
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import argparse

logger = logging.getLogger(__name__)

# Cryo-ET slices exhibit heavy intensity tails from missing-wedge reconstruction
# artifacts. Clipping at the 2nd and 98th percentile removes those outliers while
# preserving the biologically relevant contrast in the central 96% of the histogram.
BOX_SIZE_PX = 24  # approximate motor diameter in pixels at 512×512 training resolution


def normalize_data(slices: np.ndarray) -> np.ndarray:
    '''
    Removes the pixel values with extreme intensities
    Clipped within 2nd and 98th percentile
    Then rescales the intensities to [0,255]
    '''
    p2, p98 = np.percentile(slices, [2, 98])
    if p98 == p2:
        return np.zeros_like(slices, dtype=np.uint8)
    clipped_slices = np.clip(slices, p2, p98)
    normalized_slices = 255 * (clipped_slices - p2) / (p98 - p2)
    return np.uint8(normalized_slices)


def split_train_val_test(
        labels_df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Splits dataset into training, validation, and test sets in 70:15:15
    Keeps only those tomograms from the dataset where at least one flagellar motor exists
    '''
    tomo_df = labels_df[labels_df['Number of motors'] > 0].copy()
    unique_tomos = np.array(tomo_df['tomo_id'].unique())
    np.random.shuffle(unique_tomos)

    total = len(unique_tomos)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return unique_tomos[:train_end], unique_tomos[train_end:val_end], unique_tomos[val_end:]


def get_motor_coordinates(
        labels_df: pd.DataFrame,
        tomogram_ids: np.ndarray,
) -> list[tuple[str, int, int, int, int]]:
    '''
    Returns coordinates of the flagellar motor and the depth of the tomogram (z-axis length)
    Each entry is (tomo_id, axis_0, axis_1, axis_2, z_depth).
    '''
    motor_coords: list[tuple[str, int, int, int, int]] = []
    for tomo_id in tomogram_ids:
        tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
        for _, motor in tomo_motors.iterrows():
            if any(pd.isna(motor[f'Motor axis {i}']) for i in range(3)):
                continue
            motor_coords.append((
                tomo_id,
                int(motor['Motor axis 0']),
                int(motor['Motor axis 1']),
                int(motor['Motor axis 2']),
                int(motor['Array shape (axis 0)'])
            ))
    return motor_coords


def process_data(
        motor_coords: list[tuple[str, int, int, int, int]],
        images_dir: Path,
        labels_dir: Path,
        set_name: str,
        slice_margin: int,
        train_dir: Path,
) -> tuple[int, int]:
    '''
    Normalizes images and adds bounding box around detected flagellar motors
    Creates label files required for YOLO compatible dataset
    Returns (processed_slices, num_motors).
    '''
    logger.info(
        "Number of slices to be processed for %s is %d",
        set_name, len(motor_coords) * (2 * slice_margin + 1))
    processed_slices = 0

    for tomo_id, z_coord, y_coord, x_coord, z_max in tqdm(
            motor_coords, desc=f"Processing {set_name} motors"):
        z_min = max(0, z_coord - slice_margin)
        z_upper = min(z_max - 1, z_coord + slice_margin)

        for z in range(z_min, z_upper + 1):
            slice_filename = f"slice_{z:04d}.jpg"
            src_path = train_dir / tomo_id / slice_filename

            if not src_path.exists():
                logger.warning("Skipping because %s does not exist", src_path)
                continue

            img = Image.open(src_path)
            img_array = np.array(img)
            normalized_img = normalize_data(img_array)

            dest_filename = f"{tomo_id}_z{z:04d}_y{y_coord:04d}_x{x_coord:04d}.jpg"
            dest_path = images_dir / dest_filename
            Image.fromarray(normalized_img).save(dest_path)

            img_width, img_height = img.size
            x_coord_norm = x_coord / img_width
            y_coord_norm = y_coord / img_height
            box_width_norm = BOX_SIZE_PX / img_width
            box_height_norm = BOX_SIZE_PX / img_height

            label_path = labels_dir / dest_filename.replace('.jpg', '.txt')
            with open(label_path, 'w') as f:
                f.write(
                    f"0 {x_coord_norm} {y_coord_norm} {box_width_norm} {box_height_norm}\n")

            processed_slices += 1

    return processed_slices, len(motor_coords)


def create_yaml_file(yolo_dataset_dir: Path) -> Path:
    '''
    Generates YOLO-compatible dataset configuration file
    '''
    yaml_content = {
        'path': str(yolo_dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'motor'}
    }
    with open(yolo_dataset_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    return yolo_dataset_dir / 'dataset.yaml'


def create_yolo_data(
        train_dir: Path,
        yolo_dataset_dir: Path,
        labels_file: Path,
        slice_margin: int,
) -> dict:
    '''
    Prepares and processes the dataset into YOLO format with train, val, and test splits, and returns metadata
    '''
    yolo_images_train = yolo_dataset_dir / "images" / "train"
    yolo_images_val = yolo_dataset_dir / "images" / "val"
    yolo_images_test = yolo_dataset_dir / "images" / "test"
    yolo_labels_train = yolo_dataset_dir / "labels" / "train"
    yolo_labels_val = yolo_dataset_dir / "labels" / "val"
    yolo_labels_test = yolo_dataset_dir / "labels" / "test"

    labels_df = pd.read_csv(labels_file)
    motors_all = labels_df['Number of motors'].sum()
    logger.info("Total number of motors present: %d", motors_all)

    train_set, val_set, test_set = split_train_val_test(labels_df)
    logger.info(
        "Dataset split to %d train, %d val, %d test tomograms",
        len(train_set), len(val_set), len(test_set))

    train_motor_coords = get_motor_coordinates(labels_df, train_set)
    val_motor_coords = get_motor_coordinates(labels_df, val_set)
    test_motor_coords = get_motor_coordinates(labels_df, test_set)

    train_slices, train_motors = process_data(
        train_motor_coords, yolo_images_train, yolo_labels_train, "training", slice_margin, train_dir)
    val_slices, val_motors = process_data(
        val_motor_coords, yolo_images_val, yolo_labels_val, "validation", slice_margin, train_dir)
    test_slices, test_motors = process_data(
        test_motor_coords, yolo_images_test, yolo_labels_test, "testing", slice_margin, train_dir)

    yaml_path = create_yaml_file(yolo_dataset_dir)

    return {
        "dataset_dir": yolo_dataset_dir,
        "yaml_path": yaml_path,
        "train_tomograms": len(train_set),
        "val_tomograms": len(val_set),
        "test_tomograms": len(test_set),
        "train_motors": train_motors,
        "val_motors": val_motors,
        "test_motors": test_motors,
        "train_slices": train_slices,
        "val_slices": val_slices,
        "test_slices": test_slices,
        "labels_df": labels_df
    }


def save_dataset_info(yolo_data: dict, output_path: Path) -> None:
    '''
    Saves dataset information for later use in evaluation
    '''
    eval_data = {
        "dataset_dir": str(yolo_data["dataset_dir"]),
        "yaml_path": str(yolo_data["yaml_path"]),
        "test_tomograms": yolo_data["test_tomograms"],
        "test_motors": yolo_data["test_motors"],
        "test_slices": yolo_data["test_slices"]
    }

    with open(output_path, 'w') as f:
        json.dump(eval_data, f, indent=2)

    logger.info("Dataset info saved to %s", output_path)


def visualize_slices(
        results_vis_dir: Path,
        num_slices: int,
        images_dir: Path,
        labels_dir: Path,
) -> None:
    '''
    Visualizes flagellar motor in random training images
    '''
    image_files = list(images_dir.rglob("*.jpg"))

    if not image_files:
        logger.warning("No training images found for visualization.")
        return

    selected_images = random.sample(
        image_files, min(
            num_slices, len(image_files)))
    rows = int(np.ceil(len(selected_images) / 2))
    cols = min(len(selected_images), 2)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = np.array(axes).flatten()

    for i, img_path in enumerate(selected_images):
        label_path = labels_dir / (img_path.stem + ".txt")
        img = Image.open(img_path)
        img_width, img_height = img.size

        img_array = np.array(img)
        normalized_img = normalize_data(img_array)
        img_normalized = Image.fromarray(normalized_img).convert("RGB")

        overlay = Image.new("RGBA", img_normalized.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    # class_label, x_center, y_center, width, height of the
                    # bounding box
                    cls, xc, yc, w, h = map(float, line.strip().split())
                    x1 = int((xc - w / 2) * img_width)
                    y1 = int((yc - h / 2) * img_height)
                    x2 = int((xc + w / 2) * img_width)
                    y2 = int((yc + h / 2) * img_height)
                    draw.rectangle([x1, y1, x2, y2], outline=(
                        255, 0, 0, 255), width=2)
                    draw.text(
                        (x1,
                         y1 - 10),
                        f"Class {int(cls)}",
                        fill=(
                            255,
                            0,
                            0,
                            255))
        else:
            draw.text((10, 10), "No motor found", fill=(255, 0, 0, 255))

        final_img = Image.alpha_composite(
            img_normalized.convert("RGBA"),
            overlay).convert("RGB")
        axes[i].imshow(np.array(final_img))
        axes[i].set_title(img_path.name)
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(
        results_vis_dir /
        f"{num_slices}_sample_annotations.png",
        dpi=300)
    logger.info("Saved visualization to: %s", results_vis_dir)
    plt.close()


def train_yolo(
        yaml_path: Path,
        epochs: int,
        batch_size: int,
        img_size: int,
        model_name: str,
        results_dir: Path,
) -> tuple[YOLO, object]:
    model = YOLO(model_name)
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=str(results_dir),
        name="yolo_train",
        exist_ok=True,
        patience=10,
        val=True,
        verbose=True,
        workers=4
    )
    return model, results


def plot_loss_curve(results_dir: Path) -> None:
    results_csv = results_dir / "yolo_train" / "results.csv"
    if not results_csv.exists():
        logger.warning("results.csv not found, skipping loss curve: %s", results_csv)
        return
    df = pd.read_csv(results_csv)
    missing = [c for c in ("train/box_loss", "val/box_loss", "epoch") if c not in df.columns]
    if missing:
        logger.warning("Columns missing from results.csv, skipping loss curve: %s", missing)
        return

    train_loss = df["train/box_loss"]
    val_loss = df["val/box_loss"]

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], train_loss, label="Train Loss")
    plt.plot(df['epoch'], val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training and Validation Loss")
    plt.savefig(results_dir / "loss_curve.png")
    plt.close()
    logger.info("Loss curve saved at: %s", results_dir / "loss_curve.png")


def train_pipeline(
        epochs: int,
        batch_size: int,
        img_size: int,
        model_name: str,
        yolo_dataset_dir: Path,
        results_dir: Path,
) -> None:
    '''
    Trains the model using training data
    Uses validation data for model selection
    '''
    yaml_path = yolo_dataset_dir / "dataset.yaml"

    logger.info("Starting model training...")
    model, results = train_yolo(
        yaml_path, epochs, batch_size, img_size, model_name, results_dir)
    plot_loss_curve(results_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    np.random.seed(42)
    random.seed(42)

    EPOCHS = 50
    BATCH_SIZE = 16
    IMG_SIZE = 512

    project_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing train/ and train_labels.csv")
    args = parser.parse_args()

    train_dir = args.data_dir / "train"
    labels_file = args.data_dir / "train_labels.csv"

    if not args.data_dir.is_dir():
        parser.error(f"--data-dir does not exist: {args.data_dir}")
    if not train_dir.exists():
        parser.error(f"--data-dir must contain a 'train/' subdirectory: {train_dir}")
    if not labels_file.exists():
        parser.error(f"--data-dir must contain 'train_labels.csv': {labels_file}")

    yolo_dataset_dir = project_root / "datasets" / "yolo_data"
    yolo_images_train = yolo_dataset_dir / "images" / "train"
    yolo_images_val = yolo_dataset_dir / "images" / "val"
    yolo_images_test = yolo_dataset_dir / "images" / "test"
    yolo_labels_train = yolo_dataset_dir / "labels" / "train"
    yolo_labels_val = yolo_dataset_dir / "labels" / "val"
    yolo_labels_test = yolo_dataset_dir / "labels" / "test"

    results_dir = project_root / "results" / args.model_name
    results_vis_dir = results_dir / "vis"

    for path in [
            yolo_dataset_dir,
            yolo_images_train,
            yolo_images_val,
            yolo_images_test,
            yolo_labels_train,
            yolo_labels_val,
            yolo_labels_test,
            results_dir,
            results_vis_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # create dataset in YOLO format
    yolo_data = create_yolo_data(
        train_dir,
        yolo_dataset_dir,
        labels_file,
        slice_margin=5)

    save_dataset_info(yolo_data, results_dir / "dataset_info.json")

    # visualize data
    visualize_slices(results_vis_dir, num_slices=4,
                     images_dir=yolo_images_train, labels_dir=yolo_labels_train)

    # train model
    train_pipeline(
        EPOCHS,
        BATCH_SIZE,
        IMG_SIZE,
        args.model_name,
        yolo_dataset_dir,
        results_dir)
