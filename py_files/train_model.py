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


def normalize_data(slices):
    '''
    Removes the pixel values with extreme intensities
    Clipped within 2nd and 98th percentile
    Then rescales the intensities to [0,255]
    '''
    p2, p98 = np.percentile(slices, [2, 98])
    clipped_slices = np.clip(slices, p2, p98)
    normalized_slices = 255 * (clipped_slices - p2) / (p98 - p2)
    return np.uint8(normalized_slices)


def split_train_val_test(labels_df, train_ratio=0.7, val_ratio=0.15):
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


def get_motor_coordinates(labels_df, tomogram_ids):
    '''
    Returns coordinates of the flagellar motor and the depth of the tomogram (z-axis length)
    '''
    motor_coords = []
    for tomo_id in tomogram_ids:
        tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
        for _, motor in tomo_motors.iterrows():
            if pd.isna(motor['Motor axis 0']):
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
        motor_coords,
        images_dir,
        labels_dir,
        set_name,
        slice_margin,
        train_dir):
    '''
    Normalizes images and adds bounding box around detected flagellar motors
    Creates label files required for YOLO compatible dataset
    '''
    print(
        f"Number of slices to be processed for {set_name} is {len(motor_coords) * (2 * slice_margin + 1)}")
    processed_slices = 0

    for tomo_id, z_coord, y_coord, x_coord, z_max in tqdm(
            motor_coords, desc=f"Processing {set_name} motors"):
        z_min = max(0, z_coord - slice_margin)
        z_max = min(z_max - 1, z_coord + slice_margin)

        for z in range(z_min, z_max + 1):
            slice_filename = f"slice_{z:04d}.jpg"
            src_path = train_dir / tomo_id / slice_filename

            if not src_path.exists():
                print(f"Warning: Skipping because {src_path} does not exist")
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
            # using box size as 24
            box_width_norm = 24 / img_width
            box_height_norm = 24 / img_height

            label_path = labels_dir / dest_filename.replace('.jpg', '.txt')
            with open(label_path, 'w') as f:
                f.write(
                    f"0 {x_coord_norm} {y_coord_norm} {box_width_norm} {box_height_norm}\n")

            processed_slices += 1

    return processed_slices, len(motor_coords)


def create_yaml_file(yolo_dataset_dir):
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


def create_yolo_data(train_dir, yolo_dataset_dir, labels_file, slice_margin):
    '''
    Prepares and processes the dataset into YOLO format with train, val, and test splits, and returns metadata
    '''
    labels_df = pd.read_csv(labels_file)
    motors_all = labels_df['Number of motors'].sum()
    print(f"Total number of motors present: {motors_all}")

    train_set, val_set, test_set = split_train_val_test(labels_df)
    print(
        f"Dataset split to {len(train_set)} train, {len(val_set)} val, {len(test_set)} test tomograms")

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


def save_dataset_info(yolo_data, output_path):
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

    print(f"Dataset info saved to {output_path}")


def visualize_slices(results_vis_dir, num_slices):
    '''
    Visualizes flagellar motor in random training images
    '''
    image_files = list(yolo_images_train.rglob("*.jpg"))

    if not image_files:
        print("Warning: No training images found for visualization.")
        return

    selected_images = random.sample(
        image_files, min(
            num_slices, len(image_files)))
    rows = int(np.ceil(len(selected_images) / 2))
    cols = min(len(selected_images), 2)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = np.array(axes).flatten()

    for i, img_path in enumerate(selected_images):
        label_path = yolo_labels_train / (img_path.stem + ".txt")
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
    print(f"Saved visualization to: {results_vis_dir}")
    plt.close()


def train_yolo(
        yaml_path,
        epochs,
        batch_size,
        img_size,
        model_name,
        results_dir):
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


def plot_loss_curve(results_dir):
    results_csv = results_dir / "yolo_train" / "results.csv"
    df = pd.read_csv(results_csv)

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
    print("Loss curve saved at:", results_dir / "loss_curve.png")


def train_pipeline(
        epochs,
        batch_size,
        img_size,
        model_name,
        yolo_dataset_dir,
        results_dir):
    '''
    Trains the model using training data
    Uses validation data for model selection
    '''
    yaml_path = yolo_dataset_dir / "dataset.yaml"

    print("Starting model training...")
    model, results = train_yolo(
        yaml_path, epochs, batch_size, img_size, model_name, results_dir)
    plot_loss_curve(results_dir)


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    EPOCHS = 50
    BATCH_SIZE = 16
    IMG_SIZE = 512

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    args = parser.parse_args()

    # change the username in base_data_path to yours
    base_data_path = Path("/N/scratch/anobajaj/data")
    train_dir = base_data_path / "train"
    labels_file = base_data_path / "train_labels.csv"

    yolo_dataset_dir = Path("../datasets/yolo_data")
    yolo_images_train = yolo_dataset_dir / "images" / "train"
    yolo_images_val = yolo_dataset_dir / "images" / "val"
    yolo_images_test = yolo_dataset_dir / "images" / "test"
    yolo_labels_train = yolo_dataset_dir / "labels" / "train"
    yolo_labels_val = yolo_dataset_dir / "labels" / "val"
    yolo_labels_test = yolo_dataset_dir / "labels" / "test"

    results_dir = Path("../results") / args.model_name
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
    visualize_slices(results_vis_dir, num_slices=4)

    # train model
    train_pipeline(
        EPOCHS,
        BATCH_SIZE,
        IMG_SIZE,
        args.model_name,
        yolo_dataset_dir,
        results_dir)
