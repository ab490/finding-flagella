# Detection of Bacterial Flagellar Motors in Tomograms

## Table of Contents
- [Abstract](#abstract)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Results](#results)

## Abstract

Flagellar motors are important nanomachines in bacteria that drive cell motility and play a key role in understanding biological processes such as microbial behavior, drug development, and synthetic biology. While cryogenic electron tomography (cryo-ET) enables detailed 3D imaging of these motors, manually locating them across thousands of tomographic slices is time-consuming and prone to errors. We address the challenge of automating flagellar motor detection by leveraging the YOLO object detection framework. We benchmark multiple YOLOv8 and YOLOv9 variants on labeled cryo-ET slices, evaluating performance using precision, recall, and mean Average Precision (mAP). Our results demonstrate that YOLOv9e outperforms all other models, achieving the highest mAP of 0.793. We also reconstruct 3D motor locations across slices and visualize confidence weighted scatter plots to evaluate spatial consistency. This automated approach significantly accelerates the identification process, paving the way for scalable analysis of macromolecular complexes in bacterial tomography.

## Repository Structure

| File/Folder      | Description                                                                      |
|------------------|----------------------------------------------------------------------------------|
| `py_files/`      | Python scripts for training and inference.                                       |
| `bash_scripts/`  | Portable shell scripts for running training and inference.                       |
| `imgs/`          | Images used in this README.                                                      |
| `pyproject.toml` | Project metadata and dependencies (managed with [uv](https://github.com/astral-sh/uv)). |
| `tests/`         | pytest test suite.                                                               |
| `.gitignore`     | Specifies untracked files ignored by git.                                        |

#### Files in `py_files/`
- `train_model.py` – Prepares the dataset and trains the YOLO model on 2D cryo-ET slices.
- `predict_model.py` – Runs inference on test tomograms and generates 2D overlays and 3D visualizations of predicted motor locations.

#### Files in `bash_scripts/`
- `train.sh` – Trains one or more YOLO models.
- `predict.sh` – Runs inference with a trained model.


## Quick Start

**Requirements:** [uv](https://docs.astral.sh/uv/getting-started/installation/) — installs Python 3.12 automatically.

```bash
# 1. Clone and install
git clone <repo-url>
cd finding-flagella
uv sync

# 2. Download the dataset (requires a Kaggle account)
uv sync --extra download
uv run kaggle competitions download -c byu-locating-bacterial-flagellar-motors-2025
unzip byu-locating-bacterial-flagellar-motors-2025.zip -d /path/to/data
rm byu-locating-bacterial-flagellar-motors-2025.zip
# /path/to/data must contain:
#   train/                   one subdirectory per tomogram, each with slice_NNNN.jpg files
#   train_labels.csv         columns: tomo_id, Number of motors, Motor axis 0/1/2, Array shape (axis 0)

# 3. Train (all models, or name specific ones)
bash bash_scripts/train.sh --data-dir /path/to/data
bash bash_scripts/train.sh --data-dir /path/to/data yolov9e

# 4. Run inference
bash bash_scripts/predict.sh yolov9e

# 5. Run tests
uv run pytest tests/
```

Results are written to `results/<model_name>/`:
- `yolo_train/` – model weights, metrics, and Ultralytics outputs
- `predictions/` – per-slice overlay images, `3d_predictions_test.csv`, and 3D scatter plots
- `vis/` – annotated sample images from the training set
- `loss_curve.png` – training vs. validation box loss

You can also invoke the scripts directly (from any working directory):
```bash
uv run python py_files/train_model.py yolov9e --data-dir /path/to/data
uv run python py_files/predict_model.py yolov9e
```

## Results

### YOLOv8/YOLOv9 Model Performance Comparison

| Model     | # Parameters | Model Size | Best Epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-----------|--------------|------------|------------|-----------|--------|---------|--------------|
| YOLOv8n   | 3.2M         | 6 MB       | 20         | 0.7987    | 0.6901 | 0.76483 | 0.28271      |
| YOLOv8s   | 11.2M        | 22 MB      | 13         | 0.7816    | 0.6542 | 0.70076 | 0.24893      |
| YOLOv8m   | 25.9M        | 52 MB      | 8          | 0.7553    | 0.6857 | 0.71108 | 0.24099      |
| YOLOv8l   | 43.7M        | 90 MB      | 20         | 0.6988    | 0.6157 | 0.61955 | 0.15407      |
| YOLOv8x   | 68.2M        | 128 MB     | 13         | 0.7305    | 0.6426 | 0.66636 | 0.19793      |
| YOLOv9s   | 7.2M         | 28 MB      | 21         | 0.8065    | 0.7180 | 0.77711 | 0.25570      |
| YOLOv9m   | 20.1M        | 78 MB      | 15         | 0.8018    | 0.6612 | 0.71714 | 0.22759      |
| YOLOv9c   | 25.5M        | 100 MB     | 14         | 0.7608    | 0.6467 | 0.68184 | 0.19065      |
| YOLOv9e   | 58.1M        | 220 MB     | 22         | 0.8100    | 0.7486 | 0.79304 | 0.30113      |

**Training Parameters Used:**

- **Epochs**: 50  
- **Batch Size**: 16  
- **Image Size**: 512×512  
- **Early Stopping Patience**: 10

### Model Performance vs Number of Parameters:
<img src="/imgs/map_vs_params.png" width="600"><br>

### Precision-Recall Curves:
<table>
  <tr>
    <td align="center"><img src="/imgs/yolov8s_PR_curve.png" width="200"><br><strong>YOLOv8s</strong></td>
    <td align="center"><img src="/imgs/yolov8m_PR_curve.png" width="200"><br><strong>YOLOv8m</strong></td>
    <td align="center"><img src="/imgs/yolov8l_PR_curve.png" width="200"><br><strong>YOLOv8l</strong></td>
    <td align="center"><img src="/imgs/yolov8x_PR_curve.png" width="200"><br><strong>YOLOv8x</strong></td>
  </tr>
  <tr>
    <td align="center"><img src="/imgs/yolov9s_PR_curve.png" width="200"><br><strong>YOLOv9s</strong></td>
    <td align="center"><img src="/imgs/yolov9m_PR_curve.png" width="200"><br><strong>YOLOv9m</strong></td>
    <td align="center"><img src="/imgs/yolov9c_PR_curve.png" width="200"><br><strong>YOLOv9c</strong></td>
    <td align="center"><img src="/imgs/yolov9e_PR_curve.png" width="200"><br><strong>YOLOv9e</strong></td>
  </tr>
</table>

<img src="/imgs/PR_curve.png" width="600"><br>

We selected YOLOv9e for further evaluation because it achieved the highest accuracy and had the highest AUC for precision-recall curve, and our available compute can support it.

### Loss Curve for YOLOv9e:
<img src="/imgs/loss_curve.png"><br>


### 2D Detection of Flagellar Motors Across Slices:

<p align="center">
  <img src="imgs/pred_tomo_256717_z0174_y0770_x0514.jpg" width="186"/>&nbsp;&nbsp;
  <img src="imgs/pred_tomo_2b3cdf_z0137_y0173_x0662.jpg" width="180"/>&nbsp;&nbsp;
  <img src="imgs/pred_tomo_bbe766_z0198_y0855_x0288.jpg" width="180"/>&nbsp;&nbsp;
  <img src="imgs/pred_tomo_d2b1bc_z0095_y0535_x0192.jpg" width="180"/>
  <br>
</p>

YOLOv9e accurately detects flagellar motors across bacterial cells of varying shapes and motor counts. Each prediction is marked with a bounding box and confidence score.


### 3D Localization of Flagellar Motors Across Tomogram Volumes:
<p align="center">
  <img src="imgs/3d_scatter_tomo_256717.png" width="400"/>&nbsp;&nbsp;
  <img src="imgs/3d_scatter_tomo_2b3cdf.png" width="400"/>
  <br>
</p>

<p align="center">
  <img src="imgs/3d_scatter_tomo_bbe766.png" width="400"/>&nbsp;&nbsp;
  <img src="imgs/3d_scatter_tomo_da79d8.png" width="400"/>
    <br>
</p>

Each 3D scatter plot shows YOLOv9e-predicted flagellar motor positions aggregated across tomographic slices for individual volumes. Detections are color-coded and scaled by confidence. In most cases, predictions align well with the ground truth and are also able to accurately capture multiple motors and maintaining spatial consistency. However, occasional false positives are also observed where the model predicts motors in regions without annotated ground truth.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT)