# Small MRF Detection Model

A compact PyTorch object-detection project for damage / MRF detection built around a multi-receptive-field backbone, FPN-style fusion neck, and decoupled classification/regression heads.

This repository is designed for experiments on image-level damage detection, with configurable training, augmentation, evaluation, and checkpointing.

## Highlights

- Lightweight, practical training pipeline in PyTorch
- Config-driven model and training setup via `config.yaml`
- Built-in evaluation with confidence threshold and NMS options
- Visual prediction examples under `eval_images/`

## Project structure

- `config.yaml` – model, training, loss, and evaluation settings.
- `data.py` – dataset loading and image/box augmentation wrappers.
- `model.py` – MRF backbone, neck, detection head, and inference logic.
- `train.py` – training loop, loss function, scheduler, and checkpoint handling.
- `eval.py` – evaluation utilities and CLI evaluation entry point.

## Requirements

Use Python 3.11 or newer.

Install the main dependencies with:

```bash
pip install torch torchvision torchaudio torchmetrics tqdm pyyaml numpy
```

## Dataset layout

The training and evaluation code expects a dataset root like this:

```text
<dataset_root>/
  train.txt
  val.txt
  test.txt
  images/
    image_001.jpg
    ...
  labels/
    image_001.txt
    ...
```

Each label file should use YOLO-style bounding boxes in this format:

```text
class cx cy w h
```

where values are normalized to image size.

## Example predictions

The repository includes evaluation visualizations in `eval_images/` to show ground truth vs. predicted detections.

<p align="center">
  <img src="./eval_images/00_gt_pred.jpg" alt="Prediction example 1" width="320" />
  <img src="./eval_images/04_gt_pred.jpg" alt="Prediction example 2" width="320" />
  <img src="./eval_images/05_gt_pred.jpg" alt="Prediction example 3" width="320" />
</p>

## Training

From the project root:

```bash
python train.py --config config.yaml --dataset_root ../coco_dataset
```

Optional flags:

- `--epochs`
- `--batch_size`
- `--checkpoint_dir`
- `--log_dir`
- `--resume`

## Evaluation

```bash
python eval.py --checkpoint checkpoints/checkpoint_best.pth --dataset_root ../coco_dataset --split val
```

You can also override the confidence threshold with:

```bash
python eval.py --checkpoint checkpoints/checkpoint_best.pth --dataset_root ../coco_dataset --split val --conf 0.15
```

## Additional visual samples

More examples are available in `eval_images/`:

- `00_gt_pred.jpg`
- `04_gt_pred.jpg`
- `05_gt_pred.jpg`
- `10_gt_pred.jpg`
- `11_gt_pred.jpg`
- `17_gt_pred.jpg`
- `21_gt_pred.jpg`
- `24_gt_pred.jpg`
- `29_gt_pred.jpg`

## Notes

- The config file controls model depth, learning rate, augmentation, and NMS behavior.
- Checkpoints are saved under `checkpoints/` by default.
- The training code uses EMA, warmup, cosine decay, and optional fine-tuning phases.
