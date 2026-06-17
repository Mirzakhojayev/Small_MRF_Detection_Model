# Small MRF Detection Model

A compact PyTorch object-detection project for damage / MRF detection built around a multi-receptive-field backbone, FPN-style fusion neck, and decoupled classification/regression heads.

This repository is designed for experiments on image-level damage detection, with configurable training, augmentation, evaluation, and checkpointing.

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
python train.py
```

Optional flags:

- `--epochs`
- `--batch_size`
- `--checkpoint_dir`
- `--log_dir`
- `--resume`

## Evaluation

```bash
python eval.py
```

You can also override the confidence threshold with:

```bash
python eval.py
```

## Notes

- The config file controls model depth, learning rate, augmentation, and NMS behavior.
- Checkpoints are saved under `checkpoints/` by default.
- The training code uses EMA, warmup, cosine decay, and optional fine-tuning phases.
