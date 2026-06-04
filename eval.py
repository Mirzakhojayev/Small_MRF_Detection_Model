import os
import torch
from torchvision.ops import nms as tv_nms, batched_nms
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision


def hard_nms(boxes, scores, labels, iou_threshold=0.5, score_threshold=0.25):
    """
    Multi-class Hard-NMS via torchvision.ops.batched_nms.
    Each class is offset into its own 'lane' so NMS never crosses class boundaries.
    """
    keep = scores >= score_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if scores.numel() == 0:
        device = boxes.device
        return (torch.zeros((0, 4), device=device),
                torch.zeros((0,),  device=device),
                torch.zeros((0,),  dtype=torch.long, device=device))
    idxs = batched_nms(boxes, scores, labels, iou_threshold)
    return boxes[idxs], scores[idxs], labels[idxs]


def soft_nms(boxes, scores, labels, iou_threshold=0.5, sigma=0.5, score_threshold=0.001):
    """
    Multi-class Soft-NMS with Gaussian decay.
    torchvision does not ship a soft-NMS op, so this stays as a thin helper.
    The loop is per-class and exits as soon as scores drop below threshold.
    """
    from torchvision.ops import box_iou
    device = boxes.device
    out_b, out_s, out_l = [], [], []

    for lbl in torch.unique(labels):
        m = labels == lbl
        cb = boxes[m][scores[m] >= score_threshold]
        cs = scores[m][scores[m] >= score_threshold]
        if cs.numel() == 0:
            continue
        wb, ws, kb, ks = cb.clone(), cs.clone(), [], []
        while ws.numel() > 0 and ws.max() >= score_threshold:
            i = ws.argmax()
            kb.append(wb[i])
            ks.append(ws[i])
            rem = torch.arange(ws.numel(), device=device) != i
            ious = box_iou(wb[i:i+1], wb[rem]).squeeze(0)
            ws = ws[rem] * torch.exp(-(ious**2) / sigma)
            wb = wb[rem][ws >= score_threshold]
            ws = ws[ws >= score_threshold]
        if kb:
            out_b.append(torch.stack(kb))
            out_s.append(torch.stack(ks))
            out_l.append(torch.full((len(kb),), lbl,
                         dtype=torch.long, device=device))

    if not out_b:
        return (torch.zeros((0, 4), device=device),
                torch.zeros((0,),  device=device),
                torch.zeros((0,),  dtype=torch.long, device=device))
    return torch.cat(out_b), torch.cat(out_s), torch.cat(out_l)


def evaluate(model, dataloader: DataLoader, config: dict, ema=None) -> dict:
    """
    Evaluates the model using torchmetrics.detection.MeanAveragePrecision.
    No JSON files, no pycocotools. Just pass preds and targets directly.
    Returns dict: {map50, map50_95, recall50, precision50, f1_50}.
    """
    eval_model = ema if ema is not None else model
    eval_model.eval()
    device = next(eval_model.parameters()).device
    conf_thresh = config.get('eval', {}).get('conf_threshold', 0.25)

    metric = MeanAveragePrecision(
        iou_type='bbox', iou_thresholds=[0.5]).to(device)

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            preds = eval_model(imgs, conf_threshold=conf_thresh)

            # torchmetrics expects lists of dicts
            pred_list = [
                {
                    'boxes':  p['boxes'].to(device),
                    'scores': p['scores'].to(device),
                    'labels': p['labels'].to(device),
                }
                for p in preds
            ]
            gt_list = [
                {
                    'boxes':  t['boxes'].to(device),
                    'labels': t['labels'].to(device),
                }
                for t in targets
            ]
            metric.update(pred_list, gt_list)

    result = metric.compute()

    map50 = float(result.get('map_50',    0.0))
    map50_95 = float(result.get('map',       0.0))
    # torchmetrics exposes per-class recall/precision as tensors; take mean
    mar50 = result.get('mar_100', torch.tensor(0.0))
    recall50 = float(mar50.mean() if mar50.numel() > 1 else mar50)

    p50 = result.get('map_per_class', torch.tensor([0.0]))
    precision50 = float(p50.clamp(0).mean())

    denom = precision50 + recall50
    f1_50 = 2 * precision50 * recall50 / (denom + 1e-9)

    metrics = {
        'map50':       map50,
        'map50_95':    map50_95,
        'recall50':    recall50,
        'precision50': precision50,
        'f1_50':       f1_50,
    }
    print(f"[eval] mAP@0.5={map50:.4f}  mAP@0.5:0.95={map50_95:.4f}  "
          f"recall={recall50:.4f}  precision={precision50:.4f}  F1={f1_50:.4f}")
    return metrics


if __name__ == '__main__':
    import argparse
    from train import load_config, collate_fn, seed_everything
    from data import DamageDataset, AugmentationWrapper
    from model import build_model

    p = argparse.ArgumentParser()
    p.add_argument('--config',       default='config.yaml')
    p.add_argument('--dataset_root', default='../coco_dataset')
    p.add_argument('--checkpoint',   required=True)
    p.add_argument('--split',        default='val', choices=['val', 'test'])
    p.add_argument('--conf',         type=float, default=None)
    a = p.parse_args()

    config = load_config(a.config)
    seed_everything(config['training'].get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = DamageDataset(os.path.join(a.dataset_root, f"{a.split}.txt"))
    ds = AugmentationWrapper(ds, config, is_train=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False,
                        num_workers=2, collate_fn=collate_fn, pin_memory=True)

    model = build_model(config).to(device)
    ckpt = torch.load(a.checkpoint, map_location=device)
    model.load_state_dict(
        ckpt.get('ema_state_dict', ckpt['model_state_dict']), strict=False)

    if a.conf is not None:
        config.setdefault('eval', {})['conf_threshold'] = a.conf

    results = evaluate(model, loader, config)
    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
