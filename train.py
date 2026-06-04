import os
import csv
import math
import warnings
import random
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ConstantLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchvision.ops import complete_box_iou_loss, box_iou, sigmoid_focal_loss
from tqdm import tqdm


class FCOSAssigner(nn.Module):
    """
    Assigns GT boxes to FPN grid points for a single image.
    Scale assignment by sqrt(box_area):
      P2 [0,64) | P3 [64,128) | P4 [128,256) | P5 [256,inf)
    A point is a candidate if inside the GT box OR within
    center_sampling_radius * stride of the GT center.
    Ties broken by smallest GT area.
    """
    SIZE_RANGES = [(0, 64), (64, 128), (128, 256), (256, float('inf'))]

    def __init__(self, center_sampling_radius: float = 1.5, strides: list = [4, 8, 16, 32]):
        super().__init__()
        self.radius = center_sampling_radius
        self.strides = strides

    def _grid(self, H, W, stride, device):
        sy = torch.arange(H // stride, dtype=torch.float32, device=device)
        sx = torch.arange(W // stride, dtype=torch.float32, device=device)
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        cx = sx.reshape(-1) * stride + stride / 2.0
        cy = sy.reshape(-1) * stride + stride / 2.0
        return torch.stack([cx, cy], dim=-1)

    def forward(self, gt_boxes: torch.Tensor, H: int, W: int):
        device, num_gts = gt_boxes.device, gt_boxes.shape[0]

        def empty(n): return {
            'assigned_indices': torch.full((n,), -1, dtype=torch.long, device=device),
            'assigned_boxes':   torch.zeros((n, 4), dtype=torch.float32, device=device)
        }
        if num_gts == 0:
            return [empty((H // s) * (W // s)) for s in self.strides]

        x1, y1, x2, y2 = gt_boxes.unbind(1)
        gt_cx = (x1 + x2) / 2.0
        gt_cy = (y1 + y2) / 2.0
        gt_area = (x2 - x1) * (y2 - y1)
        gt_side = gt_area.sqrt()

        results = []
        for lvl, stride in enumerate(self.strides):
            grid = self._grid(H, W, stride, device)
            n_l = grid.shape[0]
            cx, cy = grid[:, 0], grid[:, 1]
            min_s, max_s = self.SIZE_RANGES[lvl]
            size_mask = (gt_side >= min_s) & (gt_side < max_s)

            if not size_mask.any():
                results.append(empty(n_l))
                continue

            li = torch.where(size_mask)[0]
            lb = gt_boxes[li]
            lc = gt_cx[li]
            lcy = gt_cy[li]
            la = gt_area[li]
            r_px = self.radius * stride
            cand = ((cx.unsqueeze(1) - lc).abs() <= r_px) & \
                ((cy.unsqueeze(1) - lcy).abs() <= r_px) | \
                ((cx.unsqueeze(1) >= lb[:, 0]) & (cx.unsqueeze(1) <= lb[:, 2]) &
                 (cy.unsqueeze(1) >= lb[:, 1]) & (cy.unsqueeze(1) <= lb[:, 3]))

            ai = torch.full((n_l,), -1, dtype=torch.long, device=device)
            ab = torch.zeros((n_l, 4), dtype=torch.float32, device=device)
            if cand.any():
                am = la.unsqueeze(0).expand(n_l, -1).clone()
                am[~cand] = float('inf')
                mv, mi = am.min(dim=1)
                v = mv < float('inf')
                if v.any():
                    gi = li[mi[v]]
                    ai[v] = gi
                    ab[v] = gt_boxes[gi]
            results.append({'assigned_indices': ai, 'assigned_boxes': ab})

        return results


class MRFDetLoss(nn.Module):
    """
    Three-component loss normalized by number of positive samples:
    - Focal loss via torchvision.ops.sigmoid_focal_loss
    - CIoU loss via torchvision.ops.complete_box_iou_loss
    - Distribution Focal Loss (DFL) — no pytorch built-in, kept minimal
    Soft cls targets use predicted IoU quality (GFL/TOOD style).
    """

    def __init__(self, config: dict):
        super().__init__()
        lcfg = config['loss']
        self.lam_cls = lcfg.get('lambda_cls',  1.0)
        self.lam_box = lcfg.get('lambda_box',  2.5)
        self.lam_dfl = lcfg.get('lambda_dfl',  0.5)
        self.alpha = lcfg.get('focal_alpha', 0.25)
        self.gamma = lcfg.get('focal_gamma', 2.0)
        self.reg_max = 32
        self.strides = [4, 8, 16, 32]
        radius = config.get('assigner', {}).get('center_sampling_radius', 1.5)
        self.assigner = FCOSAssigner(
            center_sampling_radius=radius, strides=self.strides)

    def _grids(self, H, W, device):
        gs, ss = [], []
        for stride in self.strides:
            sy = torch.arange(H // stride, dtype=torch.float32, device=device)
            sx = torch.arange(W // stride, dtype=torch.float32, device=device)
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            cx = sx.reshape(-1) * stride + stride / 2.0
            cy = sy.reshape(-1) * stride + stride / 2.0
            gs.append(torch.stack([cx, cy], dim=-1))
            ss.append(torch.full((len(cx), 1), stride,
                      dtype=torch.float32, device=device))
        return torch.cat(gs), torch.cat(ss)

    def _dfl(self, pred_dist, gt_dist):
        """Distribution Focal Loss — cross-entropy over two adjacent bins."""
        gt_dist = gt_dist.clamp(0.0, self.reg_max - 1 - 1e-4)
        y_l = gt_dist.floor().long()
        y_r = y_l + 1
        w_l = y_r.float() - gt_dist
        w_r = gt_dist - y_l.float()
        lp = F.log_softmax(pred_dist, dim=-1)
        idx = torch.arange(gt_dist.numel(), device=pred_dist.device)
        return -(w_l * lp[idx, y_l] + w_r * lp[idx, y_r]).sum()

    def forward(self, preds: list, targets: list, img_shape: tuple):
        H, W = img_shape
        device = preds[0][0].device
        B = len(targets)
        nc = preds[0][0].shape[1]

        cls_p = torch.cat(
            [p[0].permute(0, 2, 3, 1).reshape(B, -1, nc) for p in preds], 1)
        reg_p = torch.cat([p[1].permute(0, 2, 3, 1).reshape(
            B, -1, 4*self.reg_max) for p in preds], 1)

        grids, strides = self._grids(H, W, device)

        idx_l, box_l = [], []
        for b in range(B):
            res = self.assigner(targets[b]['boxes'], H, W)
            idx_l.append(torch.cat([r['assigned_indices'] for r in res]))
            box_l.append(torch.cat([r['assigned_boxes'] for r in res]))

        a_idx = torch.stack(idx_l)
        a_box = torch.stack(box_l)
        pos = a_idx >= 0
        K = pos.sum().item()

        if K == 0:
            warnings.warn("No positive samples in this batch.")
            return (cls_p * 0).sum() + (reg_p * 0).sum(), \
                torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

        pr_reg = reg_p[pos]                                    # (K, 4·reg_max)
        pg_box = a_box[pos]                                    # (K, 4)
        b_grid = grids.unsqueeze(0).expand(B, -1, -1)[pos]      # (K, 2)
        b_stride = strides.unsqueeze(0).expand(B, -1, -1)[pos]    # (K, 1)
        s = b_stride[:, 0]

        # Decode predicted boxes via DFL expectation
        pr_dfl = pr_reg.reshape(-1, 4, self.reg_max)
        sup = torch.arange(self.reg_max, dtype=torch.float32,
                           device=device).view(1, 1, -1)
        dist = (pr_dfl.softmax(-1) * sup).sum(-1)              # (K, 4)
        pred_boxes = torch.stack([
            b_grid[:, 0] - dist[:, 0]*s, b_grid[:, 1] - dist[:, 1]*s,
            b_grid[:, 0] + dist[:, 2]*s, b_grid[:, 1] + dist[:, 3]*s,
        ], dim=-1)

        # IoU-quality soft classification targets  (GFL / TOOD)
        with torch.no_grad():
            pb = pred_boxes.detach().clone()
            pb[:, [0, 2]] = pb[:, [0, 2]].clamp(0, W)
            pb[:, [1, 3]] = pb[:, [1, 3]].clamp(0, H)
            iou_q = box_iou(pb, pg_box).diag().clamp(0, 1)

        cls_tgt = torch.zeros_like(cls_p)
        cnt = 0
        for b in range(B):
            pm = pos[b]
            n_b = int(pm.sum())
            if n_b > 0:
                lbls = targets[b]['labels'][a_idx[b][pm]].long()
                cls_tgt[b, pm, lbls] = iou_q[cnt:cnt+n_b]
                cnt += n_b

        # ---- losses ----
        # sigmoid_focal_loss from torchvision replaces the hand-written focal method
        loss_cls = sigmoid_focal_loss(cls_p, cls_tgt,
                                      alpha=self.alpha, gamma=self.gamma, reduction='sum')
        loss_box = complete_box_iou_loss(pred_boxes, pg_box, reduction='sum')

        cx, cy = b_grid[:, 0], b_grid[:, 1]
        gt_dist = torch.stack([(cx-pg_box[:, 0])/s, (cy-pg_box[:, 1])/s,
                               (pg_box[:, 2]-cx)/s, (pg_box[:, 3]-cy)/s], dim=-1)
        loss_dfl = self._dfl(
            pr_dfl.reshape(-1, self.reg_max), gt_dist.reshape(-1))

        total = (self.lam_cls*loss_cls + self.lam_box *
                 loss_box + self.lam_dfl*loss_dfl) / K
        return total, loss_cls/K, loss_box/K, loss_dfl/K


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_bn_stats(m):
    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        m.eval()


def init_weights(model: nn.Module):
    """Kaiming init for Conv2d, constant for BN, focal-friendly bias on cls head."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    if hasattr(model, 'head'):
        for h in model.head.cls_heads:
            fc = h[-1]
            if isinstance(fc, nn.Conv2d):
                nn.init.normal_(fc.weight, std=0.01)
                if fc.bias is not None:
                    nn.init.constant_(fc.bias, math.log(0.01/0.99))
        for h in model.head.reg_heads:
            fc = h[-1]
            if isinstance(fc, nn.Conv2d):
                nn.init.normal_(fc.weight, std=0.01)
                if fc.bias is not None:
                    nn.init.constant_(fc.bias, 0.0)


def collate_fn(batch):
    imgs, tgts = [], []
    for img, boxes, labels in batch:
        imgs.append(img)
        tgts.append({'boxes': boxes, 'labels': labels})
    return torch.stack(imgs), tgts


def train(config_path: str, dataset_root: str,
          epochs: int = None, batch_size: int = None,
          max_train_samples: int = None, max_val_samples: int = None,
          checkpoint_dir: str = "checkpoints", log_dir: str = ".",
          grad_accumulation_steps: int = 1, resume_checkpoint: str = None):

    from data import DamageDataset, AugmentationWrapper
    from model import build_model
    from eval import evaluate

    config = load_config(config_path)
    seed_everything(config['training'].get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Datasets & loaders ---
    train_ds = DamageDataset(os.path.join(dataset_root, "train.txt"))
    val_ds = DamageDataset(os.path.join(dataset_root, "val.txt"))
    if max_train_samples and max_train_samples < len(train_ds):
        train_ds = Subset(train_ds, list(range(max_train_samples)))
    if max_val_samples and max_val_samples < len(val_ds):
        val_ds = Subset(val_ds,   list(range(max_val_samples)))

    train_ds = AugmentationWrapper(train_ds, config, is_train=True)
    val_ds = AugmentationWrapper(val_ds,   config, is_train=False)

    bs = batch_size or config['training'].get('batch_size', 32)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=2, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                            num_workers=2, collate_fn=collate_fn, pin_memory=True)

    # --- Model, loss, EMA ---
    model = build_model(config)
    init_weights(model)
    model = model.to(device)

    criterion = MRFDetLoss(config)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(
        config['training'].get('ema_decay', 0.9995)))

    # --- Optimizer + SequentialLR scheduler ---
    tcfg = config['training']
    base_lr = float(tcfg.get('base_lr',  1e-3))
    min_lr = float(tcfg.get('min_lr',   1e-6))
    ft_lr = float(tcfg.get('finetune_lr', 1e-5))
    warmup = tcfg.get('warmup_epochs', 5)
    ft_start = tcfg.get('finetune_start_epoch', 131)
    total_ep = epochs or tcfg.get('epochs', 150)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=tcfg.get('weight_decay', 5e-4))

    # Phase 1: linear warmup from 1e-5 → base_lr  (warmup epochs)
    # Phase 2: cosine decay base_lr → min_lr       (cosine epochs)
    # Phase 3: constant finetune_lr                 (remaining epochs)
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=1e-5/base_lr,
                 end_factor=1.0, total_iters=warmup),
        CosineAnnealingLR(optimizer, T_max=ft_start-warmup-1, eta_min=min_lr),
        ConstantLR(optimizer, factor=ft_lr/base_lr,
                   total_iters=total_ep-ft_start+1),
    ], milestones=[warmup, ft_start-1])

    # --- Optional resume ---
    start_epoch, best_val_map = 1, 0.0
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device)

        def _shape_load(module, state):
            cur = module.state_dict()
            ok = {k: v for k, v in state.items() if k in cur and v.shape ==
                  cur[k].shape}
            module.load_state_dict(ok, strict=False)
            return len(ok)
        print(f"Resumed — model: {_shape_load(model, ckpt['model_state_dict'])} params, "
              f"EMA: {_shape_load(ema, ckpt['ema_state_dict'])} params")
        start_epoch = ckpt['epoch'] + 1
        best_val_map = ckpt.get('best_val_map', 0.0)

    # --- Logging ---
    os.makedirs(log_dir,        exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_log.csv")
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ["epoch", "lr", "loss", "cls", "box", "dfl", "val_map50", "val_recall50"])

    patience = tcfg.get('early_stop_patience', 20)
    no_improve = 0

    def _save(name):
        torch.save({'model_state_dict': model.state_dict(),
                    'ema_state_dict':   ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch, 'best_val_map': best_val_map, 'config': config},
                   os.path.join(checkpoint_dir, name))

    # --- Training loop ---
    for epoch in range(start_epoch, total_ep + 1):
        lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch}/{total_ep}  lr={lr:.2e}")

        if epoch >= ft_start:
            train_ds.mosaic_prob = train_ds.mixup_prob = 0.0
            model.apply(freeze_bn_stats)
        else:
            model.train()

        el = ec = eb = ed = 0.0
        optimizer.zero_grad()
        for step, (imgs, tgts) in enumerate(pbar := tqdm(train_loader, desc=f"Epoch {epoch}")):
            imgs = imgs.to(device)
            tgts = [{'boxes': t['boxes'].to(
                device), 'labels': t['labels'].to(device)} for t in tgts]

            loss, lc, lb, ld = criterion(model(imgs), tgts, imgs.shape[2:])
            (loss / grad_accumulation_steps).backward()
            el += loss.item()
            ec += lc.item()
            eb += lb.item()
            ed += ld.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", cls=f"{lc.item():.3f}",
                             box=f"{lb.item():.3f}", dfl=f"{ld.item():.3f}")

            if (step + 1) % grad_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                ema.update_parameters(model)
                optimizer.zero_grad()

        scheduler.step()

        n = len(train_loader)
        avg = el/n
        ac = ec/n
        ab = eb/n
        ad = ed/n
        print(f"  Loss:{avg:.4f}  Cls:{ac:.4f}  Box:{ab:.4f}  DFL:{ad:.4f}")

        do_val = (epoch >= ft_start) or (epoch % 5 == 0)
        vm = vr = -1.0
        if do_val:
            m = evaluate(model, val_loader, config, ema=ema)
            vm, vr = m['map50'], m['recall50']
            if vm > best_val_map:
                best_val_map = vm
                no_improve = 0
                _save("checkpoint_best.pth")
                print("  Best checkpoint saved.")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping.")
                    break

        if epoch % 10 == 0:
            _save(f"checkpoint_epoch_{epoch}.pth")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, lr, avg, ac, ab, ad,
                                    vm if do_val else "", vr if do_val else ""])

    _save("checkpoint_final.pth")
    print("Training complete.")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config',         default='config.yaml')
    p.add_argument('--dataset_root',   default='../coco_dataset')
    p.add_argument('--epochs',         type=int, default=None)
    p.add_argument('--batch_size',     type=int, default=None)
    p.add_argument('--grad_accum',     type=int, default=1)
    p.add_argument('--max_train',      type=int, default=None)
    p.add_argument('--max_val',        type=int, default=None)
    p.add_argument('--checkpoint_dir', default='checkpoints')
    p.add_argument('--log_dir',        default='.')
    p.add_argument('--resume',         default=None)
    a = p.parse_args()
    train(a.config, a.dataset_root, epochs=a.epochs, batch_size=a.batch_size,
          max_train_samples=a.max_train, max_val_samples=a.max_val,
          checkpoint_dir=a.checkpoint_dir, log_dir=a.log_dir,
          grad_accumulation_steps=a.grad_accum, resume_checkpoint=a.resume)
