import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth


class MRFBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation_seq: list, drop_path: float = 0.0):
        super().__init__()
        self.dw_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=d,
                          dilation=d, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch), nn.GELU()
            ) for d in dilation_seq
        ])
        self.pw_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
        )
        self.proj = nn.Conv2d(
            in_ch, out_ch, 1, bias=True) if in_ch != out_ch else nn.Identity()
        self.sdrop = StochasticDepth(
            p=drop_path, mode="row") if drop_path > 0 else nn.Identity()

    def forward(self, x):
        skip = self.proj(x)
        for dw in self.dw_convs:
            x = dw(x)
        return self.sdrop(self.pw_conv(x)) + skip


class MRFBackbone(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        m = config['model']
        stem_ch = m.get('stem_channels', 32)
        stages_cfg = m['stages']
        max_drop = config['training'].get('stochastic_depth_rate', 0.15)

        self.stem = nn.Sequential(
            nn.Conv2d(m.get('in_channels', 3), stem_ch,
                      6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(stem_ch), nn.GELU()
        )

        total_blks = sum(s['blocks'] for s in stages_cfg)
        self.stages, cur_ch, idx = nn.ModuleList(), stem_ch, 0

        for i, scfg in enumerate(stages_cfg):
            layers, next_ch = [], scfg['channels']
            if i > 0:
                layers += [nn.Sequential(
                    nn.Conv2d(cur_ch, next_ch, 3, stride=2,
                              padding=1, bias=False),
                    nn.BatchNorm2d(next_ch), nn.GELU()
                )]
                cur_ch = next_ch
            for j in range(scfg['blocks']):
                rate = (idx / (total_blks - 1)) * \
                    max_drop if total_blks > 1 else 0.0
                in_c = stem_ch if (i == 0 and j == 0) else cur_ch
                layers.append(
                    MRFBlock(in_c, next_ch, scfg['dilation_seq'], drop_path=rate))
                cur_ch = next_ch
                idx += 1
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x):
        x = self.stages[0](self.stem(x))
        p2 = self.stages[1](x)
        p3 = self.stages[2](p2)
        p4 = self.stages[3](p3)
        p5 = self.stages[4](p4)
        return {'P2': p2, 'P3': p3, 'P4': p4, 'P5': p5}


class MRFFPNNeck(nn.Module):
    """
    Top-down FPN: 4 lateral 1×1 projections + 3 MRFBlock fusions.
    Returns {N2, N3, N4, N5}.
    """

    def __init__(self, in_channels_dict: dict, out_ch: int = 128):
        super().__init__()

        def lat(c): return nn.Sequential(
            nn.Conv2d(c, out_ch, 1, bias=False), nn.BatchNorm2d(
                out_ch), nn.GELU()
        )
        self.lat2 = lat(in_channels_dict['P2'])
        self.lat3 = lat(in_channels_dict['P3'])
        self.lat4 = lat(in_channels_dict['P4'])
        self.lat5 = lat(in_channels_dict['P5'])
        self.fn4 = MRFBlock(out_ch*2, out_ch, [1, 2, 6])
        self.fn3 = MRFBlock(out_ch*2, out_ch, [1, 2, 4])
        self.fn2 = MRFBlock(out_ch*2, out_ch, [1, 2, 4])

    def _up(self, x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)

    def forward(self, feats):
        l2, l3, l4 = self.lat2(feats['P2']), self.lat3(
            feats['P3']), self.lat4(feats['P4'])
        n5 = self.lat5(feats['P5'])
        n4 = self.fn4(torch.cat([self._up(n5, l4), l4], 1))
        n3 = self.fn3(torch.cat([self._up(n4, l3), l3], 1))
        n2 = self.fn2(torch.cat([self._up(n3, l2), l2], 1))
        return {'N2': n2, 'N3': n3, 'N4': n4, 'N5': n5}


class DecoupledHead(nn.Module):

    def __init__(self, in_ch: int = 128, num_classes: int = 1,
                 num_scales: int = 4, dropout: float = 0.1, reg_max: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        def _branch(out_ch, use_drop=False):
            layers = [nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False), nn.BatchNorm2d(in_ch), nn.GELU(),
                      nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False), nn.BatchNorm2d(in_ch), nn.GELU()]
            if use_drop and dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))
            layers.append(nn.Conv2d(in_ch, out_ch, 1, bias=True))
            return nn.Sequential(*layers)

        self.cls_heads = nn.ModuleList(
            [_branch(num_classes, use_drop=True) for _ in range(num_scales)])
        self.reg_heads = nn.ModuleList(
            [_branch(4 * reg_max) for _ in range(num_scales)])

    def forward(self, neck):
        return [(self.cls_heads[i](neck[s]), self.reg_heads[i](neck[s]))
                for i, s in enumerate(['N2', 'N3', 'N4', 'N5'])]


def _make_grids(strides, H, W, device):
    gs, ss = [], []
    for stride in strides:
        sy = torch.arange(H // stride, dtype=torch.float32, device=device)
        sx = torch.arange(W // stride, dtype=torch.float32, device=device)
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        cx = sx.reshape(-1) * stride + stride / 2.0
        cy = sy.reshape(-1) * stride + stride / 2.0
        gs.append(torch.stack([cx, cy], -1))
        ss.append(torch.full((len(cx), 1), stride,
                  dtype=torch.float32, device=device))
    return gs, ss


class MRFDet(nn.Module):
    """
    MRFDet = MRFBackbone + MRFFPNNeck + DecoupledHead.
    Training  → raw (cls_logits, reg_dist) predictions.
    Inference → list of {boxes, scores, labels} dicts with NMS applied.
    """
    STRIDES = [4, 8, 16, 32]

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        stages = config['model']['stages']
        neck_ch = config['model'].get('neck_channels', 128)
        self.reg_max = 32

        self.backbone = MRFBackbone(config)
        self.neck = MRFFPNNeck(
            {f'P{i+1}': stages[i]['channels'] for i in range(1, 5)}, neck_ch
        )
        self.head = DecoupledHead(
            in_ch=neck_ch,
            num_classes=config['model'].get('num_classes', 1),
            num_scales=4,
            dropout=config['training'].get('dropout_rate', 0.1),
            reg_max=self.reg_max,
        )

    def _decode_boxes(self, reg_pred, grid, stride):
        sup = torch.arange(self.reg_max, dtype=torch.float32,
                           device=reg_pred.device)
        dist = (reg_pred.view(-1, 4, self.reg_max).softmax(-1)
                * sup).sum(-1)  # (N, 4)
        cx, cy = grid[:, 0], grid[:, 1]
        return torch.stack([cx - dist[:, 0]*stride, cy - dist[:, 1]*stride,
                            cx + dist[:, 2]*stride, cy + dist[:, 3]*stride], -1)

    def forward(self, x: torch.Tensor, conf_threshold: float = None) -> list:
        preds = self.head(self.neck(self.backbone(x)))
        if self.training:
            return preds

        from eval import soft_nms, hard_nms
        B, _, H, W = x.shape
        device = x.device
        ecfg = self.config.get('eval', {})
        thresh = conf_threshold if conf_threshold is not None else ecfg.get(
            'conf_threshold', 0.25)
        iou_thr = ecfg.get('nms_iou_threshold', 0.5)
        nms_fn = soft_nms if ecfg.get(
            'nms_type', 'soft') == 'soft' else hard_nms
        grids, _ = _make_grids(self.STRIDES, H, W, device)

        out = []
        for b in range(B):
            boxes_all, scores_all, labels_all = [], [], []

            for lvl, (cls_pred, reg_pred) in enumerate(preds):
                stride = self.STRIDES[lvl]
                cls_flat = cls_pred[b].permute(
                    1, 2, 0).reshape(-1, self.head.num_classes)
                reg_flat = reg_pred[b].permute(
                    1, 2, 0).reshape(-1, 4*self.reg_max)
                scores = cls_flat.sigmoid()
                boxes = self._decode_boxes(reg_flat, grids[lvl], stride).clamp(
                    min=torch.tensor(
                        [0, 0, 0, 0], device=device, dtype=torch.float32),
                    max=torch.tensor([W, H, W, H], device=device, dtype=torch.float32))

                for c in range(self.head.num_classes):
                    keep = scores[:, c] > thresh
                    if keep.any():
                        boxes_all.append(boxes[keep])
                        scores_all.append(scores[keep, c])
                        labels_all.append(torch.full(
                            (keep.sum(),), c, dtype=torch.long, device=device))

            if not boxes_all:
                out.append({'boxes':  torch.zeros((0, 4), device=device),
                            'scores': torch.zeros((0,),  device=device),
                            'labels': torch.zeros((0,),  dtype=torch.long, device=device)})
                continue

            bc = torch.cat(boxes_all)
            sc = torch.cat(scores_all)
            lc = torch.cat(labels_all)
            if sc.numel() > 3000:
                _, top = sc.topk(3000)
                bc, sc, lc = bc[top], sc[top], lc[top]

            kb, ks, kl = nms_fn(
                bc, sc, lc, iou_threshold=iou_thr, score_threshold=thresh)
            out.append({'boxes': kb, 'scores': ks, 'labels': kl})

        return out


def build_model(config: dict) -> MRFDet:
    return MRFDet(config)
