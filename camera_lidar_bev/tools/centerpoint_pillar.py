"""Standalone CenterPoint-PointPillar (OpenPCDet cbgs_dyn_pp_centerpoint, nuScenes).

Pure PyTorch: no OpenPCDet CUDA extensions or torch_scatter.

* ``forward_dynamic``: exact DynamicPillarVFE semantics (every point in a pillar),
  used as the reference.
* ``bev_heads``: backbone and CenterHead on the dense [1,64,512,512] pillar canvas.
  This is the part exported to ONNX/TensorRT; the runtime computes the pillar
  features with a CUDA kernel (bev_lidar_detection/src/centerpoint_vfe.cu) from
  ``pfn_weights`` and scatters them into the canvas.

Coordinates follow the nuScenes LIDAR_TOP frame the model was trained in:
x right, y forward, z up. ``ouster_to_model`` converts from os_sensor
(x forward, y left, z up).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
               'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
HEAD_CLASSES = [['car'], ['truck', 'construction_vehicle'], ['bus', 'trailer'],
                ['barrier'], ['motorcycle', 'bicycle'], ['pedestrian', 'traffic_cone']]
HEAD_ORDER = ['center', 'center_z', 'dim', 'rot', 'vel']
HEAD_CHANNELS = {'center': 2, 'center_z': 1, 'dim': 3, 'rot': 2, 'vel': 2}
REG_CHANNELS = 10  # center(2) z(1) log-dims(3) cos sin vel(2); exp is applied in decode (FP16-safe)

PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
VOXEL_SIZE = (0.2, 0.2, 8.0)
GRID_X = GRID_Y = 512
FEATURE_STRIDE = 4
NUM_POINT_FEATURES = 5  # x, y, z, intensity (0..255), sweep time lag (s)
NUM_PFN_INPUT = NUM_POINT_FEATURES + 6

SCORE_THRESH = 0.1
POST_CENTER_LIMIT = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)
TOPK_PER_HEAD = 500
NMS_IOU = 0.2
NMS_PRE_MAX = 1000
NMS_POST_MAX = 83


class PFN(nn.Module):
    def __init__(self, cin: int, cout: int, last: bool):
        super().__init__()
        self.last = last
        self.linear = nn.Linear(cin, cout if last else cout // 2, bias=False)
        self.norm = nn.BatchNorm1d(cout if last else cout // 2, eps=1e-3, momentum=0.01)

    def point_features(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        y = self.norm(self.linear(x).reshape(-1, self.linear.out_features))
        return F.relu(y).reshape(*shape[:-1], -1)


class BEVBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        nums, strides, filters = [3, 5, 5], [2, 2, 2], [64, 128, 256]
        cin = [64, 64, 128]
        self.blocks = nn.ModuleList()
        for i in range(3):
            layers = [nn.ZeroPad2d(1), nn.Conv2d(cin[i], filters[i], 3, strides[i], 0, bias=False),
                      nn.BatchNorm2d(filters[i], eps=1e-3, momentum=0.01), nn.ReLU()]
            for _ in range(nums[i]):
                layers += [nn.Conv2d(filters[i], filters[i], 3, padding=1, bias=False),
                           nn.BatchNorm2d(filters[i], eps=1e-3, momentum=0.01), nn.ReLU()]
            self.blocks.append(nn.Sequential(*layers))

        def bn_relu():
            return [nn.BatchNorm2d(128, eps=1e-3, momentum=0.01), nn.ReLU()]
        self.deblocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, 128, 2, stride=2, bias=False), *bn_relu()),
            nn.Sequential(nn.ConvTranspose2d(128, 128, 1, stride=1, bias=False), *bn_relu()),
            nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False), *bn_relu()),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ups = []
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            ups.append(deblock(x))
        return torch.cat(ups, dim=1)


class SeparateHead(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        for name, cout in [*HEAD_CHANNELS.items(), ('hm', num_classes)]:
            setattr(self, name, nn.Sequential(
                nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=True), nn.BatchNorm2d(64), nn.ReLU()),
                nn.Conv2d(64, cout, 3, padding=1, bias=True)))


class CenterPointPillar(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn_layers = nn.ModuleList([PFN(NUM_PFN_INPUT, 64, False), PFN(64, 64, True)])
        self.backbone_2d = BEVBackbone()
        self.shared_conv = nn.Sequential(nn.Conv2d(384, 64, 3, padding=1, bias=True), nn.BatchNorm2d(64), nn.ReLU())
        self.heads_list = nn.ModuleList([SeparateHead(len(c)) for c in HEAD_CLASSES])

    @classmethod
    def from_checkpoint(cls, path: str) -> 'CenterPointPillar':
        state = torch.load(path, map_location='cpu', weights_only=True)['model_state']
        renamed = {}
        for k, v in state.items():
            if k == 'global_step':
                continue
            k = k.replace('vfe.pfn_layers', 'pfn_layers').replace('dense_head.', '')
            renamed[k] = v
        model = cls()
        model.load_state_dict(renamed, strict=True)
        return model.eval()

    def pfn_weights(self) -> np.ndarray:
        """Eval-mode PFN with BatchNorm folded, flattened for the CUDA VFE kernel.

        Layout (float32): W1[32][11], scale1[32], shift1[32], W2[64][64], scale2[64], shift2[64].
        Layer 2 input is [h1 (32), pillar max of h1 (32)].
        """
        parts = []
        for pfn in self.pfn_layers:
            bn = pfn.norm
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            shift = bn.bias - bn.running_mean * scale
            parts += [pfn.linear.weight, scale, shift]
        return torch.cat([p.detach().reshape(-1).float() for p in parts]).numpy()

    def scatter(self, pillar_features: torch.Tensor, flat_index: torch.Tensor) -> torch.Tensor:
        """pillar_features [P,64], flat_index [P] = y_idx * GRID_X + x_idx -> canvas [1,64,512,512]."""
        canvas = pillar_features.new_zeros(64, GRID_Y * GRID_X)
        canvas[:, flat_index.long()] = pillar_features.t()
        return canvas.view(1, 64, GRID_Y, GRID_X)

    def bev_heads(self, canvas: torch.Tensor):
        x = self.shared_conv(self.backbone_2d(canvas))
        heatmaps, regs = [], []
        for head in self.heads_list:
            heatmaps.append(torch.sigmoid(head.hm(x)))
            rot = head.rot(x)
            regs.append(torch.cat([head.center(x), head.center_z(x), head.dim(x), rot, head.vel(x)], dim=1))
        # heatmap [1,10,H,W] in CLASS_NAMES order; reg [1,6*10,H,W] per head.
        return torch.cat(heatmaps, 1), torch.cat(regs, 1)

    # -- exact DynamicPillarVFE reference ------------------------------------------
    @torch.no_grad()
    def forward_dynamic(self, points: torch.Tensor) -> torch.Tensor:
        """points [N,5] (x, y, z, intensity, dt) in the model frame -> BEV canvas [1,64,512,512]."""
        rng = points.new_tensor(PC_RANGE)
        vs = points.new_tensor(VOXEL_SIZE)
        coords = torch.floor((points[:, :2] - rng[:2]) / vs[:2]).int()
        keep = ((coords >= 0) & (coords < GRID_X)).all(1)
        points, coords = points[keep], coords[keep]
        merge = coords[:, 0] * GRID_Y + coords[:, 1]
        unq, inv = torch.unique(merge, return_inverse=True)
        xyz = points[:, :3]
        count = torch.zeros(len(unq), device=xyz.device).index_add_(0, inv, torch.ones_like(xyz[:, 0]))
        mean = torch.zeros(len(unq), 3, device=xyz.device).index_add_(0, inv, xyz) / count[:, None]
        f_center = torch.stack([
            xyz[:, 0] - (coords[:, 0].float() * VOXEL_SIZE[0] + VOXEL_SIZE[0] / 2 + PC_RANGE[0]),
            xyz[:, 1] - (coords[:, 1].float() * VOXEL_SIZE[1] + VOXEL_SIZE[1] / 2 + PC_RANGE[1]),
            xyz[:, 2] - (VOXEL_SIZE[2] / 2 + PC_RANGE[2])], 1)
        x = torch.cat([points, xyz - mean[inv], f_center], 1)
        for pfn in self.pfn_layers:
            x = pfn.point_features(x)
            x_max = x.new_zeros(len(unq), x.shape[1]).scatter_reduce(
                0, inv[:, None].expand_as(x), x, 'amax', include_self=False)
            x = x_max if pfn.last else torch.cat([x, x_max[inv]], 1)
        x_idx, y_idx = unq // GRID_Y, unq % GRID_Y
        return self.scatter(x, y_idx * GRID_X + x_idx)


def ouster_to_model(xyz_sensor: np.ndarray, z_offset: float = 0.0) -> np.ndarray:
    """os_sensor (x fwd, y left, z up) -> nuScenes lidar (x right, y fwd, z up)."""
    return np.stack([-xyz_sensor[:, 1], xyz_sensor[:, 0], xyz_sensor[:, 2] + z_offset], 1)


def model_boxes_to_ouster(boxes: np.ndarray, z_offset: float = 0.0) -> np.ndarray:
    """[x,y,z,dx,dy,dz,yaw] model frame -> os_sensor; dx stays along the heading."""
    out = boxes.copy()
    out[:, 0], out[:, 1], out[:, 2] = boxes[:, 1], -boxes[:, 0], boxes[:, 2] - z_offset
    out[:, 6] = np.remainder(boxes[:, 6] - math.pi / 2 + math.pi, 2 * math.pi) - math.pi
    return out


def _rotated_corners(b: np.ndarray) -> np.ndarray:
    c, s = math.cos(b[6]), math.sin(b[6])
    hx, hy = b[3] / 2, b[4] / 2
    local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
    return local @ np.array([[c, s], [-s, c]]) + b[:2]


def _clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    out = subject
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        if len(out) == 0:
            break
        inp, out = out, []
        edge = b - a
        side = lambda p: edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])
        for j in range(len(inp)):
            cur, prev = inp[j], inp[j - 1]
            sc, sp = side(cur), side(prev)
            if sc >= 0:
                if sp < 0:
                    out.append(prev + (cur - prev) * (sp / (sp - sc)))
                out.append(cur)
            elif sp >= 0:
                out.append(prev + (cur - prev) * (sp / (sp - sc)))
        out = np.array(out)
    return out


def bev_iou(a: np.ndarray, b: np.ndarray) -> float:
    poly = _clip(_rotated_corners(a), _rotated_corners(b))
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    inter = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return inter / max(a[3] * a[4] + b[3] * b[4] - inter, 1e-9)


def decode(heatmap: np.ndarray, reg: np.ndarray):
    """OpenPCDet CenterHead.generate_predicted_boxes with class-agnostic BEV NMS per head.

    heatmap [10,H,W] sigmoid scores; reg [60,H,W]. Returns boxes [K,9]
    (x,y,z,dx,dy,dz,yaw,vx,vy), scores [K], labels [K] (CLASS_NAMES index).
    """
    H, W = heatmap.shape[1:]
    lim = np.array(POST_CENTER_LIMIT)
    all_boxes, all_scores, all_labels = [], [], []
    first = 0
    for h, names in enumerate(HEAD_CLASSES):
        hm = heatmap[first:first + len(names)].reshape(len(names), -1)
        r = reg[h * REG_CHANNELS:(h + 1) * REG_CHANNELS].reshape(REG_CHANNELS, -1)
        flat = hm.reshape(-1)
        k = min(TOPK_PER_HEAD, flat.size)
        top = np.argpartition(-flat, k - 1)[:k]
        top = top[np.argsort(-flat[top], kind='stable')]
        scores = flat[top]
        cls, pix = top // (H * W), top % (H * W)
        ys, xs = (pix // W).astype(np.float32), (pix % W).astype(np.float32)
        xs = (xs + r[0, pix]) * FEATURE_STRIDE * VOXEL_SIZE[0] + PC_RANGE[0]
        ys = (ys + r[1, pix]) * FEATURE_STRIDE * VOXEL_SIZE[1] + PC_RANGE[1]
        dims = np.exp(np.minimum(r[3:6, pix], 10.0))
        boxes = np.stack([xs, ys, r[2, pix], dims[0], dims[1], dims[2],
                          np.arctan2(r[7, pix], r[6, pix]), r[8, pix], r[9, pix]], 1)
        ok = (boxes[:, :3] >= lim[:3]).all(1) & (boxes[:, :3] <= lim[3:]).all(1) & (scores > SCORE_THRESH)
        boxes, scores, cls = boxes[ok][:NMS_PRE_MAX], scores[ok][:NMS_PRE_MAX], cls[ok][:NMS_PRE_MAX]
        keep = []
        suppressed = np.zeros(len(boxes), bool)
        for i in range(len(boxes)):
            if suppressed[i]:
                continue
            keep.append(i)
            if len(keep) >= NMS_POST_MAX:
                break
            for j in range(i + 1, len(boxes)):
                if not suppressed[j] and bev_iou(boxes[i], boxes[j]) > NMS_IOU:
                    suppressed[j] = True
        all_boxes.append(boxes[keep])
        all_scores.append(scores[keep])
        all_labels.append(cls[keep] + first)
        first += len(names)
    return np.concatenate(all_boxes), np.concatenate(all_scores), np.concatenate(all_labels)
