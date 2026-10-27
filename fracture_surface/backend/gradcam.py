"""
gradcam.py — 슬라이딩 윈도우 추론 + GradCAM++ 및 마스크/윤곽선 유틸리티
=====================================================================
train_kaggle.py(슬라이딩 윈도우 버전)와 같은 방식으로 추론한다.
  1) 원본을 patch_size 정사각 격자(stride)로 자른다
  2) 각 패치를 img_size 로 리사이즈해 모델에 넣는다
  3) 분류 확률 = 패치 softmax 확률의 평균
  4) Grad-CAM++ = 패치별 CAM(12×12)을 원본 좌표로 되돌려 겹치는 부분은 평균,
     클래스별로 '이미지 전체 기준' 한 번만 0~1 정규화
     (패치마다 정규화하면 근거가 없는 패치도 최댓값 1로 부풀려지므로)
"""

import io
import base64

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import CLASS_NAMES, CLASS_COLORS_BGR


# ═══════════════════════════════════════════════════════
# 슬라이딩 윈도우
# ═══════════════════════════════════════════════════════

def window_boxes(W, H, patch, stride):
    """원본 (W,H)를 덮는 패치 좌표 [(x, y, w, h)].
    patch=None 이면 이미지 전체 1장 (기존 224px 가중치 호환용)."""
    if not patch:
        return [(0, 0, W, H)]
    side = min(patch, W, H)

    def starts(L):
        if L <= side:
            return [0]
        s = list(range(0, L - side + 1, stride))
        if s[-1] != L - side:
            s.append(L - side)
        return s

    return [(x, y, side, side) for y in starts(H) for x in starts(W)]


def build_patch_transform(cfg):
    return transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])


def make_patches(image, cfg, transform=None):
    """PIL 이미지 → (boxes, (N,3,S,S) 텐서)"""
    transform = transform or build_patch_transform(cfg)
    W, H = image.size
    boxes = window_boxes(W, H, cfg.get("patch_size"), cfg.get("stride"))
    x = torch.stack([transform(image.crop((bx, by, bx + bw, by + bh)))
                     for bx, by, bw, bh in boxes])
    return boxes, x


@torch.no_grad()
def predict_sliding(model, image, cfg, device, chunk=8):
    """반환: (평균 확률 (C,), 패치별 확률 (N,C), boxes)"""
    model.eval()
    boxes, x = make_patches(image, cfg)
    probs = []
    for k in range(0, len(x), chunk):
        out = model(x[k:k + chunk].to(device))
        probs.append(torch.softmax(out, dim=1).cpu())
    probs = torch.cat(probs).numpy()
    return probs.mean(axis=0), probs, boxes


# ═══════════════════════════════════════════════════════
# GradCAM++
# ═══════════════════════════════════════════════════════

class GradCAMPlusPlus:
    """
    target layer = model.aspp 출력 (B,256,12,12 @384px)
    주의: 훅이 인스턴스에 상태를 저장하므로 동시에 여러 요청이 들어오면
          threading.Lock 으로 감싸서 호출할 것.
    """
    def __init__(self, model):
        self.model       = model
        self.activations = None
        self.gradients   = None
        model.aspp.register_forward_hook(self._save_activation)
        model.aspp.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output

    def _save_gradient(self, _, __, grad_out):
        self.gradients = grad_out[0].detach()

    @staticmethod
    def _campp(acts, grads):
        """배치 단위 Grad-CAM++ (정규화 전 값). acts/grads: (B,C,h,w) → (B,h,w)"""
        grads_sq = grads ** 2
        grads_cu = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        denom    = 2 * grads_sq + sum_acts * grads_cu
        denom    = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha    = grads_sq / denom
        weights  = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        return F.relu((weights * acts).sum(dim=1))

    @staticmethod
    def _normalize(cam):
        lo, hi = float(cam.min()), float(cam.max())
        if hi < 1e-8:
            return np.zeros_like(cam, dtype=np.float32)
        return ((cam - lo) / (hi - lo + 1e-8)).astype(np.float32)

    def generate_sliding(self, image, class_indices, cfg, device, chunk=4):
        """
        image         : PIL RGB 원본
        class_indices : CAM 을 계산할 클래스 인덱스 목록 (CPU 에서는 상위 1~2개 권장)
        반환          : {class_idx: (H,W) float32 0~1}  ← 원본 해상도
        """
        self.model.eval()
        W, H = image.size
        boxes, x = make_patches(image, cfg)
        acc = {c: np.zeros((H, W), np.float32) for c in class_indices}
        cnt = np.zeros((H, W), np.float32)

        for k in range(0, len(x), chunk):
            xb = x[k:k + chunk].to(device)
            bboxes = boxes[k:k + chunk]
            out = self.model(xb)                       # 그래프 유지 (no_grad 아님)
            for j, c in enumerate(class_indices):
                self.model.zero_grad(set_to_none=True)
                # eval 모드에서는 샘플 간 상호작용이 없으므로 합을 미분해도 각 패치 기울기와 같음
                out[:, c].sum().backward(retain_graph=j < len(class_indices) - 1)
                cams = self._campp(self.activations.detach(), self.gradients).cpu().numpy()
                for cam, (bx, by, bw, bh) in zip(cams, bboxes):
                    acc[c][by:by + bh, bx:bx + bw] += cv2.resize(
                        cam.astype(np.float32), (bw, bh), interpolation=cv2.INTER_LINEAR)
            for bx, by, bw, bh in bboxes:
                cnt[by:by + bh, bx:bx + bw] += 1
            del out

        self.model.zero_grad(set_to_none=True)
        cnt = np.maximum(cnt, 1)
        return {c: self._normalize(acc[c] / cnt) for c in class_indices}

    # ── 기존 API (이미지 1장 텐서 입력) — 하위 호환용 ─────────────
    def generate_for_class(self, x, class_idx):
        self.model.eval()
        self.model.zero_grad()
        out   = self.model(x)
        probs = torch.softmax(out, dim=1)[0]
        out[0, class_idx].backward(retain_graph=False)
        cam = self._campp(self.activations.detach(), self.gradients)[0].cpu().numpy()
        return self._normalize(cam), probs.detach().cpu().numpy()

    def generate_all_classes(self, x, num_classes=4):
        cams, probs_out = {}, None
        for idx in range(num_classes):
            x_in = x.clone().detach().requires_grad_(True)
            cams[idx], probs_out = self.generate_for_class(x_in, idx)
        return cams, probs_out


# ═══════════════════════════════════════════════════════
# CAM → 마스크 / 윤곽선  (visualize.py 원본 그대로)
# ═══════════════════════════════════════════════════════

def cam_to_mask(cam, target_size, cam_percentile=80, min_area_ratio=0.005):
    if cam.max() < 1e-6:
        return np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    cam_resized = cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR)
    thresh_val  = np.percentile(cam_resized, cam_percentile)
    mask = (cam_resized >= thresh_val).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = mask.shape[0] * mask.shape[1] * min_area_ratio
    cleaned  = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def split_solo_overlap(masks_dict):
    count_map = np.zeros_like(next(iter(masks_dict.values())), dtype=np.int32)
    for m in masks_dict.values():
        count_map += (m > 0).astype(np.int32)
    solo, overlap = {}, {}
    for name, m in masks_dict.items():
        active        = (m > 0)
        solo[name]    = (active & (count_map == 1)).astype(np.uint8) * 255
        overlap[name] = (active & (count_map >= 2)).astype(np.uint8) * 255
    return solo, overlap


def draw_dashed_contour(img, contour, color_bgr, thickness=2, dash_length=10):
    pts = contour.reshape(-1, 2)
    n = len(pts)
    if n < 2:
        return
    i, draw_on = 0, True
    while i < n - 1:
        end_i = min(i + dash_length, n - 1)
        if draw_on:
            for j in range(i, end_i):
                cv2.line(img, tuple(pts[j]), tuple(pts[j + 1]),
                         color_bgr, thickness, lineType=cv2.LINE_AA)
        i = end_i
        draw_on = not draw_on


def build_contour_image(img_rgb, solo_masks, overlap_masks, active_classes=None):
    """visualize.py 원본 그대로. active_classes=None 이면 전체 표시."""
    canvas = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for name, mask in solo_masks.items():
        if active_classes is not None and name not in active_classes:
            continue
        if mask.sum() == 0:
            continue
        color = CLASS_COLORS_BGR[name]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            cv2.drawContours(canvas, [cnt], -1, color, thickness=3, lineType=cv2.LINE_AA)
    for name, mask in overlap_masks.items():
        if active_classes is not None and name not in active_classes:
            continue
        if mask.sum() == 0:
            continue
        color = CLASS_COLORS_BGR[name]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            draw_dashed_contour(canvas, cnt, color, thickness=3, dash_length=12)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def build_layer_rgba(img_rgb, name, solo_mask, overlap_mask):
    """클래스별 RGBA 레이어 (프론트 canvas 합성용)."""
    H, W   = img_rgb.shape[:2]
    color  = CLASS_COLORS_BGR[name]
    canvas = np.zeros((H, W, 4), dtype=np.uint8)

    def _draw(mask, dashed):
        if mask is None or mask.sum() == 0:
            return
        tmp = np.zeros((H, W, 3), dtype=np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if dashed:
                draw_dashed_contour(tmp, cnt, color, thickness=3, dash_length=12)
            else:
                cv2.drawContours(tmp, [cnt], -1, color, thickness=3, lineType=cv2.LINE_AA)
        drawn = np.any(tmp > 0, axis=2)
        canvas[drawn, 0] = color[2]
        canvas[drawn, 1] = color[1]
        canvas[drawn, 2] = color[0]
        canvas[drawn, 3] = 255

    _draw(solo_mask,    dashed=False)
    _draw(overlap_mask, dashed=True)
    return canvas


def extract_contours_json(masks_dict, img_shape):
    result = {}
    names  = list(masks_dict.keys())
    for name, mask in masks_dict.items():
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_list = []
        for cnt in cnts:
            pts = cnt.reshape(-1, 2).tolist()
            if len(pts) < 2:
                continue
            tmp = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(tmp, [cnt], -1, 255, thickness=cv2.FILLED)
            cnt_pixels    = tmp > 0
            overlaps_with = [
                other for other in names
                if other != name and bool(np.any(cnt_pixels & (masks_dict[other] > 0)))
            ]
            contours_list.append({"pts": pts, "overlaps_with": overlaps_with})
        result[name] = contours_list
    return result


def to_b64(img_np, fmt="PNG"):
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    b64  = base64.b64encode(buf.getvalue()).decode()
    mime = "png" if fmt == "PNG" else "jpeg"
    return f"data:image/{mime};base64,{b64}"

def _position_label(cx, cy, W, H):
    col = "좌측" if cx < W / 3 else ("우측" if cx >= 2 * W / 3 else "중앙")
    row = "상단" if cy < H / 3 else ("하단" if cy >= 2 * H / 3 else "중앙")

    if row == "중앙" and col == "중앙":
        return "중앙부"
    if row == "중앙":
        return f"{col} 영역"
    if col == "중앙":
        return f"{row} 영역"
    return f"{col} {row}"


def summarize_gradcam_regions(masks_dict, overlap_masks, probs_by_class, img_shape):
    """
    LLM에 전달할 Grad-CAM++ 영역 요약.
    masks_dict      : 검출된 클래스별 이진 마스크
    overlap_masks   : split_solo_overlap()의 겹침 마스크
    probs_by_class  : {"Cleavage": 0.12, ...} (softmax, 0~1)
    """
    H, W = img_shape[:2]
    total = H * W
    summary = {}

    for name, mask in masks_dict.items():
        active = mask > 0
        area = int(active.sum())
        if area == 0:
            continue

        ys, xs = np.nonzero(active)
        cx, cy = float(xs.mean()), float(ys.mean())

        # 활성 영역이 이미지에서 퍼진 정도 (bounding box 기준)
        bbox_ratio = ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / total

        overlap_area = int((overlap_masks.get(name, np.zeros_like(mask)) > 0).sum())

        overlaps_with = [
            other for other, m in masks_dict.items()
            if other != name and bool(np.any(active & (m > 0)))
        ]

        n_labels, _ = cv2.connectedComponents(mask, connectivity=8)

        summary[name] = {
            "probability": round(float(probs_by_class[name]) * 100, 1),
            "position": _position_label(cx, cy, W, H),
            "spread": "넓게 분포" if bbox_ratio >= 0.5 else "국부적으로 집중",
            "region_count": int(n_labels - 1),
            "overlap_ratio": round(overlap_area / area * 100, 1),
            "overlaps_with": overlaps_with,
        }

    return summary