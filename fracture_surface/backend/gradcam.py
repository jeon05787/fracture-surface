"""
gradcam.py — GradCAM++ 및 마스크/윤곽선 유틸리티
=================================================
"""

import io
import base64

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import CLASS_NAMES, CLASS_COLORS_BGR


# ═══════════════════════════════════════════════════════
# GradCAM++
# ═══════════════════════════════════════════════════════

class GradCAMPlusPlus:
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

    def generate_for_class(self, x, class_idx):
        self.model.eval()
        self.model.zero_grad()
        out      = self.model(x)
        probs    = torch.softmax(out, dim=1)[0]
        out[0, class_idx].backward(retain_graph=False)
        grads    = self.gradients
        acts     = self.activations.detach()
        grads_sq = grads ** 2
        grads_cu = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        denom    = 2 * grads_sq + sum_acts * grads_cu
        denom    = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha    = grads_sq / denom
        weights  = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam      = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam      = cam.squeeze().cpu().numpy()
        cam_max  = cam.max()
        cam      = (cam - cam.min()) / (cam_max - cam.min() + 1e-8) if cam_max > 1e-8 else np.zeros_like(cam)
        return cam, probs.detach().cpu().numpy()

    def generate_all_classes(self, x, num_classes=4):
        cams, probs_out = {}, None
        for idx in range(num_classes):
            x_in = x.clone().detach().requires_grad_(True)
            cam, probs = self.generate_for_class(x_in, idx)
            cams[idx]  = cam
            probs_out  = probs
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