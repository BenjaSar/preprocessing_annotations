"""
CubiCasa5K empirical probe — STEP 2 go/no-go gate.

Runs room-seg [21:33] + icon-seg [33:44] on real floorplan images.
Saves color overlays so human can verify channel→class semantics.

Usage:
    python _cubicasa_probe.py
Output:
    _cubicasa_probe_out/  — one overlay per image per slice
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import cv2
from pathlib import Path
from preprocessing_annotations.detection.vendor.seg_model import hg_furukawa_original

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "models" / "cubicasa5k_model.pkl"
OUT_DIR    = Path(__file__).parent / "_cubicasa_probe_out"
MAX_SIDE   = 1024

IMAGES = [
    # commercial MEP/roof — the hard case
    "/home/ubuntu/floorplan_classifier/VLM/fixA_cocofix_190747/images/326 ROCKAWAY - AVI-ON LAYOUT_page005.png",
    "/home/ubuntu/floorplan_classifier/VLM/fixA_cocofix_190747/images/326 ROCKAWAY - AVI-ON LAYOUT_page003.png",
    # commercial office suite
    "/home/ubuntu/floorplan_classifier/VLM/fixA_cocofix_190747/images/540 Madison Avenue Divcowest Suite 29A - AVI-ON LAYOUT_page000.png",
]

# Room classes (12): CubiCasa5K reduced set — labelled by index for verification.
# Overlay shows index; human confirms which index = which room type.
ROOM_COLORS = [
    (0,   0,   0  ),  # 0 Background
    (34,  139, 34 ),  # 1 Outdoor
    (128, 128, 128),  # 2 Wall
    (255, 165, 0  ),  # 3 Kitchen
    (0,   0,   255),  # 4 Living Room
    (139, 0,   0  ),  # 5 Bed Room
    (0,   255, 255),  # 6 Bath
    (255, 255, 0  ),  # 7 Entry
    (160, 82,  45 ),  # 8 Railing
    (148, 0,   211),  # 9 Storage
    (105, 105, 105),  # 10 Garage
    (255, 192, 203),  # 11 Undefined
]

# Icon classes (11): CubiCasa5K reduced set — labelled by index for verification.
ICON_COLORS = [
    (0,   0,   0  ),  # 0 Empty
    (0,   255, 0  ),  # 1 Window   (bright green)
    (255, 0,   0  ),  # 2 Door     (red)
    (0,   0,   255),  # 3 Closet   (blue)
    (255, 255, 0  ),  # 4 Electrical Appliance
    (255, 0,   255),  # 5 Toilet   (magenta)
    (0,   255, 255),  # 6 Sink     (cyan)
    (255, 128, 0  ),  # 7 Sauna Bench
    (128, 0,   128),  # 8 Fire Place
    (0,   128, 255),  # 9 Bathtub  (light blue)
    (255, 64,  64 ),  # 10 Chimney
]

ROOM_NAMES = ["Background","Outdoor","Wall","Kitchen","LivingRoom",
              "BedRoom","Bath","Entry","Railing","Storage","Garage","Undefined"]
ICON_NAMES = ["Empty","Window","Door","Closet","ElecApp",
              "Toilet","Sink","SaunaBench","FirePlace","Bathtub","Chimney"]


def load_model(device):
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    m = hg_furukawa_original(n_classes=44)
    missing, unexpected = m.load_state_dict(ckpt["model_state"], strict=True)
    assert not missing and not unexpected, f"bad load: {missing} {unexpected}"
    m.eval().to(device)
    print(f"Model loaded (epoch={ckpt['epoch']})")
    return m


def preprocess(img_rgb, max_side):
    h, w = img_rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img_rgb = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    img = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std  = np.array([0.229, 0.224, 0.225], np.float32)
    img  = (img - mean) / std
    return torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0), img_rgb.shape[:2]


def apply_colormap(argmax_hw, colors, orig_hw, inf_hw):
    """argmax_hw: (H_inf, W_inf) int. Returns BGR (H_orig, W_orig, 3)."""
    colored = np.zeros((*argmax_hw.shape, 3), dtype=np.uint8)
    for idx, rgb in enumerate(colors):
        colored[argmax_hw == idx] = rgb[::-1]  # RGB→BGR
    if inf_hw != orig_hw:
        colored = cv2.resize(colored, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_NEAREST)
    return colored


def overlay(base_bgr, color_bgr, alpha=0.5):
    mask = color_bgr.sum(axis=2) > 0  # skip background (0,0,0)
    out = base_bgr.copy()
    out[mask] = (alpha * color_bgr[mask] + (1-alpha) * base_bgr[mask]).astype(np.uint8)
    return out


def draw_legend(img, names, colors, title, x0=10, y0=10):
    for i, (name, rgb) in enumerate(zip(names, colors)):
        y = y0 + i * 22
        bgr = rgb[::-1]
        cv2.rectangle(img, (x0, y), (x0+16, y+16), bgr, -1)
        cv2.putText(img, f"{i}:{name}", (x0+20, y+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, f"{i}:{name}", (x0+20, y+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(img, title, (x0, y0-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    return img


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    OUT_DIR.mkdir(exist_ok=True)
    model = load_model(device)

    for img_path in IMAGES:
        p = Path(img_path)
        if not p.exists():
            print(f"SKIP (not found): {p.name}")
            continue

        base_bgr = cv2.imread(str(p))
        img_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
        orig_hw = img_rgb.shape[:2]

        tensor, inf_hw = preprocess(img_rgb, MAX_SIDE)
        tensor = tensor.to(device)

        with torch.no_grad():
            out = model(tensor)  # (1, 44, H', W')

        out_np = out[0].cpu().numpy()  # (44, H', W')

        # Room segmentation: softmax over [21:33], argmax
        room_logits = out_np[21:33]  # (12, H', W')
        room_probs  = np.exp(room_logits - room_logits.max(0))
        room_probs /= room_probs.sum(0)
        room_argmax = room_probs.argmax(0)  # (H', W')

        # Icon segmentation: softmax over [33:44], argmax
        icon_logits = out_np[33:44]  # (11, H', W')
        icon_probs  = np.exp(icon_logits - icon_logits.max(0))
        icon_probs /= icon_probs.sum(0)
        icon_argmax = icon_probs.argmax(0)  # (H', W')

        stem = p.stem
        print(f"\n{stem}")
        print(f"  orig {orig_hw}  inf {inf_hw}")

        # Room overlay
        room_col = apply_colormap(room_argmax, ROOM_COLORS, orig_hw, inf_hw)
        room_ov  = overlay(base_bgr, room_col, alpha=0.55)
        draw_legend(room_ov, ROOM_NAMES, ROOM_COLORS, "ROOMS [21:33]")
        cv2.imwrite(str(OUT_DIR / f"{stem}_room_seg.png"), room_ov)

        # Icon overlay — only non-zero (skip Empty class 0)
        icon_col = apply_colormap(icon_argmax, ICON_COLORS, orig_hw, inf_hw)
        # icon_argmax is at inf_hw; mask after upsampling by zeroing Empty color
        icon_col[np.all(icon_col == np.array(ICON_COLORS[0][::-1], dtype=np.uint8), axis=2)] = 0
        # For icons, blend harder so they show on top
        icon_ov = overlay(base_bgr, icon_col, alpha=0.7)
        draw_legend(icon_ov, ICON_NAMES, ICON_COLORS, "ICONS [33:44]")
        cv2.imwrite(str(OUT_DIR / f"{stem}_icon_seg.png"), icon_ov)

        # Per-class coverage report
        total_px = room_argmax.size
        for idx, name in enumerate(ROOM_NAMES):
            pct = 100.0 * (room_argmax == idx).sum() / total_px
            if pct > 1.0:
                print(f"  ROOM  ch{21+idx:02d} {idx:2d} {name:12s} {pct:5.1f}%")
        for idx, name in enumerate(ICON_NAMES):
            pct = 100.0 * (icon_argmax == idx).sum() / total_px
            if pct > 0.5:
                print(f"  ICON  ch{33+idx:02d} {idx:2d} {name:12s} {pct:5.1f}%")

        print(f"  → {stem}_room_seg.png")
        print(f"  → {stem}_icon_seg.png")

    print(f"\nOverlays in: {OUT_DIR}")
    print("GO/NO-GO: inspect overlays. If room-seg fits space boundaries → proceed. If noise → stop.")


if __name__ == "__main__":
    run()
