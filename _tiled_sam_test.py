import cv2, json, numpy as np, torch, os
from segment_anything import sam_model_registry, SamPredictor

BASE = "/home/ubuntu/floorplan_classifier/VLM/dataset_test_fix28_verbose"
PAGE = os.environ.get("PAGE", "page003")
IMG = f"{BASE}/images/326 ROCKAWAY - AVI-ON LAYOUT_{PAGE}.png"
ANN = f"{BASE}/processed_annotations/326 ROCKAWAY - AVI-ON LAYOUT_{PAGE}.json"
CKPT = "/home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations/sam_vit_h_4b8939.pth"
OUT = "/home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations/_sam_overlays"
os.makedirs(OUT, exist_ok=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=CKPT).to(dev)
pred = SamPredictor(sam)
img = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]

rooms = [r for r in json.load(open(ANN))["rooms"]
         if (r.get("original_bbox") or r.get("bbox")) and
         (lambda b: b[2]*b[3] < 30000)(r.get("original_bbox") or r.get("bbox"))][:6]

CROPS = [1000, 1600, 2400]        # adaptive: grow on edge-touch
MAXFRAC = 0.4
print(f"{PAGE} | image {W}x{H} | {len(rooms)} labels | adaptive crops {CROPS}\n", flush=True)
clean = 0
for idx, r in enumerate(rooms):
    bx, by, bw, bh = r.get("original_bbox") or r["bbox"]
    cx, cy = bx + bw // 2, by + bh // 2
    nm = r.get("room_name", "?")
    chosen = None
    for crop in CROPS:
        half = crop // 2
        x0, y0 = max(0, cx-half), max(0, cy-half)
        x1, y1 = min(W, cx+half), min(H, cy+half)
        patch = np.ascontiguousarray(img[y0:y1, x0:x1])
        ph, pw = patch.shape[:2]
        pred.set_image(patch)
        masks, scores, _ = pred.predict(point_coords=np.array([[cx-x0, cy-y0]]),
                                        point_labels=np.array([1]), multimask_output=True)
        areas = [int(m.sum()) for m in masks]
        valid = [i for i, a in enumerate(areas) if 0 < a <= MAXFRAC*ph*pw]
        fallback = not valid
        bi = max(valid, key=lambda i: scores[i]) if valid else min(range(len(areas)), key=lambda i: areas[i])
        m = masks[bi]; ys, xs = np.where(m)
        edge = (xs.min() <= 1 or ys.min() <= 1 or xs.max() >= pw-2 or ys.max() >= ph-2)
        bbox_area = (xs.max()-xs.min()+1)*(ys.max()-ys.min()+1)
        chosen = (crop, bbox_area, fallback, edge, m, patch, x0, y0, scores[bi])
        if not edge and not fallback:
            break   # good mask, stop growing
    crop, bbox_area, fallback, edge, m, patch, x0, y0, sc = chosen
    v = "OK" if 40000 <= bbox_area <= 2_000_000 else ("tiny" if bbox_area < 40000 else "big")
    ok = (v == "OK" and not fallback and not edge)
    if ok: clean += 1
    flags = ("F" if fallback else "-") + ("E" if edge else "-")
    print(f"  {nm[:12]:12s} crop={crop} bbox={bbox_area:8d} sc={sc:.2f} {flags} {v} {'CLEAN' if ok else ''}", flush=True)
    # overlay for visual correctness check
    ov = patch.copy()
    ov[m] = (0.5*ov[m] + 0.5*np.array([255,0,0])).astype(np.uint8)
    cv2.imwrite(f"{OUT}/{PAGE}_{idx}_{nm[:8].replace('/','-')}.png",
                cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
print(f"\n{clean}/{len(rooms)} CLEAN (bbox-scale, no fallback, no edge)", flush=True)
print(f"overlays -> {OUT}", flush=True)
