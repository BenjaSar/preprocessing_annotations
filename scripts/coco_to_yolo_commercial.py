"""
One-off converter: commercial_building_obj_detection_coco_w_images (COCO,
11 images, door/double_door/window/etc) -> YOLO txt labels.

No existing COCO->YOLO converter exists in this repo (checked: eval/,
scripts/, export/exporters.py, orchestration/pipeline.py) -- this is
intentionally small and self-contained rather than a general-purpose tool.

Writes:
  <out_dir>/images/<file>.png   (copied)
  <out_dir>/labels/<file>.txt   (YOLO format: class_id xc yc w h, normalized)

Only door/double_door/window are kept (this detector's scope). bifold_door
folds into door (5 instances, too few for its own class). Everything else
(bathub, sink*, toilet, sliding_window, bypass_sliding_door) is dropped.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

CATEGORY_MAP = {
    "door": "door",
    "double_door": "double_door",
    "bifold_door": "door",
}

CLASS_IDS = {"door": 0, "window": 1, "zone": 2, "double_door": 3}


def convert(coco_path: Path, images_dir: Path, out_dir: Path, image_ids: set[int] | None = None) -> None:
    d = json.loads(coco_path.read_text())
    cat_names = {c["id"]: c["name"] for c in d["categories"]}
    images_by_id = {im["id"]: im for im in d["images"]}

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    anns_by_image: dict[int, list] = {}
    for ann in d["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    written = 0
    for img_id, im in images_by_id.items():
        if image_ids is not None and img_id not in image_ids:
            continue
        basename = Path(im["file_name"].replace("\\", "/")).name
        src_img = images_dir / basename
        if not src_img.exists():
            raise FileNotFoundError(f"GT references {basename} but it's not in {images_dir}")
        shutil.copy2(src_img, out_dir / "images" / basename)

        w, h = im["width"], im["height"]
        lines = []
        for ann in anns_by_image.get(img_id, []):
            cat = cat_names[ann["category_id"]]
            mapped = CATEGORY_MAP.get(cat)
            if mapped is None:
                continue
            cls_id = CLASS_IDS[mapped]
            x, y, bw, bh = ann["bbox"]
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nbw = bw / w
            nbh = bh / h
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nbw:.6f} {nbh:.6f}")

        label_path = out_dir / "labels" / (Path(basename).stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        written += 1

    print(f"wrote {written} image/label pairs to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coco", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--image-ids", type=int, nargs="*", default=None,
                    help="Restrict conversion to these COCO image ids (default: all)")
    args = p.parse_args()
    convert(args.coco, args.images, args.out,
            set(args.image_ids) if args.image_ids else None)
