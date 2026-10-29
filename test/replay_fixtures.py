"""Shared on-disk fixtures for the room replay / held-out tests."""

import json


def write_run_and_gt(tmp_path):
    """One GT room [0,0,400,400] on one page; the run's raw annotation has a
    single 250x250 OFFICE inside it; its recorded final room is the same box."""
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "docA_page000.png"}],
        "categories": [{"id": 1, "name": "office"}],
        "annotations": [
            {"image_id": 1, "category_id": 1,
             "segmentation": [[0, 0, 400, 0, 400, 400, 0, 400]]}
        ],
    }))
    run = tmp_path / "run"
    (run / "annotations").mkdir(parents=True)
    (run / "processed_annotations").mkdir()
    (run / "annotations" / "docA_page000.json").write_text(json.dumps({
        "image_file": "docA_page000.png",
        "image_size": {"width": 2000, "height": 2000},
        "rooms": [{"room_name": "OFFICE", "bbox": [50, 50, 250, 250],
                   "source": "vlm_ocr_merged", "confidence": 0.9}],
        "ocr_rooms": [],
    }))
    (run / "processed_annotations" / "docA_page000.json").write_text(json.dumps({
        "roomsRecognized": [{"coordinates": {"bbox": [50, 50, 300, 300]}}],
    }))
    return gt_path, run
