import sys, cv2
sys.path.insert(0, "/home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations/src")
sys.path.insert(0, "/tmp")
from wallband_box_prompt_spike import separate_bands, box_from_walls, MIN_EXPANSION_AREA_PX
from preprocessing_annotations.detection.window_strip_detector import find_wall_bands, WindowStripDetectorConfig
from preprocessing_annotations.detection.sam_segmenter import RoomSegmenter, is_label_scale_result
from preprocessing_annotations.config import SAMConfig, OCRConfig
from preprocessing_annotations.ingestion.two_pass_ocr_extractor import TwoPassOCRExtractor
from pathlib import Path

img_path = Path("/home/ubuntu/floorplan_classifier/VLM/test_pcs/commercial/commercial_building_obj_detection_coco_w_images/images/aceb87d1-326_ROCKAWAY_-_AVI-ON_LAYOUT_page001.png")
gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
h, w = gray.shape
page_area = h * w
bands = find_wall_bands(gray, WindowStripDetectorConfig())
horiz, vert = separate_bands(bands)

extractor = TwoPassOCRExtractor(OCRConfig(backend="paddleocr"))
rooms, _ = extractor.find_rooms_pass1(img_path)

segmenter = RoomSegmenter(config=SAMConfig(device="cuda"))

print(f"page {w}x{h}={page_area}px2")
for r in rooms:
    x1,y1,x2,y2 = r.bbox
    seed = ((x1+x2)/2, (y1+y2)/2)
    box = box_from_walls(seed, horiz, vert, w, h)
    if box is None:
        print(f"{r.room_name[:20]:20} NO BOX")
        continue
    box_area = box[2]*box[3]
    result = segmenter.segment_from_point(img_path, (int(seed[0]), int(seed[1])), include_mask=False, label_bbox=box)
    sam_area = result.bbox[2]*result.bbox[3]
    print(f"{r.room_name[:20]:20} box={box} box_area_pct={100*box_area/page_area:5.1f}%  sam_bbox={result.bbox} sam_area_pct={100*sam_area/page_area:5.1f}%  is_noop={is_label_scale_result(sam_area, MIN_EXPANSION_AREA_PX)}")
