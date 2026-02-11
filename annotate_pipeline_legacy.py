import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
import io

def extract_pdf_pages(pdf_dir: str, output_dir: str, dpi: int = 200):
    """Convert all PDFs to high-resolution images."""
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pdf_path in pdf_dir.glob("*.pdf"):
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc):
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            output_path = output_dir / f"{pdf_path.stem}_page{page_num:02d}.png"
            img.save(output_path, "PNG")
            print(f"Extracted: {output_path}")
        
        doc.close()

import easyocr
import cv2
import re
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RoomCandidate:
    bbox: Tuple[int, int, int, int]
    room_number: str
    room_name: str
    confidence: float

class MEPTextExtractor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)
        
        # Patterns for room identification
        self.room_number_pattern = re.compile(r'^(\d{3}[A-Z]?|\d{2,3})$')
        self.room_name_patterns = [
            re.compile(r'(SUITE|OFFICE|ROOM)\s*\d*', re.I),
            re.compile(r'MECHANICAL\s*(ROOM)?', re.I),
            re.compile(r'ELECTRICAL\s*(ROOM)?', re.I),
            re.compile(r'ELEVATOR', re.I),
            re.compile(r'STAIR', re.I),
            re.compile(r'CUSTODIAL', re.I),
            re.compile(r'(MEN|WOMEN|RESTROOM)', re.I),
            re.compile(r'ENTRANCE|ENTRY|LOBBY', re.I),
            re.compile(r'CORRIDOR|HALLWAY', re.I),
            re.compile(r'RISER', re.I),
        ]
    
    def extract_text(self, image_path: str):
        """Extract all text with bounding boxes."""
        image = cv2.imread(image_path)
        results = self.reader.readtext(image)
        return results
    
    def find_room_candidates(self, results) -> List[RoomCandidate]:
        """Identify text that represents room labels."""
        candidates = []
        
        for (bbox, text, conf) in results:
            room_number = None
            room_name = None
            
            # Check patterns
            if self.room_number_pattern.match(text.strip()):
                room_number = text.strip()
            
            for pattern in self.room_name_patterns:
                if pattern.search(text):
                    room_name = text.strip()
                    break
            
            if room_number or room_name:
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x, y = int(min(x_coords)), int(min(y_coords))
                w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                
                candidates.append(RoomCandidate(
                    bbox=(x, y, w, h),
                    room_number=room_number or "",
                    room_name=room_name or "",
                    confidence=conf
                ))
        
        return candidates

import cv2
import numpy as np
from pathlib import Path

class SymbolTemplateDetector:
    def __init__(self, template_dir: str):
        """
        Load symbol templates. Structure:
        template_dir/receptacle/*.png
        template_dir/switch/*.png
        """
        self.templates = {}
        template_path = Path(template_dir)
        
        for category_dir in template_path.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                self.templates[category] = []
                
                for template_file in category_dir.glob("*.png"):
                    template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        for scale in [0.8, 1.0, 1.2]:
                            scaled = cv2.resize(template, None, fx=scale, fy=scale)
                            self.templates[category].append(scaled)
    
    def detect_symbols(self, image_path: str, threshold: float = 0.7):
        """Detect symbols using template matching."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        detections = []
        
        for category, templates in self.templates.items():
            for template in templates:
                h, w = template.shape
                result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= threshold)
                
                for pt in zip(*locations[::-1]):
                    detections.append({
                        'category': category,
                        'bbox': (pt[0], pt[1], w, h),
                        'confidence': float(result[pt[1], pt[0]])
                    })
        
        # Apply NMS
        return self._nms(detections)
    
    def _nms(self, detections, iou_threshold=0.3):
        if not detections:
            return []
        
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []
        
        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [d for d in detections 
                         if self._iou(best['bbox'], d['bbox']) < iou_threshold]
        
        return keep
    
    def _iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = min(x1 + w1, x2 + w2) - xi
        hi = min(y1 + h1, y2 + h2) - yi
        
        if wi <= 0 or hi <= 0:
            return 0.0
        
        intersection = wi * hi
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union
    
import anthropic
import base64
import json
from pathlib import Path
from PIL import Image

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

def vlm_annotate_mep_plan(image_path: str, client: anthropic.Anthropic) -> dict:
    """Use Claude to generate draft annotations."""
    
    image_data = encode_image(image_path)
    
    # Get image dimensions
    with Image.open(image_path) as img:
        img_width, img_height = img.size
    
    prompt = f"""Analyze this MEP/Electrical floor plan and extract structured annotations.

IMAGE DIMENSIONS: {img_width} x {img_height} pixels

For each ROOM or SPACE visible:
1. Room number (e.g., "113", "S1.100")  
2. Room name (e.g., "MECHANICAL ROOM", "SUITE 102")
3. Bounding box in PIXELS: [x, y, width, height] where (x,y) is top-left corner
4. Category from: office_suite, mechanical_room, electrical_room, elevator_core, 
   stair, custodial, restroom_men, restroom_women, corridor, entrance, storage

For ELECTRICAL PANELS:
1. Panel label (e.g., "PANEL H1")
2. Bounding box in pixels

Output ONLY valid JSON:
{{
  "rooms": [
    {{"room_number": "113", "room_name": "MECHANICAL ROOM", "category": "mechanical_room", "bbox": [x, y, w, h]}}
  ],
  "panels": [
    {{"label": "PANEL H1", "bbox": [x, y, w, h]}}
  ],
  "electrical_counts": {{
    "fixtures": 0,
    "receptacles": 0,
    "switches": 0,
    "sensors": 0
  }}
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    
    # Parse JSON from response
    response_text = response.content[0].text
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    
    if json_match:
        result = json.loads(json_match.group())
        result["image_size"] = {"width": img_width, "height": img_height}
        return result
    
    return {"error": "Parse failed", "raw": response_text}

def batch_vlm_annotate(image_dir: str, output_dir: str):
    """Process all images with VLM."""
    client = anthropic.Anthropic()
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for image_path in sorted(image_dir.glob("*.png")):
        print(f"Processing: {image_path.name}")
        
        try:
            annotations = vlm_annotate_mep_plan(str(image_path), client)
            annotations["image_file"] = image_path.name
            
            output_path = output_dir / f"{image_path.stem}.json"
            with open(output_path, "w") as f:
                json.dump(annotations, f, indent=2)
            
            print(f"  ✓ Found {len(annotations.get('rooms', []))} rooms")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")

# Usage
batch_vlm_annotate("./images", "./vlm_annotations")

from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np

class RoomSegmenter:
    def __init__(self, checkpoint: str = "sam_vit_h_4b8939.pth"):
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        sam.to("cuda")
        self.predictor = SamPredictor(sam)
    
    def segment_from_labels(self, image_path: str, label_points: list):
        """
        Use room label locations as prompts for SAM.
        
        Args:
            label_points: [(x, y), ...] centers of detected room labels
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        
        results = []
        for point in label_points:
            input_point = np.array([[point[0], point[1]]])
            input_label = np.array([1])
            
            masks, scores, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            
            # Convert mask to bbox
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                results.append({
                    "prompt_point": point,
                    "bbox": [x, y, w, h],
                    "confidence": float(scores[best_idx]),
                    "mask": mask
                })
        
        return results

def export_to_label_studio(annotations_dir: str, images_dir: str, output_file: str):
    """Convert annotations to Label Studio format with pre-labels."""
    
    tasks = []
    
    for json_file in Path(annotations_dir).glob("*.json"):
        with open(json_file) as f:
            ann = json.load(f)
        
        img_w = ann.get("image_size", {}).get("width", 1000)
        img_h = ann.get("image_size", {}).get("height", 1000)
        
        task = {
            "data": {
                "image": f"/data/local-files/?d={images_dir}/{ann['image_file']}"
            },
            "predictions": [{"result": []}]
        }
        
        # Add room annotations
        for i, room in enumerate(ann.get("rooms", [])):
            bbox = room.get("bbox", [0, 0, 100, 100])
            
            task["predictions"][0]["result"].append({
                "id": f"room_{i}",
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "value": {
                    "x": bbox[0] / img_w * 100,
                    "y": bbox[1] / img_h * 100,
                    "width": bbox[2] / img_w * 100,
                    "height": bbox[3] / img_h * 100,
                    "rectanglelabels": [room.get("category", "unknown")]
                }
            })
        
        tasks.append(task)
    
    with open(output_file, "w") as f:
        json.dump(tasks, f, indent=2)
    
    print(f"Exported {len(tasks)} tasks")

def prioritize_for_review(annotations_dir: str, top_percent: float = 0.25):
    """Select lowest-confidence annotations for human review."""
    
    scores = []
    
    for json_file in Path(annotations_dir).glob("*.json"):
        with open(json_file) as f:
            ann = json.load(f)
        
        # Compute confidence score
        rooms = ann.get("rooms", [])
        if not rooms:
            confidence = 0.0
        else:
            # Average confidence, penalize missing data
            conf_values = []
            for room in rooms:
                base_conf = 0.7
                if room.get("room_number"):
                    base_conf += 0.15
                if room.get("room_name"):
                    base_conf += 0.15
                conf_values.append(base_conf)
            confidence = sum(conf_values) / len(conf_values)
        
        scores.append((json_file.name, confidence))
    
    # Sort by confidence (lowest first)
    scores.sort(key=lambda x: x[1])
    
    n_review = int(len(scores) * top_percent)
    to_review = [name for name, _ in scores[:n_review]]
    
    print(f"Need review: {len(to_review)} / {len(scores)} images")
    print("\nLowest confidence:")
    for name, conf in scores[:10]:
        print(f"  {name}: {conf:.2f}")
    
    return to_review

#!/usr/bin/env python3
"""
Full annotation pipeline for MEP plans.

Usage:
    python annotate_pipeline.py --input ./pdfs --output ./dataset
"""

import argparse
from pathlib import Path
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-vlm", action="store_true")
    parser.add_argument("--use-sam", action="store_true")
    args = parser.parse_args()
    
    output = Path(args.output)
    images_dir = output / "images"
    annotations_dir = output / "annotations"
    
    # Step 1: Extract images
    print("=" * 50)
    print("STEP 1: Extracting images from PDFs")
    print("=" * 50)
    extract_pdf_pages(args.input, str(images_dir), dpi=200)
    
    # Step 2: OCR extraction
    print("\n" + "=" * 50)
    print("STEP 2: OCR text extraction")
    print("=" * 50)
    extractor = MEPTextExtractor()
    
    ocr_results = {}
    for img in images_dir.glob("*.png"):
        results = extractor.extract_text(str(img))
        rooms = extractor.find_room_candidates(results)
        ocr_results[img.name] = rooms
        print(f"  {img.name}: {len(rooms)} room labels found")
    
    # Step 3: VLM annotation (optional but recommended)
    if args.use_vlm:
        print("\n" + "=" * 50)
        print("STEP 3: VLM zero-shot annotation")
        print("=" * 50)
        batch_vlm_annotate(str(images_dir), str(annotations_dir))
    
    # Step 4: SAM refinement (optional)
    if args.use_sam:
        print("\n" + "=" * 50)
        print("STEP 4: SAM boundary refinement")
        print("=" * 50)
        segmenter = RoomSegmenter()
        
        for img in images_dir.glob("*.png"):
            if img.name in ocr_results:
                points = [(r.bbox[0] + r.bbox[2]//2, r.bbox[1] + r.bbox[3]//2) 
                          for r in ocr_results[img.name]]
                if points:
                    refined = segmenter.segment_from_labels(str(img), points)
                    # Update annotations with refined bboxes...
    
    # Step 5: Identify for review
    print("\n" + "=" * 50)
    print("STEP 5: Prioritizing for human review")
    print("=" * 50)
    needs_review = prioritize_for_review(str(annotations_dir))
    
    with open(output / "needs_review.txt", "w") as f:
        f.write("\n".join(needs_review))
    
    # Step 6: Export for Label Studio
    print("\n" + "=" * 50)
    print("STEP 6: Exporting to Label Studio")
    print("=" * 50)
    export_to_label_studio(
        str(annotations_dir),
        str(images_dir),
        str(output / "label_studio_import.json")
    )
    
    # Summary
    print("\n" + "=" * 50)
    print("COMPLETE")
    print("=" * 50)
    n_images = len(list(images_dir.glob("*.png")))
    n_annotations = len(list(annotations_dir.glob("*.json")))
    print(f"Images processed: {n_images}")
    print(f"Annotations generated: {n_annotations}")
    print(f"Flagged for review: {len(needs_review)} ({len(needs_review)/n_images*100:.0f}%)")
    print(f"\nNext steps:")
    print(f"1. Import {output}/label_studio_import.json into Label Studio")
    print(f"2. Review the {len(needs_review)} flagged images")
    print(f"3. Export corrected annotations as COCO JSON")

if __name__ == "__main__":
    main()