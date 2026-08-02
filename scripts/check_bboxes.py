#!/usr/bin/env python3
"""
Interactive Bbox Checker - Validate and inspect bboxes in annotation files

Usage:
    python3 check_bboxes.py                    # Show all files
    python3 check_bboxes.py --file page001     # Check specific file
    python3 check_bboxes.py --stats            # Show bbox statistics
    python3 check_bboxes.py --viz page001      # Create visualization
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_annotations.bbox.bbox_validator import (
    validate_bbox,
    AnnotationValidator,
    filter_oob_bboxes,
    normalize_bboxes,
    denormalize_bboxes,
)


class BboxChecker:
    """Interactive bbox validator and inspector."""
    
    def __init__(self, annotation_dir: str = "dataset_test_fix13/processed_annotations"):
        self.annotation_dir = Path(annotation_dir)
        self.image_width = 4500
        self.image_height = 3375
        self.validator = AnnotationValidator(
            image_width=self.image_width,
            image_height=self.image_height
        )
        
        # Load all annotations
        self.annotations = {}
        self._load_annotations()
    
    def _load_annotations(self):
        """Load all annotation files."""
        if not self.annotation_dir.exists():
            print(f"Error: Directory not found: {self.annotation_dir}")
            return
        
        for json_file in sorted(self.annotation_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    self.annotations[json_file.name] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load {json_file.name}: {e}")
    
    def list_files(self):
        """List all annotation files."""
        print("\n" + "=" * 80)
        print("AVAILABLE ANNOTATION FILES")
        print("=" * 80)
        
        if not self.annotations:
            print("No annotation files found.")
            return
        
        for i, filename in enumerate(sorted(self.annotations.keys()), 1):
            data = self.annotations[filename]
            rooms = data.get('rooms', [])
            print(f"\n{i}. {filename}")
            print(f"   Rooms: {len(rooms)}")
            
            # Show first 3 rooms
            for j, room in enumerate(rooms[:3]):
                bbox = room.get('bbox', [])
                name = room.get('room_name', 'N/A')
                confidence = room.get('confidence', 'N/A')
                print(f"     - {name:20s} bbox={bbox} conf={confidence}")
            
            if len(rooms) > 3:
                print(f"     ... and {len(rooms) - 3} more")
    
    def check_file(self, filename: Optional[str] = None):
        """Check a specific annotation file."""
        if not filename:
            if not self.annotations:
                print("No annotations available")
                return
            # Use first file
            filename = list(self.annotations.keys())[0]
        
        # Find matching file
        matching_files = [
            f for f in self.annotations.keys()
            if filename.lower() in f.lower()
        ]
        
        if not matching_files:
            print(f"Error: No file matching '{filename}' found")
            print(f"Available files: {list(self.annotations.keys())}")
            return
        
        filename = matching_files[0]
        data = self.annotations[filename]
        rooms = data.get('rooms', [])
        
        print("\n" + "=" * 80)
        print(f"FILE: {filename}")
        print("=" * 80)
        
        # Overall validation
        is_valid, errors = self.validator.validate_annotation(data)
        print(f"\nOverall Status: {'✓ VALID' if is_valid else '✗ INVALID'}")
        if errors:
            print("Errors:")
            for error in errors:
                print(f"  - {error}")
        
        # Check each bbox
        print(f"\nRooms: {len(rooms)}")
        print("-" * 80)
        
        oob_count = 0
        invalid_count = 0
        
        for i, room in enumerate(rooms, 1):
            bbox = room.get('bbox', [])
            name = room.get('room_name', 'Unknown')
            confidence = room.get('confidence', 'N/A')
            
            # Validate this bbox
            result = validate_bbox(
                bbox=bbox,
                image_width=self.image_width,
                image_height=self.image_height
            )
            
            status = "✓" if result.is_valid else "✗"
            
            # Check if OOB
            is_oob = False
            if len(bbox) == 4:
                x, y, w, h = bbox
                if x < 0 or y < 0 or x + w > self.image_width or y + h > self.image_height:
                    is_oob = True
                    oob_count += 1
            
            if not result.is_valid:
                invalid_count += 1
            
            print(f"\nRoom {i}: {status} {name}")
            print(f"  Bbox:       {bbox}")
            print(f"  Confidence: {confidence}")
            
            if is_oob:
                print(f"  ⚠ OUT OF BOUNDS (image: {self.image_width}x{self.image_height})")
            
            if result.errors:
                for error in result.errors:
                    print(f"  ERROR: {error}")
            
            if result.warnings:
                for warning in result.warnings:
                    print(f"  WARNING: {warning}")
            
            # Show normalized version
            if len(bbox) == 4 and result.is_valid:
                try:
                    normalized, ref_dims = normalize_bboxes(
                        [room],
                        self.image_width,
                        self.image_height
                    )
                    if normalized:
                        norm_bbox = normalized[0].get("bbox", bbox)
                        print(f"  Normalized: {[f'{v:.4f}' for v in norm_bbox]}")
                except Exception as e:
                    # Skip if normalization fails
                    pass
        
        # Summary
        print("\n" + "-" * 80)
        print("SUMMARY")
        print("-" * 80)
        print(f"Total rooms:     {len(rooms)}")
        print(f"Valid bboxes:    {len(rooms) - invalid_count}")
        print(f"Invalid bboxes:  {invalid_count}")
        print(f"Out of bounds:   {oob_count}")
    
    def show_statistics(self):
        """Show bbox dimension statistics across all files."""
        print("\n" + "=" * 80)
        print("BBOX STATISTICS (ALL FILES)")
        print("=" * 80)
        
        widths = []
        heights = []
        x_coords = []
        y_coords = []
        confidences = []
        total_rooms = 0
        
        for data in self.annotations.values():
            for room in data.get('rooms', []):
                bbox = room.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    x_coords.append(x)
                    y_coords.append(y)
                    widths.append(w)
                    heights.append(h)
                    total_rooms += 1
                    
                    conf = room.get('confidence', None)
                    if conf is not None:
                        confidences.append(conf)
        
        if not widths:
            print("No valid bboxes found")
            return
        
        print(f"\nTotal rooms across {len(self.annotations)} files: {total_rooms}")
        
        print("\n" + "-" * 80)
        print("X-COORDINATES (left edge)")
        print("-" * 80)
        print(f"  Mean:      {statistics.mean(x_coords):8.1f}")
        print(f"  Median:    {statistics.median(x_coords):8.1f}")
        print(f"  Stdev:     {statistics.stdev(x_coords):8.1f}")
        print(f"  Min:       {min(x_coords):8.1f}")
        print(f"  Max:       {max(x_coords):8.1f}")
        print(f"  Range:     {max(x_coords) - min(x_coords):8.1f}")
        
        print("\n" + "-" * 80)
        print("Y-COORDINATES (top edge)")
        print("-" * 80)
        print(f"  Mean:      {statistics.mean(y_coords):8.1f}")
        print(f"  Median:    {statistics.median(y_coords):8.1f}")
        print(f"  Stdev:     {statistics.stdev(y_coords):8.1f}")
        print(f"  Min:       {min(y_coords):8.1f}")
        print(f"  Max:       {max(y_coords):8.1f}")
        print(f"  Range:     {max(y_coords) - min(y_coords):8.1f}")
        
        print("\n" + "-" * 80)
        print("BBOX WIDTHS")
        print("-" * 80)
        print(f"  Mean:      {statistics.mean(widths):8.1f}")
        print(f"  Median:    {statistics.median(widths):8.1f}")
        print(f"  Stdev:     {statistics.stdev(widths):8.1f}")
        print(f"  Min:       {min(widths):8.1f}")
        print(f"  Max:       {max(widths):8.1f}")
        print(f"  Range:     {max(widths) - min(widths):8.1f}")
        
        print("\n" + "-" * 80)
        print("BBOX HEIGHTS")
        print("-" * 80)
        print(f"  Mean:      {statistics.mean(heights):8.1f}")
        print(f"  Median:    {statistics.median(heights):8.1f}")
        print(f"  Stdev:     {statistics.stdev(heights):8.1f}")
        print(f"  Min:       {min(heights):8.1f}")
        print(f"  Max:       {max(heights):8.1f}")
        print(f"  Range:     {max(heights) - min(heights):8.1f}")
        
        if confidences:
            print("\n" + "-" * 80)
            print("CONFIDENCE SCORES")
            print("-" * 80)
            print(f"  Mean:      {statistics.mean(confidences):8.3f}")
            print(f"  Median:    {statistics.median(confidences):8.3f}")
            print(f"  Min:       {min(confidences):8.3f}")
            print(f"  Max:       {max(confidences):8.3f}")
    
    def show_oob_bboxes(self):
        """Show all out-of-bounds bboxes."""
        print("\n" + "=" * 80)
        print("OUT-OF-BOUNDS BBOXES")
        print("=" * 80)
        
        oob_found = False
        
        for filename, data in sorted(self.annotations.items()):
            rooms = data.get('rooms', [])
            
            for i, room in enumerate(rooms):
                bbox = room.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    if x < 0 or y < 0 or x + w > self.image_width or y + h > self.image_height:
                        if not oob_found:
                            oob_found = True
                        
                        name = room.get('room_name', 'Unknown')
                        conf = room.get('confidence', 'N/A')
                        print(f"\n{filename}")
                        print(f"  Room {i+1}: {name}")
                        print(f"  Bbox: {bbox}")
                        print(f"  Confidence: {conf}")
                        
                        # Calculate overshoot
                        overshoot = []
                        if x < 0:
                            overshoot.append(f"left by {-x:.1f}")
                        if y < 0:
                            overshoot.append(f"top by {-y:.1f}")
                        if x + w > self.image_width:
                            overshoot.append(f"right by {x + w - self.image_width:.1f}")
                        if y + h > self.image_height:
                            overshoot.append(f"bottom by {y + h - self.image_height:.1f}")
                        
                        if overshoot:
                            print(f"  Overshoot: {', '.join(overshoot)}")
        
        if not oob_found:
            print("\n✓ No out-of-bounds bboxes found!")
    
    def filter_and_report(self):
        """Filter OOB bboxes and report statistics."""
        print("\n" + "=" * 80)
        print("OOB FILTERING REPORT")
        print("=" * 80)
        
        total_before = 0
        total_after = 0
        total_filtered = 0
        
        for filename, data in sorted(self.annotations.items()):
            rooms = data.get('rooms', [])
            
            # Filter OOB manually since the function expects room dicts
            filtered_rooms = []
            for room in rooms:
                bbox = room.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    # Check if in bounds
                    if (x >= 0 and y >= 0 and 
                        x + w <= self.image_width and 
                        y + h <= self.image_height):
                        filtered_rooms.append(room)
                else:
                    # Keep if invalid format (will be caught later)
                    filtered_rooms.append(room)
            
            total_before += len(rooms)
            total_after += len(filtered_rooms)
            total_filtered += len(rooms) - len(filtered_rooms)
            
            if len(rooms) != len(filtered_rooms):
                print(f"\n{filename}")
                print(f"  Before: {len(rooms)} bboxes")
                print(f"  After:  {len(filtered_rooms)} bboxes")
                print(f"  Filtered: {len(rooms) - len(filtered_rooms)}")
        
        print("\n" + "-" * 80)
        print("TOTAL ACROSS ALL FILES")
        print("-" * 80)
        print(f"Before filtering: {total_before}")
        print(f"After filtering:  {total_after}")
        print(f"Total filtered:   {total_filtered}")
        if total_before > 0:
            print(f"Filtering rate:   {(total_filtered/total_before*100):.1f}%")
        else:
            print(f"Filtering rate:   0.0%")


def main():
    parser = argparse.ArgumentParser(
        description="Check and validate bboxes in annotation files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 check_bboxes.py                      # List all files
  python3 check_bboxes.py --file page001       # Check specific file
  python3 check_bboxes.py --stats              # Show statistics
  python3 check_bboxes.py --oob                # Show OOB bboxes
  python3 check_bboxes.py --filter             # Filter and report
        """
    )
    
    parser.add_argument(
        '--file',
        help='Check specific file (partial name match)',
        default=None
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show bbox statistics'
    )
    parser.add_argument(
        '--oob',
        action='store_true',
        help='Show out-of-bounds bboxes'
    )
    parser.add_argument(
        '--filter',
        action='store_true',
        help='Filter OOB bboxes and report'
    )
    parser.add_argument(
        '--dir',
        default='dataset_test_fix13/processed_annotations',
        help='Annotation directory (default: dataset_test_fix13/processed_annotations)'
    )
    
    args = parser.parse_args()
    
    # Create checker
    checker = BboxChecker(annotation_dir=args.dir)
    
    # Execute commands
    if args.file:
        checker.check_file(args.file)
    elif args.stats:
        checker.show_statistics()
    elif args.oob:
        checker.show_oob_bboxes()
    elif args.filter:
        checker.filter_and_report()
    else:
        # Default: list all files
        checker.list_files()


if __name__ == "__main__":
    main()
