"""
Hallucination detection for VLM room detections.

This module provides shared hallucination detection utilities used by multiple VLM backends.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def detect_hallucinations(rooms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect and truncate autoregressive hallucination patterns in room detections.
    
    VLMs can enter loops generating repetitive or fabricated bboxes. This function detects
    multiple hallucination patterns:
    1. Identical bbox repetition: N rooms with exact same bbox
    2. Grid pattern: rooms on regular grid (constant x-step AND y-step)
    3. Incremental y-delta: same x-coords, incrementing y
    
    Args:
        rooms: List of room detections from VLM
    
    Returns:
        Truncated list with hallucinations removed
    """
    if len(rooms) < 2:
        return rooms
    
    # Detect coordinate space: if any bbox value > 2, assume pixel coords; else assume fractions (0.0-1.0)
    # This handles both UnslothQwenBackend (fraction prompt) and Qwen2_5VLBackend (percentage prompt)
    max_coord_value = 0
    for room in rooms:
        bbox = room.get("bbox", [])
        if bbox and len(bbox) == 4:
            try:
                max_coord_value = max(max_coord_value, max(abs(float(v)) for v in bbox))
            except (ValueError, TypeError):
                pass
    
    is_pixel_space = max_coord_value > 2.0  # Thresholds assume 0-1 or 0-100; pixel coords are in thousands
    
    # Log input coordinate space for diagnostic purposes
    if len(rooms) > 0:
        first_bbox = rooms[0].get("bbox", [])
        logger.debug(
            f"Hallucination detection: {len(rooms)} rooms, max_coord={max_coord_value:.1f}, "
            f"is_pixel_space={is_pixel_space}, first_bbox={first_bbox[:4] if first_bbox else None}"
        )
    
    # Normalize all bboxes to fraction space (0.0-1.0) for consistent threshold application
    if is_pixel_space:
        # For pixel coords, we need image dimensions. Use max values as proxy.
        # Typical floorplans: 4000-10000 px width/height
        # Scale by dividing by max observed coordinate
        scale_factor = max_coord_value if max_coord_value > 0 else 4096
        normalized_rooms = []
        for room in rooms:
            normalized_room = room.copy()
            bbox = room.get("bbox", [])
            if bbox and len(bbox) == 4:
                try:
                    normalized_bbox = [float(v) / scale_factor for v in bbox]
                    normalized_room["bbox"] = normalized_bbox
                except (ValueError, TypeError):
                    pass
            normalized_rooms.append(normalized_room)
    else:
        # Already in fraction space (0.0-1.0)
        normalized_rooms = rooms
    
    # Apply detection patterns on normalized (fraction-space) bboxes
    detected_hallucination = False
    
    # Pattern 1: Detect identical bboxes (indicates infinite loop at same location)
    # If we see 3+ identical bboxes, keep only the first one
    try:
        seen_bboxes = {}
        first_unique_indices = []
        duplicate_start = None
        
        for i, room in enumerate(normalized_rooms):
            bbox = tuple(room.get("bbox", []))
            if bbox and len(bbox) == 4:
                # Convert to a hashable tuple for comparison
                # Work in normalized fraction space (0.0-1.0), quantize to 1/100
                bbox_key = tuple(int(v * 100) for v in bbox)
                
                if bbox_key not in seen_bboxes:
                    seen_bboxes[bbox_key] = []
                    first_unique_indices.append(i)
                seen_bboxes[bbox_key].append(i)
        
        # Check if any bbox appears 3+ times
        for bbox_key, indices in seen_bboxes.items():
            if len(indices) >= 3:
                duplicate_start = min(indices[1:])  # start of duplicates (skip first)
                logger.warning(
                    f"Detected identical bbox hallucination at room {indices[0]}: "
                    f"bbox appears {len(indices)} times (indices {indices}). "
                    f"Truncating list at position {duplicate_start}"
                )
                return rooms[:duplicate_start]
    except (TypeError, ValueError, IndexError):
        pass
    
    # Pattern 2: Detect grid pattern (room grid like 3x3, 4x3, etc)
    # If rooms form a regular grid with constant x-step and y-step, it's likely hallucinated
    if len(normalized_rooms) >= 6:
        try:
            # Extract all bboxes and check for grid regularity
            bboxes = []
            for room in normalized_rooms:
                bbox = room.get("bbox", [])
                if bbox and len(bbox) == 4:
                    bboxes.append(bbox)
                else:
                    break  # Stop if we hit an invalid bbox
            
            if len(bboxes) >= 6:
                # Check for grid pattern: collect all unique x1 and y1 values
                x1_values = set()
                y1_values = set()
                
                for bbox in bboxes:
                    # Quantize to avoid floating point issues
                    # Working in normalized 0.0-1.0 space
                    x1_key = int(bbox[0] * 100)
                    y1_key = int(bbox[1] * 100)
                    x1_values.add(x1_key)
                    y1_values.add(y1_key)
                
                # Grid detection: if rooms fit into a regular grid pattern (e.g., 3x3, 2x3, 4x2)
                # then we expect num_rooms = num_x_values * num_y_values
                num_x = len(x1_values)
                num_y = len(y1_values)
                
                if num_x >= 2 and num_y >= 2 and (num_x * num_y) >= 6:
                    # Check if actual rooms match grid dimensions (allowing some tolerance)
                    expected_grid = num_x * num_y
                    actual_rooms = len(bboxes)
                    
                    # If we have close to expected grid size, it's probably a hallucination
                    if abs(actual_rooms - expected_grid) <= 1:
                        logger.warning(
                            f"Detected grid pattern hallucination: "
                            f"{actual_rooms} rooms on {num_x}x{num_y} grid. "
                            f"Truncating entire list"
                        )
                        return rooms[:0]  # Return empty list
        except (TypeError, ValueError, IndexError):
            pass
    
    # Pattern 3: Check for pattern with same x-coords, incrementing y by consistent delta
    # This is the original pattern detector
    if len(normalized_rooms) >= 4:
        for i in range(len(normalized_rooms) - 3):
            r0, r1, r2, r3 = normalized_rooms[i:i+4]
            
            # Extract bboxes (normalized to 0.0-1.0)
            b0 = r0.get("bbox", [])
            b1 = r1.get("bbox", [])
            b2 = r2.get("bbox", [])
            b3 = r3.get("bbox", [])
            
            if not all(len(b) == 4 for b in [b0, b1, b2, b3]):
                continue
            
            try:
                # Check if x-coords match (same left/right edges)
                x0_match = abs(b0[0] - b1[0]) < 0.01 and abs(b0[2] - b1[2]) < 0.01
                x1_match = abs(b1[0] - b2[0]) < 0.01 and abs(b1[2] - b2[2]) < 0.01
                x2_match = abs(b2[0] - b3[0]) < 0.01 and abs(b2[2] - b3[2]) < 0.01
                
                if not (x0_match and x1_match and x2_match):
                    continue
                
                # Check if y-coords increment by consistent delta (in normalized 0.0-1.0 space)
                delta_01 = b1[1] - b0[1]
                delta_12 = b2[1] - b1[1]
                delta_23 = b3[1] - b2[1]
                
                # Allow ±5% tolerance on delta consistency
                if (abs(delta_01 - delta_12) < 0.05 and 
                    abs(delta_12 - delta_23) < 0.05 and 
                    0.15 < delta_01 < 0.35):
                    
                    logger.warning(
                        f"Detected y-delta hallucination pattern at room {i}: "
                        f"repetitive bboxes with constant y-delta={delta_01:.3f}. "
                        f"Truncating list at position {i}"
                    )
                    return rooms[:i]
            except (TypeError, ValueError, IndexError):
                continue
    
    # Pattern 4: Check for uniform bbox sizes (all bboxes nearly identical width/height)
    # Real floorplans have rooms of varied sizes; uniform size grid suggests hallucination.
    #
    # Exception — residential unit grids: a multi-family building legitimately has
    # dozens of identical-size apartment units arranged in a grid.  To distinguish
    # genuine uniform-unit layouts from VLM autoregressive loops, we check NAME
    # DIVERSITY: if the room names are mostly distinct, the repetition is real.
    # If the VLM is looping it typically reuses generic names like "Office 1",
    # "Office 2" … or emits identical names for every room.
    #
    # Rule: truncate ONLY when BOTH conditions hold:
    #   (a) bbox size CV < 5%   (uniform dimensions)
    #   (b) name uniqueness < 40%  (fewer than 40% of names are distinct)
    if len(normalized_rooms) >= 7:
        try:
            bboxes = []
            for room in normalized_rooms:
                bbox = room.get("bbox", [])
                if bbox and len(bbox) == 4:
                    bboxes.append(bbox)
                else:
                    break
            
            if len(bboxes) >= 7:
                # Convert from [x1, y1, x2, y2] format (normalized 0-1) to widths and heights
                widths = []
                heights = []
                for bbox in bboxes:
                    try:
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        if width > 0 and height > 0:  # Skip invalid bboxes
                            widths.append(width)
                            heights.append(height)
                    except (TypeError, IndexError):
                        pass
                
                if len(widths) >= 7 and len(heights) >= 7:
                    import statistics
                    width_mean = statistics.mean(widths)
                    height_mean = statistics.mean(heights)
                    
                    if width_mean > 0 and height_mean > 0:
                        width_std = statistics.stdev(widths) if len(widths) > 1 else 0
                        height_std = statistics.stdev(heights) if len(heights) > 1 else 0
                        
                        width_cv = width_std / width_mean
                        height_cv = height_std / height_mean
                        
                        # Check name diversity before deciding to truncate.
                        # Collect names from the original (non-normalized) rooms so we
                        # work with the actual labels the VLM produced.
                        names = [
                            str(r.get("room_name") or r.get("name") or "").strip()
                            for r in rooms[:len(bboxes)]
                        ]
                        unique_names = len(set(n.upper() for n in names if n))
                        total_names = len([n for n in names if n])
                        name_uniqueness = unique_names / total_names if total_names > 0 else 0.0

                        # Uniform size AND low name diversity → hallucination
                        if width_cv < 0.05 and height_cv < 0.05:
                            if name_uniqueness < 0.40:
                                logger.warning(
                                    f"Detected uniform bbox size hallucination: "
                                    f"all {len(bboxes)} rooms have nearly identical dimensions "
                                    f"(width_cv={width_cv:.3f}, height_cv={height_cv:.3f}) "
                                    f"and low name diversity ({unique_names}/{total_names}={name_uniqueness:.2f}). "
                                    f"Truncating entire list"
                                )
                                return rooms[:0]
                            else:
                                # Uniform size but diverse names → likely a real residential
                                # unit grid (TYPE-A1, TYPE-B3 …).  Keep all rooms.
                                logger.info(
                                    f"Uniform bbox sizes (width_cv={width_cv:.3f}, "
                                    f"height_cv={height_cv:.3f}) but high name diversity "
                                    f"({unique_names}/{total_names}={name_uniqueness:.2f}) — "
                                    f"treating as real residential unit grid, not hallucination"
                                )
        except (ValueError, TypeError, statistics.StatisticsError, ImportError):
            pass
    
    return rooms
