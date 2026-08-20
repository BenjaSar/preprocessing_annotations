"""
SAM (Segment Anything Model) segmentation module for room boundary refinement.

This module uses SAM to refine room boundaries based on room label
locations detected by OCR or VLM annotation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ..config import SAMConfig
except ImportError:
    from config import SAMConfig

logger = logging.getLogger(__name__)


class SAMSegmentationError(Exception):
    """Raised when SAM segmentation fails."""

    pass


@dataclass
class SegmentationResult:
    """
    Result of SAM segmentation for a single room.

    Attributes:
        prompt_point: The (x, y) point used to prompt SAM.
        bbox: Refined bounding box as [x, y, width, height].
        confidence: SAM confidence score.
        mask: Binary segmentation mask (numpy array).
        contour: Contour points of the segmented region.
    """

    prompt_point: Tuple[int, int]
    bbox: List[int]
    confidence: float
    mask: Optional[np.ndarray] = None
    contour: Optional[np.ndarray] = None


class RoomSegmenter:
    """
    Segments room boundaries using SAM with point prompts.

    Uses room label locations (from OCR or VLM) as prompts to
    SAM for generating precise room boundary masks.

    Attributes:
        config: SAMConfig with model parameters.

    Example:
        segmenter = RoomSegmenter(config)
        results = segmenter.segment_from_labels("floorplan.png", [(100, 200), (300, 400)])
    """

    def __init__(self, config: Optional[SAMConfig] = None):
        self.config = config or SAMConfig()
        self._predictor = None
        self._current_image_path = None

    @property
    def predictor(self):
        """Lazy initialization of SAM predictor."""
        if self._predictor is None:
            try:
                from segment_anything import SamPredictor, sam_model_registry
            except ImportError:
                raise SAMSegmentationError(
                    "segment_anything package not installed. "
                    "Install with: pip install segment-anything"
                )

            # Resolve checkpoint path: if relative, resolve relative to module directory
            checkpoint_path = Path(self.config.checkpoint)
            if not checkpoint_path.is_absolute():
                # Resolve relative to this module's directory
                module_dir = Path(__file__).parent
                checkpoint_path = module_dir / checkpoint_path
            
            if not checkpoint_path.exists():
                raise SAMSegmentationError(
                    f"SAM checkpoint not found: {checkpoint_path}. "
                    f"Download from https://github.com/facebookresearch/segment-anything"
                )

            try:
                sam = sam_model_registry[self.config.model_type](
                    checkpoint=str(checkpoint_path)
                )

                # Move to appropriate device
                if self.config.device == "cuda":
                    import torch

                    if torch.cuda.is_available():
                        sam.to("cuda")
                        logger.info("SAM loaded on CUDA")
                    else:
                        logger.warning("CUDA requested but not available, using CPU")
                elif self.config.device == "mps":
                    import torch

                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        sam.to("mps")
                        logger.info("SAM loaded on MPS")
                    else:
                        logger.warning("MPS requested but not available, using CPU")
                else:
                    logger.info("SAM loaded on CPU")

                self._predictor = SamPredictor(sam)

            except Exception as e:
                raise SAMSegmentationError(f"Failed to load SAM model: {e}") from e

        return self._predictor

    def _set_image(self, image_path: str | Path) -> np.ndarray:
        """
        Load and set image for SAM predictor.

        Caches the current image to avoid reloading for multiple prompts.
        """
        image_path = Path(image_path)

        if str(image_path) == self._current_image_path:
            return None  # Image already set

        image = cv2.imread(str(image_path))
        if image is None:
            raise SAMSegmentationError(f"Failed to load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        self._current_image_path = str(image_path)

        logger.debug(f"Set image for SAM: {image_path.name}")
        return image

    def segment_from_point(
        self,
        image_path: str | Path,
        point: Tuple[int, int],
        include_mask: bool = True,
        label_bbox: Optional[List[int]] = None,
    ) -> SegmentationResult:
        """
        Segment a room from a label centroid, optionally constrained by a box prompt.

        Args:
            image_path: Path to the floor plan image.
            point: (x, y) point inside the room (typically label center).
            include_mask: Whether to include the full mask in result.
            label_bbox: Optional [x, y, w, h] label box. When provided, used as a
                SAM box prompt (F-B) to confine segmentation to the room around
                the label rather than the whole floor plate.

        Returns:
            SegmentationResult with refined bbox and optional mask.
        """
        self._set_image(image_path)

        input_point = np.array([[point[0], point[1]]])
        input_label = np.array([1])  # 1 = foreground

        # F-B: box prompt from label bbox (xywh -> xyxy) constrains SAM region
        box = None
        if label_bbox is not None and len(label_bbox) == 4:
            bx, by, bw, bh = label_bbox
            box = np.array([bx, by, bx + bw, by + bh])

        # Always request 3 candidates so mask selection can avoid the floor-plate.
        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=box,
            multimask_output=True,
        )

        # F-A: pick a room-scale mask, NOT argmax(score) (which favours the
        # largest floor-plate region on floor plans).
        mask, confidence = self._select_room_mask(masks, scores)

        # Extract bounding box and contour from mask
        bbox, contour = self._mask_to_bbox(mask)

        result = SegmentationResult(
            prompt_point=point,
            bbox=bbox,
            confidence=confidence,
            mask=mask if include_mask else None,
            # Contour is NOT gated on include_mask: it is a small (N, 1, 2)
            # polygon already computed by _mask_to_bbox above, whereas `mask`
            # is a full HxW array -- the memory cost include_mask exists to
            # avoid. Callers need the polygon for COCO `segmentation` export.
            contour=contour,
        )

        return result

    def _select_room_mask(
        self, masks: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Select a room-scale mask from SAM multimask candidates (F-A).

        Thin delegate to the module-level `select_room_mask` so any promptable
        segmenter (not just this SAM1 predictor) can reuse the identical
        selection algorithm — see sam21_probe for the SAM2.1 comparison.
        """
        return select_room_mask(masks, scores, self.config.max_expand_frac)

    def segment_from_labels(
        self,
        image_path: str | Path,
        label_points: List[Tuple[int, int]],
        include_masks: bool = False,
    ) -> List[SegmentationResult]:
        """
        Segment multiple rooms using label locations as prompts.

        Args:
            image_path: Path to the floor plan image.
            label_points: List of (x, y) center points of room labels.
            include_masks: Whether to include masks in results.

        Returns:
            List of SegmentationResult objects.
        """
        if not label_points:
            logger.warning("No label points provided for segmentation")
            return []

        results = []
        for point in label_points:
            try:
                result = self.segment_from_point(
                    image_path, point, include_mask=include_masks
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Segmentation failed for point {point}: {e}")
                continue

        logger.info(
            f"Segmented {len(results)}/{len(label_points)} rooms from {Path(image_path).name}"
        )

        return results

    def _split_mask_instances(
        self,
        mask: np.ndarray,
        min_area: int = 40000,
        erosion_px: int = 3,
    ) -> List[List[int]]:
        """T1: Split a SAM binary mask into per-instance bboxes via connected components.

        Erodes the mask to disconnect adjacent rooms that share a thin wall, then
        labels connected components. Returns one [x, y, w, h] per component that
        meets the min_area threshold.

        Args:
            mask: Boolean or uint8 binary mask (H, W).
            min_area: Minimum component area in px² to emit a bbox (default 40000).
            erosion_px: Square erosion kernel side length (default 3).

        Returns:
            List of [x, y, w, h] bboxes, one per qualifying component.
            Empty list if no component meets min_area.
        """
        mask_u8 = mask.astype(np.uint8)
        if erosion_px > 0:
            kernel = np.ones((erosion_px, erosion_px), np.uint8)
            mask_u8 = cv2.erode(mask_u8, kernel)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        bboxes = []
        for label in range(1, n_labels):  # skip background (label 0)
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            bboxes.append([x, y, w, h])
        return bboxes

    @staticmethod
    def _iou_masks(a: np.ndarray, b: np.ndarray) -> float:
        """T3: Compute IoU of two boolean or uint8 masks of equal shape."""
        inter = float(np.logical_and(a, b).sum())
        if inter == 0.0:
            return 0.0
        union = float(np.logical_or(a, b).sum())
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _is_multi_component(mask: np.ndarray) -> bool:
        """T3: Return True if mask contains more than one connected component."""
        n_labels, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return n_labels > 2  # label 0 = background; >1 real component → multi

    def _multistage_filter(
        self,
        pool: List[Tuple[np.ndarray, dict]],
    ) -> List[dict]:
        """T3: Apply two-pass filter to a pool of (mask, room_dict) tuples.

        Coarse pass:
          1. Drop entries whose mask contains more than one connected component.
          2. Among remaining, drop IoU >= 0.8 duplicates (keep higher sam_confidence).
        Fine pass (greedy covering set):
          3. Sort survivors by mask area descending.
          4. Greedily accept masks with mutual IoU <= 0.01 with all already-accepted.

        This method is a no-op (returns room dicts unchanged) when called with an
        empty pool or a pool of one entry.

        Args:
            pool: List of (binary_mask, room_dict) pairs. Masks must be same shape.

        Returns:
            List of room_dicts for the covering set. _mask keys stripped from dicts.
        """
        if len(pool) <= 1:
            result = [d.copy() for _, d in pool]
            for d in result:
                d.pop("_mask", None)
            return result

        # --- Coarse pass 1: drop multi-component masks ---
        survivors = [
            (m, d) for m, d in pool
            if not self._is_multi_component(m)
        ]
        if not survivors:
            survivors = list(pool)

        # --- Coarse pass 2: IoU >= 0.8 dedup ---
        # Prefer a typed detection (non-empty room_type) over a typeless density
        # prompt (BUG-2); only when both are typed-or-both-typeless fall back to
        # higher sam_confidence. This prevents a density_prompt from evicting a
        # classified room merely because SAM scored its mask higher.
        def _is_typed(d: dict) -> bool:
            return bool(d.get("room_type"))

        deduped: List[Tuple[np.ndarray, dict]] = []
        for mask_i, dict_i in survivors:
            dominated = False
            for j, (mask_j, dict_j) in enumerate(deduped):
                if self._iou_masks(mask_i, mask_j) >= 0.8:
                    typed_i, typed_j = _is_typed(dict_i), _is_typed(dict_j)
                    if typed_i != typed_j:
                        # Exactly one is typed → keep the typed one.
                        if typed_i:
                            deduped[j] = (mask_i, dict_i)
                    else:
                        # Both typed or both typeless → higher confidence wins.
                        conf_i = dict_i.get("sam_confidence", dict_i.get("confidence", 0.0))
                        conf_j = dict_j.get("sam_confidence", dict_j.get("confidence", 0.0))
                        if conf_i > conf_j:
                            deduped[j] = (mask_i, dict_i)
                    dominated = True
                    break
            if not dominated:
                deduped.append((mask_i, dict_i))

        # --- Fine pass: greedy covering set (IoU <= 0.01) ---
        deduped.sort(key=lambda t: -int(t[0].sum()))  # largest area first
        covering: List[Tuple[np.ndarray, dict]] = []
        for mask_c, dict_c in deduped:
            if all(self._iou_masks(mask_c, sel_m) <= 0.01 for sel_m, _ in covering):
                covering.append((mask_c, dict_c))

        result = [d.copy() for _, d in covering]
        for d in result:
            d.pop("_mask", None)
        return result

    def _density_peak_prompts(
        self,
        gray: np.ndarray,
        exclusion_zones: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[Tuple[int, int]]:
        """T2: Extract SAM prompt points from Sobel gradient density peaks.

        Computes Sobel magnitude, thresholds at tau_factor * mean, labels
        connected peaks, takes their centroids, applies minimum spacing and
        BOM-zone exclusion, then caps at density_max_prompts.

        Args:
            gray: Grayscale image as uint8 numpy array (H, W).
            exclusion_zones: List of (x1, y1, x2, y2) rects to suppress.
                Prompt points inside any zone are dropped.

        Returns:
            List of (x, y) prompt point tuples, capped at density_max_prompts.
        """
        gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)

        tau = self.config.density_tau_factor * mag.mean()
        peaks_mask = (mag > tau).astype(np.uint8)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            peaks_mask, connectivity=8
        )

        # Collect (magnitude_sum, cx, cy) per component (skip background)
        candidates = []
        for lbl in range(1, n_labels):
            cx = int(centroids[lbl][0])
            cy = int(centroids[lbl][1])
            mag_sum = float(mag[labels == lbl].sum())
            candidates.append((mag_sum, cx, cy))

        # Sort by magnitude descending so strongest peaks are selected first
        candidates.sort(key=lambda t: -t[0])

        # Greedy minimum-spacing filter
        min_sp = self.config.density_min_spacing
        kept: List[Tuple[int, int]] = []
        for _, cx, cy in candidates:
            too_close = any(
                abs(cx - kx) < min_sp and abs(cy - ky) < min_sp
                for kx, ky in kept
            )
            if too_close:
                continue

            # BOM-zone exclusion: drop points inside any exclusion rect (x1,y1,x2,y2)
            if exclusion_zones:
                in_zone = any(
                    x1 <= cx <= x2 and y1 <= cy <= y2
                    for x1, y1, x2, y2 in exclusion_zones
                )
                if in_zone:
                    continue

            kept.append((cx, cy))
            if len(kept) >= self.config.density_max_prompts:
                break

        return kept

    def _mask_to_bbox(
        self, mask: np.ndarray
    ) -> Tuple[List[int], Optional[np.ndarray]]:
        """Convert binary mask to bounding box and contour.

        Thin delegate to the module-level `mask_to_bbox` (shared with
        sam21_probe's SAM2.1 adapter).
        """
        return mask_to_bbox(mask)

    def refine_annotations(
        self,
        image_path: str | Path,
        annotations: List[dict],
        img_width: int = 0,
        img_height: int = 0,
        max_expand_frac: float = 0.25,
        exclusion_zones: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[dict]:
        """
        Expand label-sized bboxes to room-boundary bboxes using SAM.

        For each annotation, seeds SAM with the label centroid, segments the
        enclosing room, and replaces the label bbox with the room mask bbox.

        Guardrails (keep original on fail):
          - Seed centroid inside an exclusion_zones rect → BOM/title-block
            label, no room to find (checked before calling SAM at all)
          - SAM bbox area > max_expand_frac * image_area → over-segmentation
            (SAM grabbed the full floor or a multi-room region)
          - SAM bbox area < original label area → collapse (SAM shrunk)

        Args:
            image_path:      Path to the floor plan image.
            annotations:     List of annotation dicts with 'bbox' [x,y,w,h].
            img_width:       Image width in px (auto-read from file if 0).
            img_height:      Image height in px (auto-read from file if 0).
            max_expand_frac: Max fraction of image area any single SAM box may occupy.

        Returns:
            List of annotations with expanded bboxes (original preserved as
            'original_bbox'; sam_expanded=True marks successful expansions).
        """
        # Resolve image dimensions for guardrail checks
        if img_width == 0 or img_height == 0:
            try:
                from PIL import Image as _PIL
                with _PIL.open(image_path) as _img:
                    img_width, img_height = _img.size
            except Exception:
                img_width, img_height = 4500, 3375  # safe fallback

        # Grayscale array for per-room content-density (FIX-1). Boxes over blank
        # space (hallucinations) have near-zero non-white density; real rooms
        # contain walls/fixtures and score higher.
        try:
            from PIL import Image as _PIL
            _gray = np.asarray(_PIL.open(image_path).convert("L"))
        except Exception:
            _gray = None

        img_area = img_width * img_height
        max_sam_area = img_area * max_expand_frac

        refined = []
        expanded = 0
        kept_original = 0
        # P2 (sprint1_verify47 audit, 2026-08-19): every SAM failure was
        # already logger.warning'd per-box, but 3,093 of them scrolled by
        # in one run without anyone noticing every single expansion had
        # failed -- the per-page summary line below was logger.info even
        # when expanded==0 for 57/57 pages. Track error causes so that
        # case gets one loud, impossible-to-miss line instead of relying
        # on someone counting per-box warnings by hand.
        error_reasons: Dict[str, int] = {}

        # Precompute every seed's centroid so each expansion can be rejected if it
        # floods past its own room and swallows a neighbouring seed (G1).
        seed_centers = [seed_center(a.get("bbox", [])) for a in annotations]

        for idx, raw_ann in enumerate(annotations):
            # A previous refine pass (re-run over an already-processed output
            # dir, or --skip-existing) may have left a "segmentation" polygon
            # on this dict. That polygon belongs to THAT run's bbox. Drop it up
            # front so every branch below either writes a matching polygon or
            # writes none -- a kept-original/cc_split result must never carry a
            # polygon that disagrees with its bbox.
            ann = {k: v for k, v in raw_ann.items() if k != "segmentation"}
            bbox = ann.get("bbox", [0, 0, 100, 100])

            # Centroid of label box → SAM prompt point
            if len(bbox) == 4:
                bx, by, bw, bh = bbox
                center_x = bx + bw // 2
                center_y = by + bh // 2
                original_area = max(1, bw * bh)
            else:
                center_x, center_y, original_area = 50, 50, 1

            # Guardrail 0: exclusion zone. exclusion_zones was accepted by
            # this method and threaded here from the pipeline (BOM tables,
            # title blocks) but was only ever consumed inside the
            # use_density_prompts branch below -- default off, so this
            # check never ran on the main label-centroid path. Confirmed
            # gap (sprint1_verify47 audit, 2026-08-19): a seed centroid
            # sitting on the word "ELECTRICAL" inside a sheet title
            # ("ELECTRICAL LIGHTING FIRST FLOOR PLAN") got expanded by SAM
            # into the entire title block (revision table, seal, sheet
            # number) -- 3.90% of the page, verified visually. Checked
            # BEFORE calling SAM (not after, like the other guardrails)
            # because there is nothing to salvage: a title-block seed has
            # no enclosing room to find. Same all-or-nothing shape as the
            # other guardrails: keep original, tag the reason, no new
            # concept introduced.
            if exclusion_zones and any(
                zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2
                for zx1, zy1, zx2, zy2 in exclusion_zones
            ):
                logger.debug(
                    f"SAM exclusion-zone guardrail: seed ({center_x},{center_y}) "
                    f"inside a BOM/title-block zone. Keeping original."
                )
                refined_ann = ann.copy()
                refined_ann["sam_expanded"] = False
                refined_ann["sam_skip_reason"] = "exclusion_zone"
                refined.append(refined_ann)
                kept_original += 1
                continue

            try:
                # T1: request mask for cc_split. T3: need mask for pool, but only
                # when density seeding is also active (T3 is a no-op otherwise).
                t3_active = self.config.use_multistage_filter and self.config.use_density_prompts
                need_mask = self.config.use_cc_split or t3_active
                # P-2B: for label-scale originals the box prompt confines SAM to
                # the label area, preventing expansion to room walls. Use point-only
                # prompt so SAM can grow to the enclosing room boundary. For large
                # originals (already room-scale or over-sized VLM boxes) keep the
                # box prompt as a coarse constraint against runaway segmentation.
                use_box = len(bbox) == 4 and should_use_box_prompt(original_area)
                result = self.segment_from_point(
                    image_path, (center_x, center_y), include_mask=need_mask,
                    label_bbox=bbox if use_box else None,
                )

                rx, ry, rw, rh = result.bbox
                sam_area = rw * rh

                # Guardrail 1: over-segmentation
                if sam_area > max_sam_area:
                    logger.debug(
                        f"SAM over-seg guardrail: {sam_area:.0f} > {max_sam_area:.0f} "
                        f"({max_expand_frac:.0%} of image). Keeping original."
                    )
                    refined_ann = ann.copy()
                    refined_ann["sam_expanded"] = False
                    refined_ann["sam_skip_reason"] = "over_segmentation"
                    refined.append(refined_ann)
                    kept_original += 1
                    continue

                # G1: anti-flood — expansion that swallows a neighbour seed would
                # absorb that room's distinct label in the VLM/OCR merge, collapsing
                # two rooms into one. Keep the label box instead.
                if expansion_swallows_other_seed(list(result.bbox), idx, seed_centers):
                    logger.debug(
                        "SAM flood guardrail: expansion swallowed a neighbour seed. "
                        "Keeping original."
                    )
                    refined_ann = ann.copy()
                    refined_ann["sam_expanded"] = False
                    refined_ann["sam_skip_reason"] = "flood_swallowed_seed"
                    refined.append(refined_ann)
                    kept_original += 1
                    continue

                # Guardrail 2: collapse — only when the original was LABEL-SCALE.
                # A label box is small; SAM shrinking below it means SAM failed, so
                # keep the label. But when the original is already a giant mislocalized
                # box (VLM drew a schedule/notes region), SAM's SMALLER result is the
                # desired refinement — accepting it fixes the huge-box error.
                if is_collapse(sam_area, original_area):
                    logger.debug(
                        f"SAM collapse guardrail: sam_area={sam_area} < "
                        f"original={original_area} (label-scale). Keeping original."
                    )
                    refined_ann = ann.copy()
                    refined_ann["sam_expanded"] = False
                    refined_ann["sam_skip_reason"] = "collapse"
                    refined.append(refined_ann)
                    kept_original += 1
                    continue

                # Guardrail 3: label-scale no-op. SAM "grew" past the label
                # (else collapse above would have caught it) but never reached
                # room scale -- its own multimask candidates were text-glyph-sized
                # blobs, not the room. Rejected results keep the original label
                # box, same as every other guardrail here; this does not fix the
                # geometry, it stops a label blob from being reported as a
                # successful room expansion. See SAMConfig.min_expansion_area_px.
                if is_label_scale_result(sam_area, self.config.min_expansion_area_px):
                    logger.debug(
                        f"SAM label-scale guardrail: sam_area={sam_area} < "
                        f"{self.config.min_expansion_area_px} (min room scale). "
                        f"Keeping original."
                    )
                    refined_ann = ann.copy()
                    refined_ann["sam_expanded"] = False
                    refined_ann["sam_skip_reason"] = "label_scale_noop"
                    refined.append(refined_ann)
                    kept_original += 1
                    continue

                # T1: connected-component split when flag is set and mask available.
                if self.config.use_cc_split and result.mask is not None:
                    cc_bboxes = self._split_mask_instances(
                        result.mask,
                        min_area=self.config.cc_min_area,
                        erosion_px=self.config.cc_erosion_px,
                    )
                    if len(cc_bboxes) >= 2:
                        logger.debug(
                            f"T1 CC split: {len(cc_bboxes)} components from mask "
                            f"(original bbox area={sam_area:.0f})"
                        )
                        for cc_bbox in cc_bboxes:
                            split_ann = ann.copy()
                            split_ann["bbox"] = cc_bbox
                            split_ann["original_bbox"] = bbox
                            split_ann["sam_confidence"] = result.confidence
                            split_ann["sam_expanded"] = True
                            split_ann["cc_split"] = True
                            refined.append(split_ann)
                        expanded += len(cc_bboxes)
                        del result.mask  # release memory
                        continue
                    # Single component or empty — fall through to normal path below.

                refined_ann = ann.copy()
                refined_ann["bbox"] = result.bbox
                refined_ann["original_bbox"] = bbox
                refined_ann["sam_confidence"] = result.confidence
                refined_ann["sam_expanded"] = True
                # Real room outline for COCO `segmentation` export. Page-pixel
                # absolute coords, same space as bbox (both derive from the same
                # mask via _mask_to_bbox). JSON-safe list-of-[x, y] -- never the
                # raw mask array, which the _mask strip below exists to keep out
                # of serialized annotations.
                if result.contour is not None:
                    refined_ann["segmentation"] = (
                        result.contour.reshape(-1, 2).tolist()
                    )
                # T3: keep mask in dict for pool collection; stripped after filter.
                if t3_active and result.mask is not None:
                    refined_ann["_mask"] = result.mask
                refined.append(refined_ann)
                expanded += 1

            except Exception as e:
                logger.warning(f"SAM segment failed, keeping original: {e}")
                reason = "CUDA out of memory" if "out of memory" in str(e).lower() else type(e).__name__
                error_reasons[reason] = error_reasons.get(reason, 0) + 1
                refined_ann = ann.copy()
                refined_ann["sam_expanded"] = False
                refined_ann["sam_skip_reason"] = "error"
                refined.append(refined_ann)
                kept_original += 1

        # T2: density-peak supplementary prompts (EXPERIMENT — default off).
        # Runs after label-centroid loop; adds rooms at structurally dense image
        # locations not already covered by an existing annotation centroid.
        if self.config.use_density_prompts and _gray is not None:
            peak_points = self._density_peak_prompts(_gray, exclusion_zones)

            # Build set of existing centroids to avoid re-seeding known rooms
            existing_centroids = []
            for ann in refined:
                b = ann.get("bbox", [])
                if len(b) == 4:
                    bx, by, bw, bh = b
                    existing_centroids.append((bx + bw // 2, by + bh // 2))

            for px, py in peak_points:
                # Skip if too close to any existing annotation centroid (20px)
                if any(abs(px - ex) < 20 and abs(py - ey) < 20 for ex, ey in existing_centroids):
                    continue
                try:
                    res = self.segment_from_point(
                        image_path, (px, py), include_mask=self.config.use_multistage_filter,
                    )
                    rx, ry, rw, rh = res.bbox
                    dp_area = rw * rh
                    if dp_area > max_sam_area or dp_area == 0:
                        continue
                    dp_ann: dict = {
                        "bbox": res.bbox,
                        "room_type": "",
                        "room_name": "",
                        "confidence": res.confidence,
                        "sam_expanded": True,
                        "sam_confidence": res.confidence,
                        "source": "density_prompt",
                    }
                    if res.mask is not None:
                        dp_ann["_mask"] = res.mask  # used by T3 pool; stripped later
                    refined.append(dp_ann)
                    existing_centroids.append((px, py))
                    expanded += 1
                except Exception as e:
                    logger.debug(f"T2 density-peak SAM failed at ({px},{py}): {e}")

        # T3: multi-stage mask pool filter. No-op unless BOTH multistage AND density
        # seeding are active (density is what creates the enlarged pool worth filtering;
        # label-only rooms are already deduped by pipeline._dedup_rooms_by_iou).
        if self.config.use_multistage_filter and self.config.use_density_prompts:
            pool_entries = [(d["_mask"], d) for d in refined if "_mask" in d]
            no_mask_entries = [d for d in refined if "_mask" not in d]
            if pool_entries:
                filtered = self._multistage_filter(pool_entries)
                refined = no_mask_entries + filtered
                logger.info(
                    f"T3 multistage filter: {len(pool_entries)} pool → "
                    f"{len(filtered)} after filter"
                )

        # Safety: strip any residual _mask (numpy array) before annotations are
        # JSON-serialized downstream. Guards against BUG-3 (ndarray not serializable).
        for d in refined:
            d.pop("_mask", None)

        # FIX-1: attach content-density (non-white fraction) on each room's final
        # bbox. Consumed by the SFT room-scale gate to drop blank-space boxes.
        if _gray is not None:
            for ann in refined:
                b = ann.get("bbox", [])
                if len(b) == 4:
                    x, y, w, h = [int(v) for v in b]
                    x2, y2 = min(x + w, _gray.shape[1]), min(y + h, _gray.shape[0])
                    x, y = max(0, x), max(0, y)
                    if x2 > x and y2 > y:
                        sub = _gray[y:y2, x:x2]
                        ann["content_density"] = float((sub < 200).sum()) / sub.size
                    else:
                        ann["content_density"] = 0.0

        logger.info(
            f"SAM refine_annotations: {expanded} expanded, "
            f"{kept_original} kept original (of {len(annotations)} total)"
        )
        # P2: a page where NOTHING expanded because EVERY box errored
        # (not because the guardrails correctly rejected bad expansions)
        # is a resource/environment failure, not a normal outcome -- log
        # it as loud as the OOMs that caused it, once, with a cause
        # breakdown, instead of leaving it indistinguishable from a
        # quiet, healthy "0 expanded" page in the per-page INFO line.
        if expanded == 0 and annotations and sum(error_reasons.values()) == len(annotations):
            logger.warning(
                f"SAM refine_annotations: ALL {len(annotations)} boxes failed with "
                f"an error (0 expanded) -- likely a resource problem, not guardrail "
                f"rejection. Causes: {dict(error_reasons)}"
            )
        return refined

    def visualize_segmentation(
        self,
        image_path: str | Path,
        results: List[SegmentationResult],
        output_path: Optional[str | Path] = None,
    ) -> np.ndarray:
        """
        Visualize segmentation results on the image.

        Args:
            image_path: Path to the original image.
            results: List of SegmentationResult objects.
            output_path: Optional path to save visualization.

        Returns:
            Visualization image as numpy array.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise SAMSegmentationError(f"Failed to load image: {image_path}")

        overlay = image.copy()

        # Draw each segmentation
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        for i, result in enumerate(results):
            color = colors[i % len(colors)]

            # Draw bounding box
            x, y, w, h = result.bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

            # Draw prompt point
            cv2.circle(overlay, result.prompt_point, 5, color, -1)

            # Draw mask if available
            if result.mask is not None:
                mask_color = np.zeros_like(image)
                mask_color[result.mask] = color
                overlay = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)

        if output_path:
            cv2.imwrite(str(output_path), overlay)
            logger.info(f"Saved visualization to {output_path}")

        return overlay


# P-2B: label-scale originals get a point-only prompt (so SAM can grow past the
# label to the enclosing room boundary); already room/VLM-scale originals keep a
# box prompt as a coarse constraint against runaway segmentation. Shared with
# sam21_probe so the SAM2.1 comparison exercises the identical prompt decision.
LABEL_SCALE_BOX_MAX = 50_000  # px²; ~224x224, generous label ceiling


def should_use_box_prompt(original_area: int) -> bool:
    """True when `original_area` is large enough that SAM should be box-constrained
    rather than given a bare point prompt."""
    return original_area > LABEL_SCALE_BOX_MAX


# Guardrail 2 (collapse): a label box is small; SAM shrinking below it means SAM
# failed, so the caller should keep the label. When the original is already a
# giant mislocalized box (VLM drew a schedule/notes region), SAM's smaller result
# is the desired refinement instead. Shared with sam21_probe so the SAM2.1
# comparison can classify collapse the same way production does.
LABEL_SCALE_MAX = 150_000  # px²; ~387x387, generous text-label ceiling


def is_collapse(sam_area: int, original_area: int) -> bool:
    """True when SAM's result should be rejected as a collapse: it shrank below
    a label-scale original (SAM failed to find the room, not a refinement)."""
    return sam_area < original_area and original_area <= LABEL_SCALE_MAX


def is_label_scale_result(sam_area: int, min_expansion_area_px: int) -> bool:
    """True when SAM's result grew (passed the collapse check above) but never
    left label scale -- a room-shaped point prompt whose only candidates were
    text-glyph-sized blobs. This is NOT a correctness check: an area at or
    above `min_expansion_area_px` is not verified as a real room, only as not
    provably a label blob (see SAMConfig.min_expansion_area_px)."""
    return sam_area < min_expansion_area_px


def seed_center(bbox: List[int]) -> Optional[Tuple[int, int]]:
    """Centroid (x, y) of a label bbox [x, y, w, h], or None if malformed."""
    if not bbox or len(bbox) != 4:
        return None
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)


def expansion_swallows_other_seed(
    sam_bbox: List[int], own_index: int, seed_centers: List[Optional[Tuple[int, int]]]
) -> bool:
    """True if the expanded SAM box contains a *different* seed's centroid.

    Such an expansion flooded past its own room into a neighbour. In the
    downstream VLM/OCR merge (overlap-based), that oversized box would absorb the
    neighbour's distinct label, collapsing two real rooms into one — the measured
    cause of the room-count regression. Reject it and keep the label box.
    """
    sx, sy, sw, sh = sam_bbox
    for i, c in enumerate(seed_centers):
        if c is None or i == own_index:
            continue
        cx, cy = c
        if sx <= cx <= sx + sw and sy <= cy <= sy + sh:
            return True
    return False


def select_room_mask(
    masks: np.ndarray, scores: np.ndarray, max_expand_frac: float
) -> Tuple[np.ndarray, float]:
    """
    Select a room-scale mask from SAM multimask candidates (F-A).

    SAM scores favour the largest (whole-floor) mask on floor plans.
    Instead: among masks whose area is below max_expand_frac of the image,
    pick the highest-scoring. If none qualify (all are floor-plate-sized),
    fall back to the smallest mask available.

    Args:
        masks: (N, H, W) boolean masks from predictor.predict.
        scores: (N,) confidence scores.
        max_expand_frac: max fraction of image area a mask may occupy.

    Returns:
        (selected_mask, confidence).
    """
    img_area = masks.shape[1] * masks.shape[2]
    max_area = img_area * max_expand_frac

    areas = [int(m.sum()) for m in masks]

    # Candidates below the over-segmentation ceiling
    valid = [i for i, a in enumerate(areas) if 0 < a <= max_area]
    if valid:
        best_idx = max(valid, key=lambda i: scores[i])
    else:
        # All masks too large -> take the smallest non-empty one
        non_empty = [i for i, a in enumerate(areas) if a > 0]
        if not non_empty:
            return masks[0], float(scores[0])
        best_idx = min(non_empty, key=lambda i: areas[i])

    return masks[best_idx], float(scores[best_idx])


def mask_to_bbox(mask: np.ndarray) -> Tuple[List[int], Optional[np.ndarray]]:
    """
    Convert binary mask to bounding box and contour.

    Args:
        mask: Binary segmentation mask.

    Returns:
        Tuple of (bbox as [x, y, w, h], largest contour).
    """
    mask_uint8 = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return [0, 0, 0, 0], None

    # Get largest contour
    largest = max(contours, key=cv2.contourArea)

    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest)

    return [x, y, w, h], largest
