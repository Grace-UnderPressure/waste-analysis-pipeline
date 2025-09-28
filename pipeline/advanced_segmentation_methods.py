#!/usr/bin/env python3
"""
Advanced segmentation methods
Optimizations inspired by SAM/FastSAM research and best practices
"""

import numpy as np
import cv2
import torch
import time
from typing import List, Dict, Tuple, Optional
from segmenter import Segmenter
from mask_postprocess import mask_postprocess
import logging

class AdvancedSegmenter:
    """
    Advanced segmenter: integrates multiple optimization strategies
    """
    
    def __init__(self, base_segmenter: Segmenter, device: str = "cuda"):
        self.base_segmenter = base_segmenter
        self.device = device
        self.logger = logging.getLogger(__name__)
        
    def multi_scale_segmentation(self, image: np.ndarray, scales: List[float] = [0.8, 1.0, 1.2], use_postprocess: bool = True) -> List[Dict]:
        """
        Multi-scale segmentation and fusion (optimized)
        """
        self.logger.info(f"Starting multi-scale segmentation, scales: {scales}")
        all_masks = []
        
        # Get resized image shape
        resized_image = self.base_segmenter._resize_image(image)
        h, w = resized_image.shape[:2]
        
        for scale in scales:
            # Scale image
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_image = cv2.resize(resized_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Segment
            masks = self.base_segmenter.segment(scaled_image, use_postprocess=use_postprocess)
            
            # Resize back to resized shape
            for mask in masks:
                # Resize mask
                scaled_mask = cv2.resize(mask["segmentation"].astype(np.float32), (w, h))
                scaled_mask = (scaled_mask > 0.5).astype(np.uint8)
                
                # Resize bbox based on resized image shape
                x0, y0, x1, y1 = mask["bbox"]
                new_bbox = [
                    max(0, int(x0 * w / new_w)), max(0, int(y0 * h / new_h)),
                    min(w, int(x1 * w / new_w)), min(h, int(y1 * h / new_h))
                ]
                
                # Ensure bbox is valid
                if new_bbox[2] > new_bbox[0] and new_bbox[3] > new_bbox[1]:
                    # Recompute area
                    new_area = int(scaled_mask.sum())
                    
                    all_masks.append({
                        **mask,
                        "segmentation": scaled_mask,
                        "bbox": new_bbox,
                        "area": new_area,
                        "scale": scale
                    })
        
        self.logger.info(f"Multi-scale segmentation finished, generated {len(all_masks)} masks")
        
        # Smart fusion and deduplication
        if len(all_masks) > 0:
            all_masks = self._smart_mask_fusion(all_masks, resized_image)
            self.logger.info(f"After smart fusion, {len(all_masks)} masks remain")
        
        return all_masks
    
    def confidence_weighted_fusion(self, masks_with_scores: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        Confidence-weighted mask fusion
        """
        if not masks_with_scores:
            return np.array([])
        
        # Get mask shape
        mask_shape = masks_with_scores[0][0].shape
        weighted_sum = np.zeros(mask_shape, dtype=np.float32)
        total_weight = 0.0
        
        for mask, score in masks_with_scores:
            weighted_sum += mask.astype(np.float32) * score
            total_weight += score
        
        if total_weight > 0:
            final_mask = weighted_sum / total_weight
            return (final_mask > 0.5).astype(np.uint8)
        else:
            return np.zeros(mask_shape, dtype=np.uint8)
    
    def adaptive_prompt_selection(self, image: np.ndarray, region: Dict) -> str:
        """
        Adaptive prompt selection
        """
        # Analyze region features
        mask = region["segmentation"]
        area = region["area"]
        bbox = region["bbox"]
        
        # Compute aspect ratio
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        
        # Compute complexity (boundary length / area)
        from skimage import measure
        contours = measure.find_contours(mask.astype(float), 0.5)
        perimeter = sum(len(contour) for contour in contours) if contours else 0
        complexity = perimeter / (area + 1e-6)
        
        # Select prompt type based on features
        if area < 1000:
            return 'point_prompts'
        elif aspect_ratio > 3:
            return 'bbox_prompts'
        elif complexity > 0.3:
            return 'text_prompts'
        else:
            return 'hybrid_prompts'
    
    def evaluate_mask_quality(self, mask: np.ndarray, image: np.ndarray) -> Dict:
        """
        Evaluate mask quality
        """
        metrics = {}
        
        # 1) Boundary smoothness
        from skimage import measure
        contours = measure.find_contours(mask.astype(float), 0.5)
        if contours:
            perimeter = sum(len(contour) for contour in contours)
            area = mask.sum()
            metrics['boundary_smoothness'] = 1.0 / (perimeter / (area + 1e-6) + 1e-6)
        else:
            metrics['boundary_smoothness'] = 0.0
        
        # 2) Internal consistency - handle size mismatch
        try:
            # Ensure mask and image sizes match
            if mask.shape[:2] != image.shape[:2]:
                # Resize mask to image size
                mask_resized = cv2.resize(mask.astype(np.float32), (image.shape[1], image.shape[0]))
                mask_resized = (mask_resized > 0.5).astype(bool)
            else:
                mask_resized = mask.astype(bool)
            
            mask_region = image[mask_resized]
            if mask_region.size > 0:
                color_std = np.std(mask_region, axis=0).mean()
                metrics['internal_consistency'] = 1.0 / (color_std + 1e-6)
            else:
                metrics['internal_consistency'] = 0.0
        except Exception as e:
            # Use default value if any error occurs
            metrics['internal_consistency'] = 0.5
        
        # 3) Shape reasonableness
        bbox = self._get_bbox_from_mask(mask)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        metrics['shape_reasonableness'] = 1.0 / (aspect_ratio + 1e-6)
        
        # 4) Overall score
        metrics['overall_score'] = (
            metrics['boundary_smoothness'] * 0.3 +
            metrics['internal_consistency'] * 0.3 +
            metrics['shape_reasonableness'] * 0.4
        )
        
        return metrics
    
    def quality_driven_improvement(self, masks: List[Dict], image: np.ndarray) -> List[Dict]:
        """Quality-driven mask improvement"""
        improved_masks = []
        
        for mask in masks:
        # Evaluate quality
            quality = self.evaluate_mask_quality(mask["segmentation"], image)
            
        # Decide whether to improve based on quality score
            if quality['overall_score'] < 0.7:
            # Try to improve
                improved_mask = self._improve_mask(mask, image)
                if improved_mask is not None:
                    improved_masks.append(improved_mask)
            else:
                improved_masks.append(mask)
        
        return improved_masks
    
    def _improve_mask(self, mask: Dict, image: np.ndarray) -> Optional[Dict]:
        """Improve a single mask"""
        # Morphological improvements
        seg = mask["segmentation"].astype(np.uint8)
        
        # Closing to fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        improved_seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel)
        
        # Opening to remove small noise
        improved_seg = cv2.morphologyEx(improved_seg, cv2.MORPH_OPEN, kernel)
        
        # Recompute bbox and area
        ys, xs = np.where(improved_seg)
        if xs.size == 0 or ys.size == 0:
            return None
        
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        new_area = int(improved_seg.sum())
        
        return {
            **mask,
            "segmentation": improved_seg,
            "bbox": [x0, y0, x1, y1],
            "area": new_area
        }
    
    def _get_bbox_from_mask(self, mask: np.ndarray) -> List[int]:
        """Compute bounding box from mask"""
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            return [0, 0, 0, 0]
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    
    def _smart_mask_fusion(self, masks: List[Dict], image: np.ndarray) -> List[Dict]:
        """
        Smart mask fusion to address over-segmentation and overlaps
        """
        if len(masks) <= 1:
            return masks
        
        # 1) Sort by area (descending)
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        
        # 2) Compute IoU matrix among masks
        n = len(masks)
        iou_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                mask1 = masks[i]["segmentation"].astype(bool)
                mask2 = masks[j]["segmentation"].astype(bool)
                
                intersection = np.logical_and(mask1, mask2).sum()
                union = np.logical_or(mask1, mask2).sum()
                
                if union > 0:
                    iou_matrix[i, j] = intersection / union
                    iou_matrix[j, i] = iou_matrix[i, j]
        
        # 3) Group and fuse masks based on IoU
        fused_masks = []
        used_indices = set()
        
        for i in range(n):
            if i in used_indices:
                continue
            
            # Find masks highly overlapping with current mask
            similar_masks = [i]
            used_indices.add(i)
            
            for j in range(i+1, n):
                if j in used_indices:
                    continue
                
                iou = iou_matrix[i, j]
                # Enhanced overlap detection
                if self._should_fuse_masks(masks[i], masks[j], iou):
                    similar_masks.append(j)
                    used_indices.add(j)
            
            # Fuse similar mask group
            if len(similar_masks) == 1:
                # Single mask, keep as-is
                fused_masks.append(masks[i])
            else:
                # Multiple similar masks, perform fusion
                fused_mask = self._fuse_similar_masks([masks[idx] for idx in similar_masks])
                if fused_mask is not None:
                    fused_masks.append(fused_mask)
        
        # 4) Filter small/background-like masks
        filtered_masks = []
        h, w = image.shape[:2]
        min_area = h * w * 0.001  # minimum area: 0.1% of image area
        
        for mask in fused_masks:
            area = mask["area"]
            if area >= min_area:
                # Compute mask compactness
                seg = mask["segmentation"].astype(bool)
                from skimage import measure
                contours = measure.find_contours(seg.astype(float), 0.5)
                if contours:
                    perimeter = sum(len(contour) for contour in contours)
                    compactness = perimeter / (area + 1e-6)
                    # Filter overly complex masks (likely background texture)
                    if compactness < 0.8:
                        # Ensure bbox within image bounds
                        bbox = mask["bbox"]
                        x0, y0, x1, y1 = bbox
                        bbox = [
                            max(0, min(x0, w-1)),
                            max(0, min(y0, h-1)),
                            max(x0+1, min(x1, w)),
                            max(y0+1, min(y1, h))
                        ]
                        mask["bbox"] = bbox
                        filtered_masks.append(mask)
                else:
                    # Ensure bbox within image bounds
                    bbox = mask["bbox"]
                    x0, y0, x1, y1 = bbox
                    bbox = [
                        max(0, min(x0, w-1)),
                        max(0, min(y0, h-1)),
                        max(x0+1, min(x1, w)),
                        max(y0+1, min(y1, h))
                    ]
                    mask["bbox"] = bbox
                    filtered_masks.append(mask)
        
        # 5) Post-process to remove contained masks
        filtered_masks = self._remove_contained_masks(filtered_masks)
        
        return filtered_masks
    
    def _should_fuse_masks(self, mask1: Dict, mask2: Dict, iou: float) -> bool:
        """
        Enhanced logic to determine if two masks should be fused
        """
        # Basic IoU threshold (lowered for more aggressive fusion)
        if iou > 0.3:
            return True
        
        # Check for containment relationship
        area1, area2 = mask1["area"], mask2["area"]
        if area1 == 0 or area2 == 0:
            return False
            
        # Calculate intersection area
        seg1 = mask1["segmentation"].astype(bool)
        seg2 = mask2["segmentation"].astype(bool)
        intersection = np.logical_and(seg1, seg2).sum()
        
        # Check if one mask is mostly contained in the other
        containment_ratio1 = intersection / area1  # How much of mask1 is in mask2
        containment_ratio2 = intersection / area2  # How much of mask2 is in mask1
        
        # If one mask is >80% contained in the other, fuse them
        if containment_ratio1 > 0.8 or containment_ratio2 > 0.8:
            return True
        
        # Check for significant overlap even with low IoU
        if iou > 0.15 and (containment_ratio1 > 0.3 or containment_ratio2 > 0.3):
            return True
            
        return False
    
    def _remove_contained_masks(self, masks: List[Dict]) -> List[Dict]:
        """
        Remove masks that are completely or mostly contained within other masks
        """
        if len(masks) <= 1:
            return masks
        
        # Sort by area (descending)
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        
        filtered_masks = []
        for i, mask1 in enumerate(masks):
            is_contained = False
            
            for j, mask2 in enumerate(masks):
                if i == j or mask2["area"] <= mask1["area"]:
                    continue
                
                # Check if mask1 is contained in mask2
                seg1 = mask1["segmentation"].astype(bool)
                seg2 = mask2["segmentation"].astype(bool)
                intersection = np.logical_and(seg1, seg2).sum()
                
                # If mask1 is >70% contained in mask2, remove mask1
                containment_ratio = intersection / mask1["area"] if mask1["area"] > 0 else 0
                if containment_ratio > 0.7:
                    is_contained = True
                    break
            
            if not is_contained:
                filtered_masks.append(mask1)
        
        return filtered_masks
    
    def _fuse_similar_masks(self, similar_masks: List[Dict]) -> Optional[Dict]:
        """
        Fuse a group of similar masks
        """
        if not similar_masks:
            return None
        
        if len(similar_masks) == 1:
            return similar_masks[0]
        
        # Use the largest mask as base
        base_mask = max(similar_masks, key=lambda x: x["area"])
        base_seg = base_mask["segmentation"].astype(bool)
        
        # Merge all masks
        fused_seg = base_seg.copy()
        for mask in similar_masks:
            if mask is not base_mask:
                fused_seg = np.logical_or(fused_seg, mask["segmentation"].astype(bool))
        
        # Morphological close to fill holes
        fused_seg = fused_seg.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fused_seg = cv2.morphologyEx(fused_seg, cv2.MORPH_CLOSE, kernel)
        
        # Recompute bbox and area
        ys, xs = np.where(fused_seg)
        if xs.size == 0 or ys.size == 0:
            return None
        
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        
        # Ensure bbox within image bounds
        h, w = fused_seg.shape[:2]
        x0 = max(0, min(x0, w-1))
        y0 = max(0, min(y0, h-1))
        x1 = max(x0+1, min(x1, w))
        y1 = max(y0+1, min(y1, h))
        
        new_area = int(fused_seg.sum())
        
        return {
            "region_id": base_mask["region_id"],
            "segmentation": fused_seg,
            "bbox": [x0, y0, x1, y1],
            "area": new_area,
            "scale": base_mask.get("scale", 1.0),
            "confidence": base_mask.get("confidence", 0.5)
        }


def create_advanced_segmentation_pipeline(config: Dict) -> AdvancedSegmenter:
    """
    Create advanced segmentation pipeline
    """
    # Create base segmenter
    base_segmenter = Segmenter(
        model_type=config.get("model_type", "fastsam"),
        model_path=config.get("model_path", "models/FastSAM-s.pt"),
        device=config.get("device", "cuda"),
        min_mask_area=config.get("min_mask_area", 3600),
        input_resize=config.get("input_resize", 1024),
        max_objects=config.get("max_objects", 20)
    )
    
    # Create advanced segmenter
    advanced_segmenter = AdvancedSegmenter(base_segmenter, config.get("device", "cuda"))
    
    return advanced_segmenter


# Example usage
if __name__ == "__main__":
    # Config
    config = {
        "model_type": "fastsam",
        "model_path": "models/FastSAM-s.pt",
        "device": "cuda",
        "min_mask_area": 3600,
        "input_resize": 1024,
        "max_objects": 20,
        "use_sam": False,
        "use_fastsam": True,
        "ensemble_weights": [1.0]
    }
    
    # Create segmenter
    advanced_segmenter = create_advanced_segmentation_pipeline(config)
    
    # Test image
    image = cv2.imread("input_images/test1.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Multi-scale segmentation
    masks = advanced_segmenter.multi_scale_segmentation(image_rgb)
    
    print(f"Generated {len(masks)} masks")
    
    # Quality evaluation (first 5 masks)
    for i, mask in enumerate(masks[:5]):
        quality = advanced_segmenter.evaluate_mask_quality(mask["segmentation"], image_rgb)
        print(f"Mask {i}: overall score = {quality['overall_score']:.3f}") 