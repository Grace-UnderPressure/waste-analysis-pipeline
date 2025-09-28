import numpy as np
import cv2
from skimage import measure
from skimage.measure import label as sk_label
from skimage.morphology import binary_closing, binary_opening
from skimage.filters import threshold_otsu
import logging

# =====================
# Core mask post-processing functions
# =====================

def filter_by_area(masks, min_area=2000, max_area_ratio=0.8):
    """
    Filter masks by area
    """
    if not masks:
        return []
    
    # Estimate image size (from the first mask)
    first_mask = masks[0]["segmentation"]
    H, W = first_mask.shape
    max_area = H * W * max_area_ratio
    
    filtered = []
    for mask in masks:
        area = mask["area"]
        if min_area <= area <= max_area:
            filtered.append(mask)
    
    return filtered

def filter_by_aspect_ratio(masks, max_ratio=12):
    """
    Filter masks by aspect ratio. Extremely thin/flat masks will be removed.
    Args:
        max_ratio: Maximum aspect ratio. Lower to drop very thin masks; higher to keep elongated objects.
    """
    filtered = []
    for m in masks:
        x0, y0, x1, y1 = m["bbox"]
        w, h = x1 - x0, y1 - y0
        if w == 0 or h == 0:
            continue
        ratio = max(w, h) / (min(w, h) + 1e-6)
        if ratio <= max_ratio:
            filtered.append(m)
    return filtered

def morph_close_masks(masks, kernel_size=5, iterations=2):
    """
    Apply morphological closing to fill small holes and smooth boundaries
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    processed = []
    
    for mask in masks:
        seg = mask["segmentation"].astype(np.uint8)
        
        # Morphological closing
        for _ in range(iterations):
            seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel)
        
        # Update mask info
        new_area = seg.sum()
        if new_area > 0:
            ys, xs = np.where(seg)
            if xs.size > 0 and ys.size > 0:
                x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                processed.append({
                    **mask,
                    "segmentation": seg,
                    "area": int(new_area),
                    "bbox": [x0, y0, x1, y1]
                })
    
    return processed

def merge_overlapping_masks(masks, overlap_thresh=0.7):
    """
    Merge overlapping masks
    """
    if len(masks) <= 1:
        return masks
    
    # Sort by area in descending order
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    merged = []
    used = set()
    
    for i, mask1 in enumerate(masks):
        if i in used:
            continue
        
        current_group = [mask1]
        used.add(i)
        
        for j, mask2 in enumerate(masks[i+1:], i+1):
            if j in used:
                continue
            
            # Compute overlap
            seg1 = mask1["segmentation"].astype(bool)
            seg2 = mask2["segmentation"].astype(bool)
            
            intersection = np.logical_and(seg1, seg2).sum()
            union = np.logical_or(seg1, seg2).sum()
            
            if union > 0 and intersection / union > overlap_thresh:
                current_group.append(mask2)
                used.add(j)
        
        # Merge all masks in the current group
        if len(current_group) > 1:
            merged_mask = merge_mask_group(current_group)
            if merged_mask:
                merged.append(merged_mask)
        else:
            merged.append(mask1)
    
    return merged

def merge_mask_group(mask_group):
    """
    Merge a group of masks
    """
    if not mask_group:
        return None
    
    # Union of all masks
    combined_seg = np.zeros_like(mask_group[0]["segmentation"], dtype=bool)
    for mask in mask_group:
        combined_seg = np.logical_or(combined_seg, mask["segmentation"].astype(bool))
    
    # Recompute bbox and area
    ys, xs = np.where(combined_seg)
    if xs.size == 0 or ys.size == 0:
        return None
    
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    area = combined_seg.sum()
    
    return {
        "region_id": mask_group[0]["region_id"],
        "segmentation": combined_seg.astype(np.uint8),
        "bbox": [x0, y0, x1, y1],
        "area": int(area)
    }

def iterative_merge_masks(masks, iou_thresh=0.2, max_iter=5):
    """
    Iteratively merge overlapping masks until count stabilizes or max iterations reached.
    Args:
        iou_thresh: IoU threshold for merging.
        max_iter: Maximum iterations.
    """
    prev_len = -1
    for _ in range(max_iter):
        masks = merge_overlapping_masks(masks, iou_thresh)
        if len(masks) == prev_len:
            break
        prev_len = len(masks)
    return masks

def mask_nms(masks, nms_thresh=0.5):
    """
    Apply Non-Maximum Suppression (NMS) on masks
    """
    if len(masks) <= 1:
        return masks
    
    # 按面积排序
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    keep = []
    
    for i, mask1 in enumerate(masks):
        keep_mask = True
        
        for j, mask2 in enumerate(keep):
            # 计算IoU
            seg1 = mask1["segmentation"].astype(bool)
            seg2 = mask2["segmentation"].astype(bool)
            
            intersection = np.logical_and(seg1, seg2).sum()
            union = np.logical_or(seg1, seg2).sum()
            
            if union > 0 and intersection / union > nms_thresh:
                keep_mask = False
                break
        
        if keep_mask:
            keep.append(mask1)
    
    return keep

def filter_by_contrast(masks, image, min_contrast=18):
    """
    Filter masks by contrast to remove background/low-contrast regions.
    Args:
        min_contrast: Minimum std; higher is stricter, lower keeps low-contrast objects.
    """
    if image is None:
        return masks
    filtered = []
    removed = []
    for m in masks:
        x0, y0, x1, y1 = m["bbox"]
        crop = image[y0:y1, x0:x1]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            removed.append(m)
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        std = gray.std()
        if std < min_contrast:
            removed.append({**m, "contrast": std})
            continue
        filtered.append(m)
    if removed:
        print(f"[filter_by_contrast] Removed {len(removed)} low-contrast masks, std: {[round(m.get('contrast', 0),2) for m in removed]}")
    return filtered

def filter_by_contrast_fastsam(masks, min_contrast=15):
    """
    FastSAM-specific contrast filtering
    Remove low-contrast masks (e.g., large uniform background)
    """
    filtered = []
    for mask in masks:
        seg = mask["segmentation"].astype(bool)
        if seg.sum() == 0:
            continue
        
        # Use std within mask region as contrast metric
        ys, xs = np.where(seg)
        if xs.size == 0 or ys.size == 0:
            continue
        
        # Needs original image; currently skipped
        # In actual use, pass in the original image
        filtered.append(mask)
    
    return filtered

def filter_by_color_variance_fastsam(masks, min_variance=50):
    """
    FastSAM-specific shape complexity filtering
    Remove masks with overly simple shapes (likely background/invalid)
    """
    filtered = []
    for mask in masks:
        seg = mask["segmentation"].astype(bool)
        if seg.sum() == 0:
            continue
        
        # Compute shape complexity for mask region
        from skimage import measure
        contours = measure.find_contours(seg.astype(float), 0.5)
        if contours:
            # Compute contour complexity
            perimeter = sum(len(contour) for contour in contours)
            area = seg.sum()
            complexity = perimeter / (area + 1e-6)
            
            # Filter overly simple masks (likely background)
            # Lower threshold to pass more masks
            if complexity > 0.05:  # 从0.1降低到0.05
                filtered.append(mask)
        else:
            filtered.append(mask)
    
    return filtered


def dynamic_min_area(image_shape, base_min_area=2000):
    """
    Dynamically adjust minimum area threshold based on image size
    """
    H, W = image_shape[:2]
    image_area = H * W
    
    # Adjust threshold by image area
    if image_area < 500000:  # small image
        return max(base_min_area // 2, 1000)
    elif image_area > 2000000:  # large image
        return base_min_area * 2
    else:
        return base_min_area

def merge_small_masks_into_large(masks, small_area_thresh=1000, overlap_thresh=0.3):
    """
    Merge small masks into larger masks
    """
    if len(masks) <= 1:
        return masks
    
    # Sort by area
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    keep = []
    removed = []
    
    for i, m in enumerate(masks):
        seg = m["segmentation"].astype(bool)
        is_covered = False
        
        for j, big in enumerate(masks):
            if j == i or big["area"] <= m["area"]:
                continue
            
            big_seg = big["segmentation"].astype(bool)
            inter = np.logical_and(seg, big_seg).sum()
            
            if inter / (seg.sum() + 1e-6) > overlap_thresh:
                is_covered = True
                break
        
        if not is_covered:
            keep.append(m)
        else:
            removed.append(m)
    
    if removed:
        print(f"[merge_small_masks_into_large] Merged/removed {len(removed)} small masks covered by larger masks")
    
    return keep


def keep_largest_connected_region(masks):
    """
    Keep only the largest connected region per mask; remove fragments
    """
    processed = []
    for m in masks:
        seg = m["segmentation"].astype(np.uint8)
        labeled = sk_label(seg, connectivity=1)
        
        if labeled.max() <= 1:
            processed.append(m)
            continue
        
        # Multiple components: keep the largest one
        max_area = 0
        max_label = 0
        for label_id in range(1, labeled.max() + 1):
            area = (labeled == label_id).sum()
            if area > max_area:
                max_area = area
                max_label = label_id
        
        new_seg = (labeled == max_label).astype(np.uint8)
        ys, xs = np.where(new_seg)
        
        if xs.size == 0 or ys.size == 0:
            continue
        
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        processed.append({
            **m,
            "segmentation": new_seg,
            "area": int(max_area),
            "bbox": [x0, y0, x1, y1]
        })
    
    return processed


# =====================
# Main mask post-processing pipeline
# =====================
def mask_postprocess(masks, min_area=2000, max_masks=None, 
                    morph_close=True, morph_kernel_size=5, morph_iterations=2,
                    merge_overlapping=True, overlap_thresh=0.7,
                    merge_small=True, small_area_thresh=1000,
                    nms=True, nms_thresh=0.5,
                    keep_connected=True,
                    filter_by_contrast=True, min_contrast=15,
                    filter_by_color_variance=True, min_variance=50):
    """
    Comprehensive mask post-processing (optimized for FastSAM)
    
    Args:
        masks: list of input masks
        min_area: minimum area threshold
        max_masks: maximum number of masks to keep
        morph_close: whether to apply morphological closing
        morph_kernel_size: kernel size for morphology
        morph_iterations: iterations for morphology
        merge_overlapping: whether to merge overlapping masks
        overlap_thresh: overlap threshold
        merge_small: whether to merge small masks into large ones
        small_area_thresh: area threshold for small masks
        nms: whether to apply NMS
        nms_thresh: NMS threshold
        keep_connected: whether to keep only the largest connected region
        filter_by_contrast: whether to filter by contrast
        min_contrast: minimum contrast (std)
        filter_by_color_variance: whether to filter by color variance/complexity
        min_variance: minimum color variance/complexity
    
    Returns:
        List of processed masks
    """
    if not masks:
        return []
    
    print(f"[mask_postprocess] Start processing {len(masks)} masks")
    
    # 1. Basic filtering
    masks = filter_by_area(masks, min_area)
    print(f"[mask_postprocess] After area filter: {len(masks)} masks")
    
    # 2. Contrast filter (FastSAM-specific)
    if filter_by_contrast:
        masks = filter_by_contrast_fastsam(masks, min_contrast)
        print(f"[mask_postprocess] After contrast filter: {len(masks)} masks")
    
    # 3. Color variance/shape complexity filter (FastSAM-specific)
    if filter_by_color_variance:
        masks = filter_by_color_variance_fastsam(masks, min_variance)
        print(f"[mask_postprocess] After complexity filter: {len(masks)} masks")
    
    # 4. Morphological processing
    if morph_close:
        masks = morph_close_masks(masks, kernel_size=morph_kernel_size, iterations=morph_iterations)
        print(f"[mask_postprocess] After morphology: {len(masks)} masks")
    
    # 5. Connected component processing
    if keep_connected:
        masks = keep_largest_connected_region(masks)
        print(f"[mask_postprocess] After connected-component keep-largest: {len(masks)} masks")
    
    # 6. Merge overlapping masks
    if merge_overlapping:
        masks = merge_overlapping_masks(masks, overlap_thresh)
        print(f"[mask_postprocess] After overlap-merge: {len(masks)} masks")
    
    # 7. Merge small masks into large ones
    if merge_small:
        masks = merge_small_masks_into_large(masks, small_area_thresh, overlap_thresh)
        print(f"[mask_postprocess] After small-into-large merge: {len(masks)} masks")
    
    # 8. NMS
    if nms:
        masks = mask_nms(masks, nms_thresh)
        print(f"[mask_postprocess] After NMS: {len(masks)} masks")
    
    # 9. Limit count
    if max_masks and len(masks) > max_masks:
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)[:max_masks]
        print(f"[mask_postprocess] After limiting count: {len(masks)} masks")
    
    return masks 