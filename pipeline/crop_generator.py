import os
import cv2

def crop_objects(image, masks, output_dir, image_id="img", min_mask_area=200, min_contrast=10, original_image=None, original_shape=None):
    """
    Crop object regions from an image.
    
    Args:
        image: current image (possibly resized)
        masks: list of segmentation masks
        output_dir: output directory
        image_id: image identifier
        min_mask_area: minimum mask area
        min_contrast: minimum contrast threshold
        original_image: original image (optional, for high-res crops)
        original_shape: original image shape (h, w) (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    crops = []
    
    # Determine crop source image and coordinate scaling
    if original_image is not None and original_shape is not None:
        # Use original image to keep high resolution
        crop_source = original_image
        h_orig, w_orig = original_shape
        h_curr, w_curr = image.shape[:2]
        
        # Compute scaling factors
        scale_h = h_orig / h_curr
        scale_w = w_orig / w_curr
        
        print(f"🔍 Using original image for crops, resolution: {w_orig}x{h_orig} -> {w_curr}x{h_curr}")
        print(f"   Scale factors: {scale_w:.2f}x{scale_h:.2f}")
    else:
        # Use current image for crops
        crop_source = image
        scale_h = scale_w = 1.0
        print(f"🔍 Using current image for crops, resolution: {image.shape[1]}x{image.shape[0]}")
    
    for idx, m in enumerate(masks):
        x0, y0, x1, y1 = m["bbox"]
        area = m.get("area", (x1-x0)*(y1-y0))
        
        if area < min_mask_area:
            continue
            
        # Scale coordinates to original image size
        x0_orig = max(0, int(x0 * scale_w))
        y0_orig = max(0, int(y0 * scale_h))
        x1_orig = min(crop_source.shape[1], int(x1 * scale_w))
        y1_orig = min(crop_source.shape[0], int(y1 * scale_h))
        
        # Ensure valid region
        if x1_orig <= x0_orig or y1_orig <= y0_orig:
            continue
            
        # Crop from source
        crop = crop_source[y0_orig:y1_orig, x0_orig:x1_orig]
        
        # Filter blank/low-contrast crops
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
            
        # Contrast check
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop
            
        if gray.std() < min_contrast:
            continue
            
        # Save crop
        crop_path = os.path.join(output_dir, f"{image_id}_crop{idx:03d}.jpg")
        
        # Ensure saving in RGB
        if len(crop.shape) == 3 and crop.shape[2] == 3:
            # Already RGB
            cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        else:
            # Convert if needed
            cv2.imwrite(crop_path, crop)
            
        crops.append({
            "region_id": m.get("region_id", idx),
            "crop_path": crop_path,
            "bbox": [x0_orig, y0_orig, x1_orig, y1_orig],  # original-image coordinates
            "area": area,
            "crop_size": f"{crop.shape[1]}x{crop.shape[0]}"  # crop size (w x h)
        })
        
    print(f"📸 Generated {len(crops)} high-resolution crops")
    return crops
