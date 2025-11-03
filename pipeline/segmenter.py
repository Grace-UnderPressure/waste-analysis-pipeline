import numpy as np
import torch
from ultralytics import YOLO
import cv2
import os
from mask_postprocess import mask_postprocess

class Segmenter:
    def __init__(self, model_type: str, model_path: str, device: str = "cuda", min_mask_area: int = 2000, input_resize: int = 1024, max_objects: int = 20):
        self.model_type = model_type.lower()
        self.device = device
        self.min_mask_area = min_mask_area
        self.input_resize = input_resize
        self.max_objects = max_objects
        
        if self.model_type == "fastsam":
            # FastSAM optimized config - based on official docs
            self.model = YOLO(model_path)
            # FastSAM-specific parameters
            self.fastsam_config = {
                'imgsz': 640,           # Inference size: balance speed and accuracy
                'conf': 0.5,            # Higher confidence to filter low-quality detections
                'iou': 0.5,             # Lower NMS IoU for stricter overlap removal
                'retina_masks': True,   # High-precision masks
                'verbose': False,       # Quiet output
                'max_det': 30,          # Fewer maximum detections
                'agnostic_nms': True,   # Class-agnostic NMS
                'classes': None,        # Detect all classes
            }
            print(f"✅ FastSAM loaded from {model_path} with optimized config")
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. This build only supports 'fastsam'.")

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        max_side = max(h, w)
        if max_side > self.input_resize:
            scale = self.input_resize / max_side
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return image

    def _get_mask_prompts(self, mask):
        seg = mask["segmentation"]
        ys, xs = np.where(seg)
        if xs.size == 0 or ys.size == 0:
            return []
        # Center point
        cx, cy = int(xs.mean()), int(ys.mean())
        # Bounding box corners
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        prompts = [
            [cx, cy],
            [x0, y0],
            [x1, y0],
            [x0, y1],
            [x1, y1]
        ]
        return np.array(prompts)

    def _filter_fastsam_masks(self, masks, image_shape):
        """
        FastSAM mask filtering by area, bbox size/aspect, compactness, etc.
        """
        H, W = image_shape[:2]
        filtered_masks = []
        
        for mask in masks:
            seg = mask["segmentation"].astype(bool)
            area = seg.sum()
            
            # 1) Area filter (also drop overly large background)
            if area < self.min_mask_area or area > H * W * 0.8:
                continue
                
            # 2) Color stats within mask (guard for empty)
            ys, xs = np.where(seg)
            if xs.size == 0 or ys.size == 0:
                continue
                
            # 3) Bounding box filter (min size / aspect ratio)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            if (x1 - x0) < 20 or (y1 - y0) < 20:
                continue
            if max(x1-x0, y1-y0) / (min(x1-x0, y1-y0)+1e-6) > 10:
                continue
                
            # 4) Mask compactness (perimeter/area)
            from skimage import measure
            contours = measure.find_contours(seg.astype(float), 0.5)
            if contours:
                perimeter = sum(len(contour) for contour in contours)
                compactness = perimeter / (area + 1e-6)
                if compactness > 0.3:
                    continue
            
            filtered_masks.append(mask)
            
        return filtered_masks

    def segment(self, image: np.ndarray, min_mask_area: int = None, use_postprocess: bool = True) -> list:
        if min_mask_area is None:
            min_mask_area = self.min_mask_area
        image = self._resize_image(image)
        
        if self.model_type == "fastsam":
            # FastSAM optimized flow
            print(f"🧠 FastSAM segmentation start, config: {self.fastsam_config}")
            
            # Strategy 1: everything prompt
            results = self.model.predict(
                image, 
                **self.fastsam_config
            )
            
            masks = []
            region_id = 0
            
            for r in results:
                if not hasattr(r, "masks") or r.masks is None:
                    continue
                    
                for i, seg in enumerate(r.masks.data.cpu().numpy()):
                    area = np.sum(seg)
                    if area < min_mask_area:
                        continue
                        
                    ys, xs = np.where(seg)
                    if xs.size == 0 or ys.size == 0:
                        continue
                        
                    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                    if (x1 - x0) < 20 or (y1 - y0) < 20:
                        continue
                        
                    # Confidence if available
                    confidence = r.boxes.conf[i].cpu().numpy() if hasattr(r, 'boxes') and r.boxes is not None else 0.5
                    
                    masks.append({
                        "region_id": region_id,
                        "segmentation": seg.astype(np.uint8),
                        "bbox": [x0, y0, x1, y1],
                        "area": int(area),
                        "confidence": float(confidence)
                    })
                    region_id += 1

            print(f"🧠 FastSAM raw masks: {len(masks)}")
            
            # FastSAM mask filtering
            masks = self._filter_fastsam_masks(masks, image.shape)
            print(f"🧠 FastSAM masks after filtering: {len(masks)}")
            
            # Mask postprocess
            if use_postprocess:
                masks = mask_postprocess(masks, min_area=min_mask_area, max_masks=self.max_objects)
                print(f"🧠 FastSAM masks after postprocess: {len(masks)}")
            
            return masks
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. This build only supports 'fastsam'.")


