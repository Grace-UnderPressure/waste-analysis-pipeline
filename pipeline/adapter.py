#!/usr/bin/env python3
"""
Pipeline Adapter
Minimal pipeline wrapper: multi-scale segmentation → crop generation → CLIP classification → prompt building → Gemini batch inference.

Design goals:
- Single entrypoint: process_image(image_path, scan_id) → Dict
- Depend only on existing modules; prefer Gemini structured output (parsed) when available
- Default CPU execution; paths and thresholds are read from config.yaml
"""

import os
import sys
import json
import time
import yaml
import cv2
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Modules in the same directory
from segmenter import Segmenter
from advanced_segmentation_methods import AdvancedSegmenter
from crop_generator import crop_objects
from clip_matcher import CLIPMatcher
from clip_results_analyzer import create_clip_results_analyzer
from gemini_prompt_builder import create_prompt_builder
from gemini_inferencer import create_gemini_inferencer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineAdapter:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "config.yaml")
        self.cfg = self._load_config(self.config_path)
        self.device = self.cfg.get("device", "cpu")

        # Initialize components
        self._init_segmenters()
        self._init_clip()
        self._init_prompt_and_gemini()

        # Caches / utilities
        self.clip_analyzer = create_clip_results_analyzer(
            cache_dir=self.cfg.get("clip_results_cache_dir", "clip_results_cache"),
            enable_smart_cache=self.cfg.get("enable_smart_cache", True)
        )



    def _load_config(self, path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # Device fallback
        dev = cfg.get("device", "cpu")
        if dev == "auto":
            try:
                import torch  # noqa
                cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                cfg["device"] = "cpu"
        return cfg

    def _init_segmenters(self):
        seg_cfg = self.cfg.get("segmenter", {})
        self.baseline_segmenter = Segmenter(
            model_type=seg_cfg.get("model_type", "fastsam"),
            model_path=seg_cfg.get("model_path", "models/FastSAM-s.pt"),
            device=self.device,
            min_mask_area=self.cfg.get("min_mask_area", 3600),
            input_resize=1024,
            max_objects=seg_cfg.get("fastsam", {}).get("max_det", 50)
        )
        self.advanced_segmenter = AdvancedSegmenter(self.baseline_segmenter, self.device)

    def _init_clip(self):
        rec_cfg = self.cfg.get("recognizer", {})
        # Validate and resolve label config path
        label_config_path = rec_cfg.get("label_config_path", "waste_labels_template.json")
        if not os.path.isabs(label_config_path):
            label_config_path = os.path.join(os.path.dirname(__file__), label_config_path)
        if not os.path.exists(label_config_path):
            raise FileNotFoundError(f"Label config not found: {label_config_path}")
        self.clip_matcher = CLIPMatcher(
            label_config_path=label_config_path,
            device=self.device,
            clip_threshold=self.cfg.get("clip_threshold", 0.25),
            cache_embeddings=rec_cfg.get("cache_embeddings", True),
            embedding_cache_path=rec_cfg.get("embedding_cache_path", "clip_embeddings_cache.pkl")
        )

    def _init_prompt_and_gemini(self):
        rec_cfg = self.cfg.get("recognizer", {})
        label_config_path = rec_cfg.get("label_config_path", "waste_labels_template.json")
        if not os.path.isabs(label_config_path):
            label_config_path = os.path.join(os.path.dirname(__file__), label_config_path)
        
        # Get prompt saving config
        gcfg = self.cfg.get("gemini", {})
        save_prompts = gcfg.get("save_prompts", False)
        prompt_save_dir = gcfg.get("prompt_save_dir", "prompts")
        
        self.prompt_builder = create_prompt_builder(
            label_config_path, 
            save_prompts=save_prompts,
            prompt_save_dir=prompt_save_dir
        )

        api_key = gcfg.get("api_key") or os.getenv("GEMINI_API_KEY")
        self.gemini = create_gemini_inferencer(
            api_key=api_key,
            model_name=gcfg.get("model_name", "gemini-1.5-flash"),
            cache_responses=gcfg.get("cache_responses", True),
            cache_dir=gcfg.get("cache_dir", "gemini_cache"),
            max_retries=gcfg.get("max_retries", 3),
            retry_delay=gcfg.get("retry_delay", 1.0),
            enable_structured_output=gcfg.get("enable_structured_output", True),
            auto_schema_selection=True
        )
        self._gemini_include_label_context = gcfg.get("include_label_context", True)
        self._gemini_uncertainty_focus = gcfg.get("uncertainty_focus", True)
        self._gemini_batch_delay = gcfg.get("batch_delay", 0.5)

    # RAG helper methods removed (RAG disabled)

    def _visualize_crops_on_original(self, image_rgb, crops: List[Dict], scan_id: int, output_dir: str = None) -> str:
        """Draw bounding boxes on original image to show crop regions."""
        try:
            # Create a copy for visualization
            vis_image = image_rgb.copy()
            h, w = vis_image.shape[:2]
            
            # Unified color (single color for all boxes)
            box_color = (0, 255, 0)  # Green
            
            # Calculate font scale based on image size (larger, more readable text)
            font_scale = max(0.8, min(1.2, w / 800))  # Increased scale for better readability
            thickness = max(3, int(w / 400))  # Thicker box lines for better visibility
            text_thickness = max(2, int(w / 600))  # Thicker text for better readability
            
            # Draw bounding boxes for each crop
            for idx, crop in enumerate(crops):
                bbox = crop.get("bbox", [])
                if len(bbox) == 4:
                    x0, y0, x1, y1 = bbox
                    
                    # Draw rectangle
                    cv2.rectangle(vis_image, (x0, y0), (x1, y1), box_color, thickness)
                    
                    # Add crop number label (1-indexed for user display)
                    crop_number = idx + 1  # Display as Crop 1, 2, 3... to match Gemini output
                    label = f"Crop {crop_number}"
                    
                    # Position text in the top-left corner of the bounding box with better padding
                    text_x = x0 + 8  # More padding from box edge
                    text_y = y0 + 30  # Position inside the box with more space
                    
                    # Draw text with larger background for better visibility
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                    # Larger background rectangle with padding
                    bg_padding = 4
                    cv2.rectangle(vis_image, 
                                (text_x - bg_padding, text_y - text_h - bg_padding), 
                                (text_x + text_w + bg_padding, text_y + bg_padding), 
                                (255, 255, 255), -1)
                    
                    # Draw text in black for better contrast
                    cv2.putText(vis_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness)
            
            # Save visualization
            if output_dir:
                vis_dir = output_dir
            else:
                vis_dir = os.path.join(os.path.dirname(__file__), "outputs")
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, f"scan_{scan_id}_crops_visualization.jpg")
            
            # Convert RGB to BGR for saving
            vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_path, vis_bgr)
            
            logger.info(f"Saved crop visualization to {vis_path}")
            return vis_path
        except Exception as e:
            logger.warning(f"Failed to create crop visualization: {e}")
            return ""

    def _format_outputs_as_text(self, scan_id: int, compact_outputs: List[Dict]) -> str:
        lines = []
        lines.append(f"🔍 Waste Analysis (scan {scan_id})")
        lines.append(f"📦 Crops analyzed: {len(compact_outputs)}\n")
        
        for idx, item in enumerate(compact_outputs, 1):
            region_id = item.get('region_id', 'Unknown')
            crop_path = item.get('crop_path', 'Unknown')
            lines.append(f"— Crop {idx} (region {region_id}): {crop_path}")
            
            parsed = item.get("gemini_parsed")
            if isinstance(parsed, dict):
                # Single-crop schema packaged in a batch wrapper (total_crops==1)
                analyses = parsed.get("crop_analyses") or []
                if analyses:
                    a = analyses[0]
                    po = a.get("primary_object_analysis", {})
                    se = a.get("secondary_elements", {})
                    oa = a.get("overall_assessment", {})
                    
                    # Primary object analysis
                    lines.append(f"  • 🧱 Material: {po.get('material_composition', 'Unknown')}")
                    lines.append(f"  • 🧩 Shape/Size: {po.get('shape', '—')} / {po.get('size', '—')}")
                    lines.append(f"  • 🎨 Color: {po.get('color', '—')}")
                    lines.append(f"  • 🧪 State: {po.get('current_state', '—')}")
                    lines.append(f"  • 🎯 Purpose: {po.get('likely_function_or_original_purpose', '—')}")
                    lines.append(f"  • 👁 Visibility: {po.get('completeness_and_visibility_in_crop', '—')}")
                    
                    # Secondary elements (if present)
                    other_objects = se.get("other_visible_objects_or_portions", [])
                    if other_objects:
                        lines.append(f"  • 🔍 Other objects: {', '.join(other_objects)}")
                        relationship = se.get("relationship_to_primary_object")
                        if relationship:
                            lines.append(f"  • 🔗 Relationship: {relationship}")
                    
                    # Overall assessment
                    lines.append(f"  • 🗂 Category: {oa.get('primary_object_category_determination', 'Unknown')}")
                    lines.append(f"  • ♻️ Recyclability: {oa.get('recyclability_assessment_if_possible', '—')}")
                    
                    # Confidence with grade explanation
                    confidence = oa.get('confidence_level_in_assessment', '—')
                    if confidence in ['A', 'B', 'C', 'D']:
                        grade_explanations = {
                            'A': 'High: Clear features, high confidence',
                            'B': 'Medium: Generally clear features, moderate confidence',
                            'C': 'Low: Unclear features, low confidence', 
                            'D': 'Unreliable: Very poor quality/unclear, very low confidence'
                        }
                        lines.append(f"  • 📈 Confidence: {confidence} ({grade_explanations.get(confidence, '')})")
                    else:
                        lines.append(f"  • 📈 Confidence: {confidence}")
                    
                    # Uncertainties (if present)
                    uncertainties = oa.get("ambiguities_or_uncertainties_observed", [])
                    if uncertainties:
                        lines.append(f"  • ❓ Uncertainties: {', '.join(uncertainties)}")
                    
                    lines.append("")  # Empty line between crops
                else:
                    lines.append("  • (No structured fields available)\n")
            else:
                txt = item.get("gemini_text")
                if txt:
                    # Try to parse text as JSON and output structured fields
                    try:
                        import json as _json
                        obj = _json.loads(txt)
                        analyses = (obj or {}).get("crop_analyses") or []
                        if analyses:
                            a = analyses[0]
                            po = a.get("primary_object_analysis", {})
                            se = a.get("secondary_elements", {})
                            oa = a.get("overall_assessment", {})
                            
                            # Primary object analysis
                            lines.append(f"  • 🧱 Material: {po.get('material_composition', 'Unknown')}")
                            lines.append(f"  • 🧩 Shape/Size: {po.get('shape', '—')} / {po.get('size', '—')}")
                            lines.append(f"  • 🎨 Color: {po.get('color', '—')}")
                            lines.append(f"  • 🧪 State: {po.get('current_state', '—')}")
                            lines.append(f"  • 🎯 Purpose: {po.get('likely_function_or_original_purpose', '—')}")
                            lines.append(f"  • 👁 Visibility: {po.get('completeness_and_visibility_in_crop', '—')}")
                            
                            # Secondary elements (if present)
                            other_objects = se.get("other_visible_objects_or_portions", [])
                            if other_objects:
                                lines.append(f"  • 🔍 Other objects: {', '.join(other_objects)}")
                                relationship = se.get("relationship_to_primary_object")
                                if relationship:
                                    lines.append(f"  • 🔗 Relationship: {relationship}")
                            
                            # Overall assessment
                            lines.append(f"  • 🗂 Category: {oa.get('primary_object_category_determination', 'Unknown')}")
                            lines.append(f"  • ♻️ Recyclability: {oa.get('recyclability_assessment_if_possible', '—')}")
                            
                            # Confidence with grade explanation
                            confidence = oa.get('confidence_level_in_assessment', '—')
                            if confidence in ['A', 'B', 'C', 'D']:
                                grade_explanations = {
                                    'A': 'High: Clear features, high confidence',
                                    'B': 'Medium: Generally clear features, moderate confidence',
                                    'C': 'Low: Unclear features, low confidence', 
                                    'D': 'Unreliable: Very poor quality/unclear, very low confidence'
                                }
                                lines.append(f"  • 📈 Confidence: {confidence} ({grade_explanations.get(confidence, '')})")
                            else:
                                lines.append(f"  • 📈 Confidence: {confidence}")
                            
                            # Uncertainties (if present)
                            uncertainties = oa.get("ambiguities_or_uncertainties_observed", [])
                            if uncertainties:
                                lines.append(f"  • ❓ Uncertainties: {', '.join(uncertainties)}")
                            
                            lines.append("")  # Empty line between crops
                        else:
                            # Fallback to readable preview
                            preview = txt[:400].replace("\n", " ") + ("…" if len(txt) > 400 else "")
                            lines.append(f"  • 📝 Summary: {preview}\n")
                    except Exception:
                        preview = txt[:400].replace("\n", " ") + ("…" if len(txt) > 400 else "")
                        lines.append(f"  • 📝 Summary: {preview}\n")
                else:
                    # If no text, check raw preview captured at inference time
                    raw_preview = (item.get("gemini_parsed") or {})
                    raw_preview = item.get("raw_preview") if isinstance(item, dict) else None
                    if raw_preview:
                        lines.append(f"  • 📝 Summary: {raw_preview}\n")
                    else:
                        lines.append("  • (No Gemini result)\n")

            # RAG guidance block removed (RAG disabled)
        return "\n".join(lines)

    def _infer_requests_concurrent(self, requests: List[Dict], max_workers: int = 3) -> List[Dict]:
        """Small-concurrency Gemini calls (per-crop), returns results in original order."""
        results: List[Optional[Dict]] = [None] * len(requests)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {}
            for idx, req in enumerate(requests):
                future = ex.submit(self.gemini.infer_single, req.get("prompt", ""), req.get("image_path"))
                future_to_idx[future] = idx
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {"success": False, "error": str(e)}
        return results  # type: ignore

    def process_image(self, image_path: str, scan_id: int, output_dir: str = None) -> Dict:
        try:
            # Prepare output directory (save crops under per-scan subfolder)
            crop_dir = self.cfg.get("crop_dir", "crops")
            os.makedirs(crop_dir, exist_ok=True)
            per_scan_crop_dir = os.path.join(crop_dir, f"scan_{scan_id}")
            os.makedirs(per_scan_crop_dir, exist_ok=True)

            # Read image and convert to RGB
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 1) Multi-scale segmentation
            logger.info("[1/5] Segmenting (multi-scale)…")
            scales = self.cfg.get("segmenter", {}).get("multi_scale_scales", [0.8, 1.0, 1.2])
            use_postprocess = self.cfg.get("segmenter", {}).get("use_postprocess", False)
            masks = self.advanced_segmenter.multi_scale_segmentation(image_rgb, scales=scales, use_postprocess=use_postprocess)
            if not masks:
                return {"success": False, "scan_id": scan_id, "error": "no_masks"}

            # 2) Generate high-resolution crops
            logger.info("[2/5] Generating crops…")
            crops = crop_objects(
                self.baseline_segmenter._resize_image(image_rgb),
                masks,
                per_scan_crop_dir,
                image_id=f"scan_{scan_id}",
                original_image=image_rgb,
                original_shape=image_rgb.shape[:2]
            )
            if not crops:
                return {"success": False, "scan_id": scan_id, "error": "no_crops"}

            # Optional: Create crop visualization
            try:
                vis_path = self._visualize_crops_on_original(image_rgb, crops, scan_id, output_dir)
                if vis_path:
                    logger.info(f"Crop regions visualized: {vis_path}")
            except Exception as e:
                logger.warning(f"Failed to create crop visualization: {e}")

            # 3) CLIP classification
            logger.info("[3/5] CLIP classifying…")
            batch_size = self.cfg.get("recognizer", {}).get("batch_size", 32)
            try:
                batch_size = max(1, int(batch_size))
            except Exception:
                batch_size = 32
            clip_results = self.clip_matcher.classify_batch(crops, batch_size=batch_size)

            # Optional: save CLIP results for reproducibility / enhanced prompts (clip_results_cache/)
            try:
                out_cfg = self.cfg.get("output", {})
                if out_cfg.get("save_clip_results", False):
                    result = self.clip_analyzer.integrate_with_existing_pipeline(
                        image_path=image_path,
                        existing_clip_results=clip_results,
                        config=self.cfg,
                        force_save=True  # Force save
                    )
                    logger.info(f"CLIP results saved: {result.get('json_path', 'NO_PATH')}")
            except Exception as e:
                logger.warning(f"Failed to save CLIP results: {e}", exc_info=True)

            # 4) Prompt building (per crop)
            logger.info("[4/5] Building prompts for Gemini…")
            requests = []
            for i, (crop, pred) in enumerate(zip(crops, clip_results)):
                # Set scan_id and crop_id for prompt saving
                self.prompt_builder._current_scan_id = scan_id
                self.prompt_builder._current_crop_id = i
                
                # Use enhanced unified prompt (based on CLIP statistics + context)
                img_path_abs = crop.get("crop_path")
                if img_path_abs and not os.path.isabs(img_path_abs):
                    img_path_abs = os.path.abspath(img_path_abs)
                
                prompt = self.prompt_builder.build_unified_prompt(
                    [pred],
                    include_statistics=True,
                    include_label_context=self._gemini_include_label_context,
                    uncertainty_focus=self._gemini_uncertainty_focus,
                    original_image_path=image_path,  # 原图路径
                    crop_path=img_path_abs  # crop图片路径
                )
                requests.append({
                    "prompt": prompt, 
                    "image_path": img_path_abs,
                    "original_image_path": image_path  # 新增：原图路径
                })

            # 5) Gemini inference (supports small concurrency)
            logger.info("[5/5] Calling Gemini…")
            # Switch structured schema based on batch size (single/multi-crop)
            try:
                schema = self.gemini.select_schema_for_batch(len(requests))
                self.gemini.update_schema_for_inference(schema)
            except Exception:
                pass
            # Concurrency parameters and optional chunked-batch mode
            out_cfg = self.cfg.get("output", {})
            batch_chunk_size = int(out_cfg.get("batch_chunk_size", 0) or 0)
            chunk_workers = int(out_cfg.get("chunk_workers", 1) or 1)
            max_workers = int(out_cfg.get("concurrent_requests", 3) or 3)

            if batch_chunk_size and batch_chunk_size > 1:
                # Chunked-batch: slice requests into chunks of size k and call infer_batch per chunk
                chunks: List[List[Dict]] = [
                    requests[i:i + batch_chunk_size] for i in range(0, len(requests), batch_chunk_size)
                ]
                logger.info(f"Using chunked-batch mode: {len(chunks)} chunks, size={batch_chunk_size}, workers={chunk_workers}")
                if chunk_workers and chunk_workers > 1:
                    # Parallelize chunks lightly; preserve overall order by chunk index
                    results_by_chunk: Dict[int, List[Dict]] = {}
                    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
                    with ThreadPoolExecutor(max_workers=chunk_workers) as ex:
                        future_to_idx = {}
                        for ci, chunk in enumerate(chunks):
                            fut = ex.submit(self.gemini.infer_batch, chunk, self._gemini_batch_delay)
                            future_to_idx[fut] = ci
                        for fut in _as_completed(future_to_idx):
                            ci = future_to_idx[fut]
                            try:
                                results_by_chunk[ci] = fut.result() or []
                            except Exception as e:
                                logger.warning(f"Chunk {ci} failed: {e}")
                                results_by_chunk[ci] = [{"success": False, "error": str(e)}] * len(chunks[ci])
                    # Flatten in original chunk order
                    gemini_responses = []
                    for ci in range(len(chunks)):
                        gemini_responses.extend(results_by_chunk.get(ci, []))
                else:
                    # Sequential chunks
                    gemini_responses = []
                    for ci, chunk in enumerate(chunks):
                        try:
                            part = self.gemini.infer_batch(chunk, batch_delay=self._gemini_batch_delay) or []
                        except Exception as e:
                            logger.warning(f"Chunk {ci} failed: {e}")
                            part = [{"success": False, "error": str(e)}] * len(chunk)
                        gemini_responses.extend(part)
            else:
                # Per-crop path: small concurrency using infer_single, or fall back to one infer_batch
                if max_workers and max_workers > 1:
                    gemini_responses = self._infer_requests_concurrent(requests, max_workers=max_workers)
                else:
                    gemini_responses = self.gemini.infer_batch(requests, batch_delay=self._gemini_batch_delay)
            if len(gemini_responses) != len(crops):
                logger.warning(f"Gemini responses count ({len(gemini_responses)}) != crops ({len(crops)}), padding failures")
                if len(gemini_responses) < len(crops):
                    pad = len(crops) - len(gemini_responses)
                    gemini_responses.extend([{"success": False, "error": "missing_response"}] * pad)

            # Extract structured data (if available)
            analyses: List[Dict] = []
            for crop, pred, resp in zip(crops, clip_results, gemini_responses):
                parsed = resp.get("parsed") if isinstance(resp, dict) else None
                analyses.append({
                    "region_id": crop.get("region_id"),
                    "crop_path": crop.get("crop_path"),
                    "clip_prediction": {
                        "main_label": pred.get("main_label", "unknown"),
                        "sub_label": pred.get("sub_label", "unknown"),
                        "confidence": pred.get("confidence", 0.0),
                    },
                    "gemini": {
                        "success": resp.get("success", False) if isinstance(resp, dict) else False,
                        "error": resp.get("error") if isinstance(resp, dict) else None,
                        "text": resp.get("response") if isinstance(resp, dict) else None,
                        "parsed": parsed.model_dump() if hasattr(parsed, "model_dump") else parsed,
                        "parse_status": ((resp.get("metadata") or {}).get("parse_status") if isinstance(resp, dict) else None),
                        "raw_preview": ((resp.get("metadata") or {}).get("raw_preview") if isinstance(resp, dict) else None)
                    }
                })

            # RAG removed

            # Output control: by default only return Gemini analysis, hide intermediate results
            out_cfg = self.cfg.get("output", {})
            include_intermediate = bool(out_cfg.get("include_intermediate", False))

            if include_intermediate:
                result = {
                    "success": True,
                    "scan_id": scan_id,
                    "num_masks": len(masks),
                    "num_crops": len(crops),
                    "crops": crops,
                    "predictions": clip_results,
                    "analyses": analyses
                }
                return result
            else:
                # Compact output: keep Gemini key info only (RAG removed)
                compact_outputs = []
                for a in analyses:
                    base = {
                        "region_id": a.get("region_id"),
                        "crop_path": a.get("crop_path"),
                        "gemini_text": (a.get("gemini") or {}).get("text"),
                        "gemini_parsed": (a.get("gemini") or {}).get("parsed")
                    }
                    compact_outputs.append(base)
                result = {
                    "success": True,
                    "scan_id": scan_id,
                    "num_crops": len(compact_outputs),
                    "outputs": compact_outputs
                }

                # Save to files (outputs/scan_{scan_id}_result.json & .txt)
                try:
                    if output_dir:
                        outputs_dir = output_dir
                    else:
                        outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
                    os.makedirs(outputs_dir, exist_ok=True)
                    out_path = os.path.join(outputs_dir, f"scan_{scan_id}_result.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved result to {out_path}")

                    # RAG save removed

                    # Pretty text
                    pretty = self._format_outputs_as_text(scan_id, compact_outputs)
                    out_txt = os.path.join(outputs_dir, f"scan_{scan_id}_result.txt")
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write(pretty)
                    logger.info(f"Saved pretty text to {out_txt}")

                    # Save Gemini metrics
                    gemini_metrics = self.gemini.get_stats()
                    metrics_path = os.path.join(outputs_dir, f"scan_{scan_id}_gemini_metrics.json")
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        json.dump(gemini_metrics, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved Gemini metrics to {metrics_path}")

                    # Attach file paths for downstream consumers on the same VM
                    try:
                        result.setdefault("files", {})
                        result["files"].update({
                            "result_json": out_path,
                            "result_txt": out_txt,
                            "gemini_metrics": metrics_path,
                            "crop_visualization": vis_path if isinstance(locals().get("vis_path"), str) else "",
                            "crops_dir": per_scan_crop_dir,
                            "original_image": image_path
                        })
                    except Exception:
                        pass
                except Exception:
                    logger.warning("Failed to save result file", exc_info=True)

                return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {"success": False, "scan_id": scan_id, "error": str(e)}


def create_pipeline_adapter(config_path: Optional[str] = None) -> PipelineAdapter:
    return PipelineAdapter(config_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Path to image")
    parser.add_argument("--scan-id", type=int, default=int(time.time()))
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    adapter = create_pipeline_adapter(args.config)
    out = adapter.process_image(args.image, args.scan_id)
    # CLI printing removed; use logs or consume the returned dict in your own entrypoint

