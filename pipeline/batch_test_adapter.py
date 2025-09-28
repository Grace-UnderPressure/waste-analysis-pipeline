#!/usr/bin/env python3
"""
Batch test runner for PipelineAdapter.
Runs adapter over all images in a directory, collects timings and key metrics,
and writes CSV and JSON summaries in outputs/. Designed for quick, automated evaluation.
"""

import os
import sys
import time
import json
import csv
import argparse
import shutil
from typing import List, Dict

# Local imports
from adapter import create_pipeline_adapter


def is_image_file(name: str) -> bool:
    name_l = name.lower()
    return name_l.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def main():
    parser = argparse.ArgumentParser(description="Batch test PipelineAdapter over a folder of images")
    parser.add_argument("input", type=str, help="Input directory or single image file")
    parser.add_argument("--config", type=str, default=None, help="Optional config.yaml path")
    parser.add_argument("--scan-base", type=int, default=int(time.time()), help="Base scan id")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between images")
    parser.add_argument("--suffix", type=str, default="", help="Suffix tag for summary filenames")
    parser.add_argument("--outdir", type=str, default=None, help="Directory to store test outputs (default outputs_tests/run_<ts>)")
    parser.add_argument("--include-clip-cache", action="store_true", help="Copy related CLIP cache files per scan when possible")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    
    # Check if input is a single image file or directory
    if os.path.isfile(input_path) and is_image_file(os.path.basename(input_path)):
        # Single image file
        in_dir = os.path.dirname(input_path)
        all_files = [os.path.basename(input_path)]
        print(f"Single image mode: {os.path.basename(input_path)}")
    elif os.path.isdir(input_path):
        # Directory of images
        in_dir = input_path
        all_files = sorted([f for f in os.listdir(in_dir) if is_image_file(f)])
        if not all_files:
            print(f"No image files found in directory: {in_dir}")
            sys.exit(1)
        print(f"Directory mode: {len(all_files)} images found")
    else:
        print(f"Input not found or not a valid image/directory: {input_path}")
        sys.exit(1)

    # Create adapter
    adapter = create_pipeline_adapter(args.config)

    # Apply limit if specified
    if args.limit and args.limit > 0:
        all_files = all_files[: args.limit]

    # Base adapter outputs (where adapter writes per-scan files)
    # Note: adapter will now write to test_out_dir instead of hardcoded outputs/
    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # Dedicated test run directory for consolidated artifacts
    run_tag = args.suffix.strip() or time.strftime("%Y%m%d_%H%M%S")
    default_test_dir = os.path.join(os.path.dirname(__file__), "outputs_tests", f"run_{run_tag}")
    test_out_dir = os.path.abspath(args.outdir) if args.outdir else default_test_dir
    os.makedirs(test_out_dir, exist_ok=True)

    summary_rows: List[Dict] = []
    results: Dict = {
        "input_dir": in_dir,
        "num_images": len(all_files),
        "run_started": int(time.time()),
        "items": []
    }

    # Determine mode from config
    out_cfg = getattr(adapter, 'cfg', {}).get('output', {}) if hasattr(adapter, 'cfg') else {}
    concurrent_requests = int(out_cfg.get('concurrent_requests', 1) or 1)
    batch_chunk_size = int(out_cfg.get('batch_chunk_size', 0) or 0)
    chunk_workers = int(out_cfg.get('chunk_workers', 1) or 1)
    if batch_chunk_size > 1:
        mode = f"chunked(size={batch_chunk_size}, workers={chunk_workers})"
    else:
        mode = f"per-crop(concurrency={concurrent_requests})"

    print(f"Mode: {mode}; Images: {len(all_files)}; Dir: {in_dir}; TestOut: {test_out_dir}")

    for idx, name in enumerate(all_files, start=1):
        img_path = os.path.join(in_dir, name)
        scan_id = args.scan_base + idx
        t0 = time.time()
        try:
            result = adapter.process_image(img_path, scan_id, output_dir=test_out_dir)
            elapsed = time.time() - t0
            success = bool(result.get("success", False))
            num_crops = int(result.get("num_crops", 0))
            json_path = os.path.join(test_out_dir, f"scan_{scan_id}_result.json")
            txt_path = os.path.join(test_out_dir, f"scan_{scan_id}_result.txt")
            vis_path = os.path.join(test_out_dir, f"scan_{scan_id}_crops_visualization.jpg")
            metrics_path = os.path.join(test_out_dir, f"scan_{scan_id}_gemini_metrics.json")

            # Per-scan subfolder under test output
            per_scan_dir = os.path.join(test_out_dir, f"scan_{scan_id}")
            os.makedirs(per_scan_dir, exist_ok=True)

            # Load per-scan Gemini metrics if available
            gm = {}
            try:
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r", encoding="utf-8") as mf:
                        gm = json.load(mf) or {}
            except Exception:
                gm = {}

            # Flatten a few key Gemini metrics for summary
            gm_flat = {
                "gm_total_input_tokens": gm.get("total_input_tokens", 0),
                "gm_total_output_tokens": gm.get("total_output_tokens", 0),
                "gm_total_tokens": gm.get("total_tokens", 0),
                "gm_avg_response_time_ms": gm.get("avg_response_time_ms", 0),
                "gm_api_calls": gm.get("api_calls", 0),
                "gm_cache_hits": gm.get("cache_hits", 0),
                "gm_errors": gm.get("errors", 0),
                "gm_structured_parse_success": gm.get("structured_parse_success", 0),
                "gm_structured_parse_failures": gm.get("structured_parse_failures", 0),
            }

            row = {
                "idx": idx,
                "scan_id": scan_id,
                "image": name,
                "success": success,
                "num_crops": num_crops,
                "elapsed_sec": round(elapsed, 3),
                "mode": mode,
                "json_exists": os.path.exists(json_path),
                "txt_exists": os.path.exists(txt_path),
                **gm_flat,
            }
            summary_rows.append(row)

            results["items"].append({
                "scan_id": scan_id,
                "image": name,
                "success": success,
                "num_crops": num_crops,
                "elapsed_sec": elapsed,
                "mode": mode,
                "gemini_metrics": gm_flat,
            })
            # Move per-scan artifacts (results, txt, vis, metrics) into test folder
            try:
                for p in (json_path, txt_path, vis_path, metrics_path):
                    if os.path.exists(p):
                        dest_p = os.path.join(per_scan_dir, os.path.basename(p))
                        try:
                            if os.path.exists(dest_p):
                                os.remove(dest_p)
                        except Exception:
                            pass
                        try:
                            shutil.move(p, dest_p)
                        except Exception:
                            # Fallback to copy if move fails (e.g., cross-device)
                            shutil.copy2(p, dest_p)
            except Exception:
                pass

            # Copy crops belonging to this scan
            try:
                crop_dir = getattr(adapter, 'cfg', {}).get('crop_dir', 'crops') if hasattr(adapter, 'cfg') else 'crops'
                crop_dir_abs = crop_dir if os.path.isabs(crop_dir) else os.path.join(os.path.dirname(__file__), crop_dir)
                per_scan_crop_dir = os.path.join(crop_dir_abs, f"scan_{scan_id}")
                dest_crop_dir = os.path.join(per_scan_dir, 'crops')
                if os.path.isdir(per_scan_crop_dir):
                    # New layout: copy whole per-scan folder
                    if os.path.exists(dest_crop_dir):
                        shutil.rmtree(dest_crop_dir, ignore_errors=True)
                    shutil.copytree(per_scan_crop_dir, dest_crop_dir)
                elif os.path.isdir(crop_dir_abs):
                    # Legacy flat layout: copy by filename prefix
                    os.makedirs(dest_crop_dir, exist_ok=True)
                    prefix = f"scan_{scan_id}_"
                    for fn in os.listdir(crop_dir_abs):
                        if fn.startswith(prefix):
                            try:
                                shutil.copy2(os.path.join(crop_dir_abs, fn), os.path.join(dest_crop_dir, fn))
                            except Exception:
                                pass
            except Exception:
                pass

            # Optionally copy related CLIP cache files (best-effort by scan_id substring)
            if args.include_clip_cache:
                try:
                    cache_dir = getattr(adapter, 'cfg', {}).get('clip_results_cache_dir', 'clip_results_cache') if hasattr(adapter, 'cfg') else 'clip_results_cache'
                    cache_dir_abs = cache_dir if os.path.isabs(cache_dir) else os.path.join(os.path.dirname(__file__), cache_dir)
                    if os.path.isdir(cache_dir_abs):
                        dest_cache_dir = os.path.join(per_scan_dir, 'clip_results_cache')
                        os.makedirs(dest_cache_dir, exist_ok=True)
                        sid = str(scan_id)
                        for fn in os.listdir(cache_dir_abs):
                            if sid in fn:
                                try:
                                    shutil.copy2(os.path.join(cache_dir_abs, fn), os.path.join(dest_cache_dir, fn))
                                except Exception:
                                    pass
                except Exception:
                    pass
            print(f"[{idx}/{len(all_files)}] {name} -> success={success} crops={num_crops} time={row['elapsed_sec']}s")
        except Exception as e:
            elapsed = time.time() - t0
            row = {
                "idx": idx,
                "scan_id": scan_id,
                "image": name,
                "success": False,
                "num_crops": 0,
                "elapsed_sec": round(elapsed, 3),
                "mode": mode,
                "error": str(e),
            }
            summary_rows.append(row)
            results["items"].append(row)
            print(f"[{idx}/{len(all_files)}] {name} -> FAILED in {row['elapsed_sec']}s: {e}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Write CSV summary
    suffix = ("_" + args.suffix.strip()) if args.suffix else ""
    csv_path = os.path.join(test_out_dir, f"batch_summary{suffix}.csv")
    fieldnames = [
        "idx", "scan_id", "image", "success", "num_crops", "elapsed_sec", "mode", "json_exists", "txt_exists",
        "gm_total_input_tokens", "gm_total_output_tokens", "gm_total_tokens", "gm_avg_response_time_ms",
        "gm_api_calls", "gm_cache_hits", "gm_errors", "gm_structured_parse_success", "gm_structured_parse_failures",
        "error"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            if "error" not in r:
                r["error"] = ""
            writer.writerow(r)
    print(f"Summary CSV: {csv_path}")

    # Write JSON summary
    results["run_finished"] = int(time.time())
    json_path = os.path.join(test_out_dir, f"batch_summary{suffix}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(results, jf, ensure_ascii=False, indent=2)
    print(f"Summary JSON: {json_path}")


if __name__ == "__main__":
    main()


