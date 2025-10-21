# Waste Analysis Pipeline — VM Developer Guide (upload_vm_bundle)

Production-ready bundle for VM deployment:
- **Multi-scale segmentation** → automated crop generation → CLIP classification → Gemini inference
- **Original image context** for Gemini analysis
- **Robust fallback mechanisms** with parse_status/raw_preview
- **60% reduction** in redundant crops through IoU-based mask fusion

## 1) Requirements
- Python 3.10+
- GPU optional (CUDA recommended)
- Google Gemini API key

Install:
```bash
python -m pip install -r requirements.txt
```

Environment:
```bash
export GEMINI_API_KEY="<YOUR_KEY>"
```

Notes:
- **Torch**: CPU or CUDA build (requirements specify baseline, adjust for your hardware)
- **SAM**: Optional (only needed if changing from default FastSAM)
- **Google SDK**: Use `google-genai` (new SDK). Legacy `google-generativeai` not required
- **API Key**: Required in config.yaml or environment variable

### On a fresh Ubuntu VM (recommended)
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv ~/wcr_env
source ~/wcr_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export GEMINI_API_KEY="<YOUR_KEY>"
```

## 2) Key Files
- `adapter.py`: **Main pipeline orchestrator** - processes single images end-to-end
- `batch_test_adapter.py`: **Batch testing utility** - processes multiple images with metrics
- `gemini_inferencer.py`: **Gemini API wrapper** - multimodal inference with original image context
- `gemini_prompt_builder.py`: **Strategy-based prompt builder** - statistical templates + image labeling
- `clip_matcher.py`: **CLIP classifier** - relaxed rules for fewer false unknowns
- `advanced_segmentation_methods.py`: **Multi-scale segmentation** - IoU-based fusion with containment removal
- `segmenter.py`: **FastSAM wrapper** - multi-scale object detection
- `waste_analysis_schema.py`: **Pydantic schemas** - structured output validation
- `config.yaml`: **Central configuration** - all pipeline parameters and VM optimization settings

## 2.1) Complete File Structure
```
upload_vm_bundle/pipeline/
├── README.md                           # This developer guide
├── requirements.txt                    # Python dependencies
├── config.yaml                         # Central configuration
│
├── Core Pipeline Files:
├── adapter.py                          # Main orchestrator
├── batch_test_adapter.py              # Batch testing utility
│
├── Segmentation Module:
├── segmenter.py                        # FastSAM wrapper
├── advanced_segmentation_methods.py   # Multi-scale + IoU-based fusion
├── mask_postprocess.py                # Mask refinement pipeline
├── crop_generator.py                  # High-resolution cropping
│
├── CLIP Module:
├── clip_matcher.py                    # CLIP classification engine
├── clip_results_analyzer.py          # Statistical analysis + prompts
├── generate_label_template.py        # Label template generator
├── waste_labels_template.json        # Waste classification taxonomy
│
├── Gemini Module:
├── gemini_inferencer.py              # API wrapper + multimodal
├── gemini_prompt_builder.py          # Strategy-based prompt generation
├── waste_analysis_schema.py          # Pydantic output schemas
│
├── Models & Data:
├── models/
│   └── FastSAM-s.pt                  # Segmentation model weights
│
├── Runtime Outputs:
├── outputs/                           # Main results directory
│   ├── scan_*_result.json            # Structured analysis
│   ├── scan_*_result.txt             # Human-readable results
│   ├── scan_*_crops_visualization.jpg # Segmentation visualization
│   └── scan_*_gemini_metrics.json    # API usage stats
├── crops/scan_*/                      # Per-scan crop images
├── clip_results_cache/                # CLIP classification cache
├── gemini_cache/                      # Gemini response cache
├── prompts/                           # Saved prompts (if enabled)
│
├── Test Data & Results:
├── test_super_u_images/               # Sample test images
├── outputs_tests/                     # Batch test results
│   └── run_*/                         # Per-run organized outputs
└── WT_experiment_dataset_picked/      # Additional test dataset
```

## 3) Models and Data
- FastSAM: put weights at `models/FastSAM-s.pt` (or set path in `config.yaml`).
- Labels: `waste_labels_template.json` (use `generate_label_template.py` to create).

## 4) Quickstart
Single image:
```bash
python adapter.py /abs/path/to/image.jpg --scan-id 12345 --config ./config.yaml
```
More adapter examples:
```bash
# Minimal (uses default config.yaml next to the script)
python adapter.py /data/images/sample.jpg --scan-id 1700000000

# Explicit config
python adapter.py /data/images/sample.jpg --scan-id 1700000001 --config /app/pipeline/config.yaml
```
Adapter input/output:
- Input: `image_path`, `scan_id`, optional `output_dir` (matches `app.py` integration)
- Outputs in `outputs/` (or custom `output_dir`):
  - `scan_<id>_result.json` (structured analysis results)
  - `scan_<id>_result.txt` (human-readable; includes parse status and raw preview on failures)
  - `scan_<id>_crops_visualization.jpg` (segmentation visualization)
  - `scan_<id>_gemini_metrics.json` (API usage stats)
  - crops saved under `crops/scan_<id>/` (independent of output_dir)

Batch folder:
```bash
python batch_test_adapter.py /abs/path/to/images --config ./config.yaml --limit 0
```
More batch examples:
```bash
# Basic run with default outputs_tests/run_<ts>
python batch_test_adapter.py /data/images --config ./config.yaml

# Limit to first 5 images, add a suffix tag, and custom output dir
python batch_test_adapter.py /data/images --config ./config.yaml --limit 5 --suffix smoke --outdir /tmp/wcr_tests

# Copy related CLIP cache files per scan (if available)
python batch_test_adapter.py /data/images --config ./config.yaml --include-clip-cache
```
Artifacts are copied into `outputs_tests/run_<ts>/scan_<id>/`.

### Programmatic usage (call adapter from other code)
```python
from adapter import create_pipeline_adapter

# Initialize once (optionally pass explicit config path)
adapter = create_pipeline_adapter("./config.yaml")

# Call for a single image
out = adapter.process_image("/abs/path/to/image.jpg", scan_id=12345, output_dir="/custom/path")

# Use returned file paths (for dashboards/APIs)
files = out.get("files", {})
print(files.get("result_json"))
print(files.get("result_txt"))
print(files.get("crop_visualization"))
print(files.get("crops_dir"))
```

Returned `out` fields (key parts):
- `success` (bool), `error` (None or str)
- `parse_status` in `adapter_result.metadata.parse_status` (sdk_parsed | manual_parsed | fallback | none)
- `files` dict with absolute paths: `result_json`, `result_txt`, `gemini_metrics`, `crop_visualization`, `crops_dir`, `original_image`

## 5) Config (config.yaml)
- `segmenter`: `model_type` (fastsam), `model_path`, `multi_scale_scales`, `use_postprocess`
- `recognizer`: `label_config_path`, `batch_size`, cache options
- `gemini`: `api_key` (or ENV), `model_name`, structured output flags, prompt saving options
- `output`: batch processing control (chunk_size, workers, concurrency)
- `clip_threshold`: Unknown classification threshold (default: 0.1)
- `device`: cpu | cuda | auto

Recommended VM deployment values (optimal v2):
- `segmenter.use_postprocess: false` (improved performance)
- `gemini.batch_delay: 1.2` (stable API timing)
- `output.concurrent_requests: 6` (per-crop concurrency used only when chunking is disabled)
- `output.batch_chunk_size: 4` (chunked-batch mode, recommended)
- `output.chunk_workers: 2` (parallel chunk processing)
- Time units: all pipeline timing fields are in seconds (s); Gemini average response is displayed in seconds but internally collected in ms

## 6) Tips
- Choose the correct interpreter in your editor if imports show warnings.
- If only using FastSAM, `segment_anything` is optional. For `model_type: sam`, install it and a SAM checkpoint.
- Prefer `google-genai`; legacy `google-generativeai` is deprecated and not required.
- Use absolute image paths for Gemini image calls.

## 6.1) Segmentation Modules (Detailed)

### 6.1.1) segmenter.py - Core Segmentation Engine
**Function**: Multi-scale object segmentation using FastSAM
**Key Class**: `Segmenter`

**Input/Output**:
- Input: `np.ndarray` image (RGB format)
- Output: List of mask dictionaries with keys:
  ```python
  {
      "region_id": int,           # Unique identifier
      "segmentation": np.ndarray, # Binary mask (uint8)
      "bbox": [x0, y0, x1, y1],  # Bounding box coordinates
      "area": int,                # Mask pixel count
      "confidence": float         # Detection confidence
  }
  ```

**Key Parameters** (config.yaml):
```yaml
segmenter:
  model_type: "fastsam"          # Only FastSAM supported
  model_path: "models/FastSAM-s.pt"
  device: "cuda"                 # cuda/cpu/auto
  min_mask_area: 2000           # Minimum mask area filter
  input_resize: 1024            # Max image dimension
  max_objects: 20               # Max objects per image
```

**FastSAM Internal Config** (optimized):
- `imgsz: 640` - Inference resolution
- `conf: 0.5` - Confidence threshold
- `iou: 0.5` - NMS IoU threshold
- `retina_masks: True` - High-precision masks
- `max_det: 30` - Maximum detections

**Run Test**:
```bash
python -c "
from segmenter import Segmenter
import cv2
seg = Segmenter('fastsam', 'models/FastSAM-s.pt', device='cpu')
img = cv2.imread('test.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
masks = seg.segment(img_rgb)
print(f'Found {len(masks)} objects')
"
```

### 6.1.2) mask_postprocess.py - Mask Refinement Pipeline
**Function**: Multi-stage mask post-processing and filtering
**Key Function**: `mask_postprocess()`

#### **Where it's called**:
- **Primary call**: `Segmenter.segment()` → `mask_postprocess()` (when `use_postprocess=True`)
- **Multi-scale call**: `AdvancedSegmenter.multi_scale_segmentation()` → `Segmenter.segment()` → `mask_postprocess()`
- **Pipeline integration**: Called automatically in `adapter.py` via multi-scale segmentation

#### **Control Parameters**:
```python
# In segmenter.py:
masks = self.base_segmenter.segment(scaled_image, use_postprocess=True)  # Always True in pipeline

# In adapter.py:
masks = self.advanced_segmenter.multi_scale_segmentation(image_rgb, scales=scales, use_postprocess=True)
```

**Note**: `use_postprocess` can now be controlled via config (default: False for improved performance).

#### **Processing Pipeline**:
1. **Area Filter**: Remove too small/large masks
2. **Contrast Filter**: Remove low-contrast regions (background)
3. **Shape Complexity**: Remove overly simple shapes
4. **Morphological Close**: Fill holes, smooth boundaries
5. **Connected Components**: Keep largest region per mask
6. **Overlap Merge**: Merge highly overlapping masks (IoU > 0.7)
7. **Small Merge**: Merge small masks into larger ones
8. **NMS**: Remove duplicate detections (IoU > 0.5)
9. **Count Limit**: Keep top N masks by area

**Note**: Mask post-processing can be disabled via config:
```yaml
segmenter:
  use_postprocess: false  # Disable all post-processing steps
```

#### **Key Parameters** (passed from config):
```python
mask_postprocess(
    masks, 
    min_area=3600,              # From config: min_mask_area
    max_masks=20,               # From config: segmenter.max_objects
    morph_close=True,           # Always enabled
    morph_kernel_size=5,        # Hardcoded
    merge_overlapping=True,     # Always enabled
    overlap_thresh=0.7,         # Hardcoded
    merge_small=True,           # Always enabled
    small_area_thresh=1000,     # Hardcoded
    nms=True,                   # Always enabled
    nms_thresh=0.5,             # Hardcoded
    filter_by_contrast=True,    # Always enabled
    min_contrast=15             # Hardcoded
)
```

**Config Parameters**:
```yaml
segmenter:
  max_objects: 20               # max_masks in mask_postprocess
min_mask_area: 3600             # min_area in mask_postprocess
```

**Debug Output**: Each step prints mask count for monitoring

### 6.1.3) advanced_segmentation_methods.py - Multi-scale Strategies
**Function**: Multi-scale segmentation and IoU-based fusion
**Key Class**: `AdvancedSegmenter`

#### ✅ **Currently Used Methods** (in adapter.py):

**`multi_scale_segmentation()`** - Main method used in pipeline:
- **Purpose**: Run segmentation at multiple scales and fuse results
- **Config scales**: `[0.5, 1.0, 1.5]` (50%, 100%, 150% of original size)
- **Detailed workflow**:
  ```python
  # 1. For each scale (0.5, 1.0, 1.5):
  new_h, new_w = int(h * scale), int(w * scale)
  scaled_image = cv2.resize(resized_image, (new_w, new_h))
  
  # 2. Run FastSAM on scaled image
  masks = self.base_segmenter.segment(scaled_image, use_postprocess=use_postprocess)
  
  # 3. Resize masks back to original resized shape
  scaled_mask = cv2.resize(mask["segmentation"], (w, h))
  
  # 4. Adjust bbox coordinates proportionally
  new_bbox = [int(x0 * w/new_w), int(y0 * h/new_h), ...]
  
  # 5. Collect all masks from all scales
  all_masks.append({...mask, "scale": scale})
  ```

**`_smart_mask_fusion()`** - Called internally by multi_scale_segmentation():
- **Purpose**: IoU-based mask fusion to handle over-segmentation and overlaps
- **Multi-stage Strategy**: 
  1. Sort masks by area (largest first)
  2. Compute IoU matrix between all mask pairs
  3. **New**: `_should_fuse_masks()` - Multi-criteria overlap detection:
     - Basic IoU threshold: 0.3 (reduced from 0.6 for improved fusion)
     - Containment detection: >80% contained masks are fused
     - Significant overlap: IoU > 0.15 + containment > 30%
  4. Fuse similar masks using logical OR + morphological closing
  5. **New**: `_remove_contained_masks()` - Post-processing step:
     - Remove masks >70% contained within larger masks
     - Eliminates redundant inner crops
  6. Filter out overly complex masks (likely background)
- **Key improvements**: Multi-criteria containment handling, reduced IoU threshold

#### ❌ **Unused Methods** :

**`evaluate_mask_quality()`** - Quality assessment:
- **Purpose**: Evaluate mask quality using multiple metrics
- **Metrics**: `boundary_smoothness`, `internal_consistency`, `shape_reasonableness`
- **Usage**: Not currently used in pipeline

**`quality_driven_improvement()`** - Mask improvement:
- **Purpose**: Improve low-quality masks using morphological operations
- **Usage**: Not currently used in pipeline

**`confidence_weighted_fusion()`** - Weighted fusion:
- **Purpose**: Fuse masks with confidence weighting
- **Usage**: Not currently used in pipeline

**`adaptive_prompt_selection()`** - Prompt selection:
- **Purpose**: Select optimal prompt type based on region features
- **Usage**: Not currently used in pipeline

#### **Integration in Pipeline**:
```python
# In adapter.py:
from advanced_segmentation_methods import AdvancedSegmenter

# Create advanced segmenter
advanced_seg = AdvancedSegmenter(base_segmenter)

# Use multi-scale segmentation (current method)
masks = advanced_seg.multi_scale_segmentation(
    image, 
    scales=[0.5, 1.0, 1.5],  # From config.yaml
    use_postprocess=True
)
```

#### **Config Options** (in config.yaml):
```yaml
segmenter:
  multi_scale_scales: [0.5, 1.0, 1.5]  # Scale factors (50%, 100%, 150%)
  use_postprocess: false                # Enable/disable mask post-processing
  # Note: fusion_iou_thresh: 0.3 and containment_thresh: 0.7 are hardcoded
```

**Run Test**:
```bash
python advanced_segmentation_methods.py  # Has built-in test
```

### 6.1.4) crop_generator.py - Object Cropping
**Function**: Generate high-resolution crops from segmentation masks
**Key Function**: `crop_objects()`

**Input/Output**:
- Input: `image`, `masks`, `output_dir`, `image_id`
- Output: List of crop dictionaries:
```python
{
    "region_id": int,
    "crop_path": str,           # Saved crop file path
    "bbox": [x0, y0, x1, y1],  # Original image coordinates
    "area": int,                # Original mask area
    "crop_size": "WxH"         # Crop dimensions
}
```

**Adaptive Cropping Strategy**:
- **High-Res Mode**: Use original image for crops, scale coordinates from resized
- **Fallback Mode**: Use current image if original not provided
- **Quality Filter**: Remove crops with low contrast (< 10 std)
- **Size Filter**: Skip crops < 5x5 pixels

**Key Parameters**:
```python
crop_objects(
    image, masks, output_dir, image_id,
    min_mask_area=200,          # Minimum area for cropping
    min_contrast=10,            # Minimum contrast threshold
    original_image=None,        # High-res source image
    original_shape=None         # Original image dimensions
)
```

**File Naming**: `{image_id}_crop{idx:03d}.jpg` (e.g., `scan_123_crop001.jpg`)

### 6.1.5) Segmentation Pipeline Integration
**Actual Data Flow** :
```
Original Image (np.ndarray, RGB, H×W×3)
    ↓
AdvancedSegmenter.multi_scale_segmentation()
    ├── For each scale (0.5, 1.0, 1.5):
    │   ├── Scale image → scaled_image (np.ndarray)
    │   ├── Segmenter.segment(scaled_image) → raw masks (List[Dict])
    │   │   ├── FastSAM.predict() → YOLO results
    │   │   ├── Extract masks → List[Dict] with keys:
    │   │   │   {"region_id": int, "segmentation": np.ndarray, "bbox": [x0,y0,x1,y1], "area": int, "confidence": float}
    │   │   └── mask_postprocess() → refined masks (List[Dict])
    │   ├── Resize masks back to original scale
    │   └── Collect all masks from all scales
    └── _smart_mask_fusion() → fused masks (List[Dict])
    ↓
crop_objects() → crop files + metadata
    ├── Input: resized_image, masks, crop_dir, image_id
    ├── For each mask: generate high-res crop from original image
    └── Output: List[Dict] with keys:
        {"region_id": int, "crop_path": str, "bbox": [x0,y0,x1,y1], "area": int, "crop_size": "W×H"}
```

**Data Types & Structures**:
```python
# Input:
image_rgb: np.ndarray  # Shape: (H, W, 3), dtype: uint8, RGB format

# Intermediate masks (from segmenter):
masks: List[Dict] = [
    {
        "region_id": int,           # 0, 1, 2, ...
        "segmentation": np.ndarray, # Shape: (H, W), dtype: uint8, binary mask
        "bbox": [x0, y0, x1, y1],  # Bounding box coordinates
        "area": int,                # Pixel count in mask
        "confidence": float,        # Detection confidence (0.0-1.0)
        "scale": float              # Scale factor (0.5, 1.0, 1.5) - added by multi-scale
    }
]

# Final crops (from crop_objects):
crops: List[Dict] = [
    {
        "region_id": int,           # Same as mask region_id
        "crop_path": str,           # Absolute path to saved crop file
        "bbox": [x0, y0, x1, y1],  # Coordinates in original image
        "area": int,                # Original mask area
        "crop_size": "W×H"         # Crop dimensions as string
    }
]
```

**Config Integration**:
```yaml
segmenter:
  model_type: "fastsam"
  model_path: "models/FastSAM-s.pt"
  device: "cuda"
  min_mask_area: 2000
  input_resize: 1024
  max_objects: 20

# Advanced options (used in adapter.py)
advanced_segmentation:
  use_multi_scale: true
  scales: [0.8, 1.0, 1.2]
  fusion_iou_thresh: 0.6
  quality_improvement: true
```

**Debugging Tips**:
1. **Too many masks**: Increase `min_mask_area`, reduce `max_objects`
2. **Too few masks**: Decrease `min_mask_area`, check `input_resize`
3. **Poor segmentation**: Try different scales, adjust FastSAM `conf`/`iou`
4. **Over-segmentation**: Increase `overlap_thresh`, enable `merge_small`
5. **Low crop quality**: Check `min_contrast`, verify `original_image` passed

#### **Performance Optimization**:
- **CPU vs GPU**: Set `device: "cuda"` if GPU available, `"cpu"` for CPU-only
- **Memory usage**: Reduce `input_resize` (1024→640) for large images
- **Speed vs Quality**: Adjust scales `[0.5, 1.0, 1.5]` - fewer scales = faster
- **Model size**: Use FastSAM-s (small) vs FastSAM-x (extra large)

#### **Common Issues & Solutions**:
```bash
# Issue: "No masks found"
# Check: min_mask_area too high, input_resize too small
# Fix: Reduce min_mask_area from 3600 to 1000, increase input_resize

# Issue: "Too many small masks"
# Check: min_mask_area too low
# Fix: Increase min_mask_area from 3600 to 5000+

# Issue: "Poor object boundaries"
# Check: FastSAM confidence too low
# Fix: Increase conf from 0.4 to 0.6 in config.yaml

# Issue: "Missing small objects"
# Check: scales don't include small enough scale
# Fix: Add 0.3 scale: [0.3, 0.5, 1.0, 1.5]
```

#### **Monitoring & Logs**:
- **Console output**: Each step prints mask counts for monitoring
- **Visualization**: Check `scan_<id>_crops_visualization.jpg` for segmentation quality
- **Debug mode**: Set `verbose=True` in FastSAM config for detailed logs

#### **Quick Reference - Segmentation Pipeline**:
```bash
# Test single module
python -c "from segmenter import Segmenter; seg=Segmenter('fastsam','models/FastSAM-s.pt','cpu'); print('OK')"

# Test full segmentation
python adapter.py test.jpg --scan-id 12345 --config config.yaml

# Check outputs
ls outputs/scan_12345_*           # JSON, TXT, visualization, metrics
ls crops/scan_12345/              # Individual crop files
```

**Key Files Generated**:
- `outputs/scan_<id>_result.json` - Structured results
- `outputs/scan_<id>_result.txt` - Human-readable results  
- `outputs/scan_<id>_crops_visualization.jpg` - Visual overview
- `outputs/scan_<id>_gemini_metrics.json` - Performance metrics
- `crops/scan_<id>/` - Individual crop files

**Essential Config Parameters**:
```yaml
segmenter:
  model_path: "models/FastSAM-s.pt"     # Model file
  multi_scale_scales: [0.5, 1.0, 1.5]   # Scale factors
  use_postprocess: false                 # Enable/disable mask post-processing
min_mask_area: 3600                      # Min object size
device: "cpu"                            # cuda/cpu/auto
```

## 6.2) CLIP Modules (Detailed)

### 6.2.0) CLIP Design Philosophy & Customization
**Core Design Principles**:
- **Generic & Balanced**: Current template maintains balance across all waste categories without bias
- **Neutral Descriptions**: No emphasis on specific attributes (position, material properties, shape, size)
- **Preliminary Scoring**: CLIP provides initial rough scores, not final classification
- **Statistical Analysis**: Results analyzer processes CLIP scores and generates strategy-based prompts
- **Gemini Integration**: CLIP results are enhanced and merged into final Gemini prompts

**Customization Strategy**:
- **Task-Specific Templates**: Generate specialized `waste_labels_template.json` for specific scenarios
- **Threshold Adjustment**: Modify `clip_threshold` and unknown/background classification rules
- **Prompt Optimization**: Adapt result analyzer prompts for domain-specific requirements
- **Category Focus**: Emphasize specific attributes (material, shape, context) when needed

**Current Template Characteristics**:
- **Balanced Coverage**: Equal treatment across Plastics, Metals, Papers, Glass, Textiles, E-waste, Organic
- **Minimal Bias**: No positional or contextual assumptions
- **Concise Prompts**: 5 prompts per category to maintain computational efficiency
- **Hierarchical Structure**: Clear main/sub relationships for consistent classification

**When to Customize**:
- **Domain-Specific Tasks**: Medical waste, industrial waste, specific material types
- **Context-Aware Scenarios**: Location-based classification, size-sensitive applications
- **Performance Tuning**: Fine-tune for specific recognition challenges
- **Integration Requirements**: Adapt to downstream system requirements

### 6.2.1) waste_labels_template.json - Label Configuration
**Function**: Waste classification taxonomy with main/sub categories and prompts
**Structure**: JSON array of label objects

**Label Structure**:
```json
{
  "main": "Plastics",           # Main material category
  "sub": "bottle",              # Specific item type
  "prompts": [                  # CLIP text prompts (5 per category)
    "this is a Plastics item, specifically a bottle",
    "a bottle which is a type of Plastics",
    "a Plastics object in bottle form",
    "Plastics bottle",
    "bottle made of Plastics material"
  ]
}
```

**Main Categories** (9 total):
- **Plastics**: bottle, bag, box, container, film, cup, other
- **Metals**: can, foil, lid, aerosol can, other  
- **Papers**: newspaper, carton, tissue, flyer, envelope, other
- **Glass**: bottle, jar, cup, fragment, other
- **Textiles**: cloth, shoe, towel, glove, other
- **E-waste**: battery, phone, cable, charger, other
- **Organic**: food, fruit peel, vegetable, bone, other
- **Background**: background (empty/no-object regions)
- **Unknown**: unknown (unrecognized items)

**Total**: 42 main/sub combinations, ~210 prompts

**Generate/Update Labels**:
```bash
python generate_label_template.py  # Regenerate waste_labels_template.json
```

### 6.2.2) clip_matcher.py - CLIP Classification Engine
**Function**: CLIP-based image-text matching for waste categorization
**Key Class**: `CLIPMatcher`

#### **Core Features**:
- **Batch Processing**: Efficient batch encoding of multiple crops
- **Embedding Caching**: Text embeddings cached to disk, image embeddings in memory
- **Unknown Fallback**: Multi-criteria classification with confidence thresholds
- **Multi-Prompt Aggregation**: Combines scores from multiple prompts per category

#### **Input/Output**:
```python
# Input: List of crop dictionaries
crop_infos = [
    {
        "region_id": 0,
        "crop_path": "crops/scan_123_crop001.jpg",
        "bbox": [x0, y0, x1, y1],
        "area": 1500
    }
]

# Output: Rich classification results
results = [
    {
        "region_id": 0,
        "crop_path": "crops/scan_123_crop001.jpg",
        "main_label": "Plastics",
        "sub_label": "bottle", 
        "confidence": 0.85,
        "top_k": [
            {"main_label": "Plastics", "sub_label": "bottle", "score": 0.85},
            {"main_label": "Plastics", "sub_label": "container", "score": 0.12}
        ],
        "probabilities": {"Plastics|bottle": 0.85, "Plastics|container": 0.12},
        "classification_context": {...},
        "prompt_details": {...}
    }
]
```

#### **Key Parameters** (config.yaml):
```yaml
recognizer:
  model_type: "clip"
  label_config_path: "waste_labels_template.json"
  cache_embeddings: true
  embedding_cache_path: "clip_embeddings_cache.pkl"
  batch_size: 16

clip_threshold: 0.1  # Below this → Unknown classification
```

#### **CLIP Model Details**:
- **Model**: ViT-B/32 (Vision Transformer Base, 32x32 patches)
- **Embedding Dimension**: 512 (both image and text embeddings)
- **Preprocessing**: Standard CLIP preprocessing (resize to 224x224, normalize)
- **Device Support**: CUDA/CPU with automatic device detection

#### **Unknown Classification Strategy** (Relaxed Rules):
1. **Absolute Threshold**: Score < `clip_threshold * 0.5` → Unknown (more strict)
2. **Low Distribution**: Mean < 0.05, std < 0.02, best < threshold → Unknown (stricter)  
3. **Low Confidence Gap**: Gap to 2nd best < 0.02 → Lower confidence (more lenient)
4. **Background Special**: Background score < 0.2 → Unknown (relaxed from 0.3)

**Note**: Rules have been relaxed to reduce false "unknown" classifications while maintaining accuracy.

#### **Key Methods**:
```python
# Batch classification
results = matcher.classify_batch(crop_infos, batch_size=32)

# Single classification (backward compatibility)
result = matcher.classify_single(crop_info)

# Cache management
matcher.clear_image_cache()
matcher.save_image_cache("custom_cache.pkl")
matcher.load_image_cache("custom_cache.pkl")

# Statistics
stats = matcher.get_embedding_stats()
```

#### **Caching System**:
- **Text Embeddings**: Cached to disk (`clip_embeddings_cache.pkl`)
- **Image Embeddings**: Cached in memory (per session)
- **Cache Key**: Based on prompts hash (text) or file mtime+size (images)

**Run Test**:
```bash
python -c "
from clip_matcher import CLIPMatcher
matcher = CLIPMatcher('waste_labels_template.json', device='cpu')
print('CLIP model loaded:', matcher.get_embedding_stats())
"
```

### 6.2.3) clip_results_analyzer.py - Results Processing & Analysis
**Function**: Unified CLIP results management, statistical analysis, and strategy-based prompt generation
**Key Class**: `CLIPResultsAnalyzer`

#### **Three-Layer Architecture**:

**1. Data Management Layer**:
- Save/load CLIP results to JSON
- Smart caching based on image content + config
- Pipeline integration with existing workflow

**2. Statistical Analysis Layer**:
- **Dominance Coefficient**: First score / second score ratio
- **Effective Variation Coefficient**: Score distribution measure
- **Score Gradient**: Rate of score decline across top-K
- **Confidence Classification**: High/Medium/Low based on thresholds

**3. Formatting Layer**:
- **Strategy-based Prompt Generation**: Statistical templates
- **Statistical Context**: Background analysis for Gemini
- **Template Selection**: Based on analysis strategy

#### **Analysis Strategies**:
```python
# Strategy determination logic:
if abs_confidence < 0.03:
    strategy = "descriptive_analysis"      # Very low confidence
elif dominance_coef >= 4.0:
    strategy = "focus_validation"          # Strong dominance
elif dominance_coef >= 2.0:
    strategy = "dual_comparison"           # Moderate competition
else:
    strategy = "multi_candidate"           # High competition
```

#### **Prompt Templates**:
- **Single Dominant**: Clear winner, focus validation
- **Dual Competitive**: Two viable options, compare both
- **Multi Competitive**: Multiple options, systematic evaluation
- **Low Confidence**: Independent statistical analysis

#### **Key Methods**:
```python
# One-stop analysis
result = analyzer.analyze_and_format_for_gemini(
    image_path, clip_results, config, force_save=False
)

# Returns:
{
    "json_path": "path/to/saved/results.json",
    "cache_hit": True/False,
    "raw_results": [...],
    "statistics": [...],
    "enhanced_prompt": "Generated prompt context",
    "dominant_strategy": "focus_validation",
    "summary": {...},
    "metadata": {...}
}
```

#### **Cache System**:
- **Smart Cache**: Content-aware (image hash + config hash)
- **Simple Cache**: Filename-based fallback
- **Statistics Cache**: In-memory statistical features cache

**Run Test**:
```bash
python clip_results_analyzer.py  # Has built-in test
```

### 6.2.4) generate_label_template.py - Label Generator
**Function**: Generate waste classification label templates
**Output**: `waste_labels_template.json`

#### **Design Principles**:
1. **Objectivity**: Focus on object type, avoid subjective states
2. **Brevity**: Max 5 prompts per category to reduce cost
3. **Hierarchy**: Preserve main-sub relationship
4. **Main Class Presence**: Every prompt mentions the main class
5. **Other Handling**: Catch-all prompts for "other" subclass

#### **Current Template Strategy**:
- **Generic Approach**: Balanced, unbiased descriptions across all categories
- **Neutral Attributes**: No emphasis on position, material properties, shape, or size
- **Preliminary Scoring**: Designed for initial CLIP scoring, not final classification
- **Statistical Processing**: Results will be processed by analyzer for intelligent prompt generation

#### **Prompt Generation Logic**:
```python
# Regular categories - Generic, balanced approach
prompts = [
    f"{main} {sub}",                           # "Plastics bottle"
    f"a {sub} which is a type of {main}",     # "a bottle which is a type of Plastics"
    f"this is a {main} item, specifically a {sub}",  # "this is a Plastics item, specifically a bottle"
    f"{sub} made of {main} material",         # "bottle made of Plastics material"
    f"a {main} object in {sub} form"          # "a Plastics object in bottle form"
]

# "Other" subclass - Catch-all approach
prompts = [
    f"other {main} item",     # "other Plastics item"
    f"a {main} object",       # "a Plastics object"
    f"this is a {main} item", # "this is a Plastics item"
    f"a piece of {main} material",  # "a piece of Plastics material"
    f"{main} waste"           # "Plastics waste"
]
```

#### **Customization Examples**:
```python
# Example 1: Medical waste - Add context awareness
medical_prompts = [
    f"medical {main} {sub}",                    # "medical Plastics syringe"
    f"disposable {main} {sub}",                # "disposable Plastics syringe"
    f"sterile {main} {sub}",                   # "sterile Plastics syringe"
    f"clinical {main} {sub}",                  # "clinical Plastics syringe"
    f"healthcare {main} {sub}"                 # "healthcare Plastics syringe"
]

# Example 2: Size-sensitive - Add dimensional attributes
size_prompts = [
    f"large {main} {sub}",                     # "large Plastics bottle"
    f"small {main} {sub}",                     # "small Plastics bottle"
    f"compact {main} {sub}",                   # "compact Plastics bottle"
    f"oversized {main} {sub}",                 # "oversized Plastics bottle"
    f"miniature {main} {sub}"                  # "miniature Plastics bottle"
]

# Example 3: Material-focused - Emphasize material properties
material_prompts = [
    f"transparent {main} {sub}",               # "transparent Plastics bottle"
    f"opaque {main} {sub}",                    # "opaque Plastics bottle"
    f"flexible {main} {sub}",                  # "flexible Plastics bottle"
    f"rigid {main} {sub}",                     # "rigid Plastics bottle"
    f"durable {main} {sub}"                    # "durable Plastics bottle"
]
```

#### **Customization Workflow**:
1. **Modify Categories**: Edit `main_categories` in `generate_label_template.py`
2. **Adjust Prompts**: Customize `generate_prompts()` function for specific attributes
3. **Update Thresholds**: Modify `clip_threshold` in config for new classification rules
4. **Adapt Analyzer**: Update prompt templates in `clip_results_analyzer.py`
5. **Test & Validate**: Run tests with domain-specific data

**Run Test**:
```bash
python generate_label_template.py  # Generates waste_labels_template.json
```

### 6.2.5) clip_results_cache/ - Results Storage
**Function**: Persistent storage for CLIP classification results
**Structure**: JSON files with metadata and results

#### **File Naming**:
```
{image_name}_{image_hash}_{config_hash}_clip_results.json
```

#### **File Structure**:
```json
{
  "metadata": {
    "image_path": "/path/to/image.jpg",
    "image_name": "ALTRIANE_image0", 
    "total_crops": 3,
    "config": {...}
  },
  "clip_results": [
    {
      "region_id": 5,
      "crop_path": "crops/scan_123_crop000.jpg",
      "main_label": "unknown",
      "sub_label": "unknown", 
      "confidence": 0.084,
      "top_k": [...],
      "top_main_categories": [...],
      "classification_summary": {...},
      "unknown_info": {...}
    }
  ],
  "summary": {
    "total_crops": 3,
    "unknown_count": 3,
    "identification_rate": 0.0,
    "main_categories": {"unknown": 3},
    "confidence_distribution": {"high": 0, "medium": 0, "low": 3}
  }
}
```

### 6.2.6) CLIP Pipeline Integration

#### **Detailed Data Flow**:
```
Input: Crop Images (from segmentation)
    ↓
┌─────────────────────────────────────────┐
│ 1. CLIPMatcher.classify_batch()         │
│ Input: List[Dict] - crop_infos          │
│ [                                     │
│   {                                   │
│     "region_id": 0,                   │
│     "crop_path": "crops/scan_123_crop001.jpg", │
│     "bbox": [x0, y0, x1, y1],         │
│     "area": 1500                      │
│   }                                   │
│ ]                                     │
│ ↓                                     │
│ Output: List[Dict] - raw CLIP results │
│ [                                     │
│   {                                   │
│     "region_id": 0,                   │
│     "main_label": "Plastics",         │
│     "sub_label": "bottle",            │
│     "confidence": 0.85,               │
│     "top_k": [...],                   │
│     "probabilities": {...},           │
│     "classification_context": {...}   │
│   }                                   │
│ ]                                     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. CLIPResultsAnalyzer.analyze_and_format() │
│ Input: image_path + raw CLIP results   │
│ ↓                                     │
│ Process:                              │
│ - Statistical analysis (dominance_coef, │
│   effective_vc, score_gradient)       │
│ - Strategy determination              │
│ - Enhanced prompt generation          │
│ ↓                                     │
│ Output: Dict - enhanced context       │
│ {                                     │
│   "json_path": "path/to/results.json", │
│   "raw_results": [...],               │
│   "statistics": [...],                │
│   "enhanced_prompt": "Generated...",  │
│   "dominant_strategy": "focus_validation" │
│ }                                     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. Save to clip_results_cache/          │
│ Input: Enhanced context + metadata     │
│ ↓                                     │
│ Output: JSON file                      │
│ {                                     │
│   "metadata": {...},                  │
│   "clip_results": [...],              │
│   "summary": {...}                    │
│ }                                     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. Pass to Gemini prompt builder       │
│ Input: Enhanced prompt context         │
│ ↓                                     │
│ Output: Final unified prompt           │
│ "CROP 0 Analysis (ID: 0):             │
│ CLIP Statistical Context:              │
│ • Top candidate: Plastics/bottle...   │
│ ..."                                  │
└─────────────────────────────────────────┘
```

#### **Data Types & Structures**:

**Input Data Types**:
```python
# crop_infos: List[Dict]
crop_infos = [
    {
        "region_id": int,           # Crop identifier
        "crop_path": str,           # Path to crop image file
        "bbox": List[int],          # [x0, y0, x1, y1] coordinates
        "area": int                 # Pixel area of crop
    }
]

# image_path: str
image_path = "/path/to/original/image.jpg"

# config: Dict (optional)
config = {
    "recognizer": {...},
    "clip_threshold": 0.1,
    "enable_smart_cache": True
}
```

**Intermediate Data Types**:
```python
# Raw CLIP results: List[Dict]
raw_results = [
    {
        "region_id": int,
        "crop_path": str,
        "main_label": str,          # "Plastics", "Metals", etc.
        "sub_label": str,           # "bottle", "can", etc.
        "confidence": float,        # 0.0-1.0
        "top_k": List[Dict],        # Top-K candidates with scores
        "probabilities": Dict,      # All label probabilities
        "classification_context": Dict,  # Context for downstream
        "prompt_details": Dict,     # Prompt-level statistics
        "unknown_info": Dict        # Only if main_label == "unknown"
    }
]

# Statistical analysis: CLIPStatistics object
statistics = CLIPStatistics(
    best_label=Tuple[str, str],     # (main, sub)
    best_score=float,
    confidence_level=str,           # "high"/"medium"/"low"
    dominance_coef=float,           # First/second score ratio
    effective_vc=float,             # Variation coefficient
    score_gradient=float,           # Score decline rate
    analysis_strategy=str,          # Strategy type
    prompt_template=str             # Template type
)
```

**Output Data Types**:
```python
# Enhanced context: Dict
enhanced_context = {
    "json_path": str,               # Path to saved JSON
    "cache_hit": bool,              # Whether cache was used
    "raw_results": List[Dict],      # Original CLIP results
    "statistics": List[Dict],       # Statistical features
    "enhanced_prompt": str,         # Generated prompt context
    "dominant_strategy": str,       # Most common strategy
    "summary": Dict,                # Overall summary
    "metadata": Dict                # File metadata
}

# Saved JSON structure
json_structure = {
    "metadata": {
        "image_path": str,
        "image_name": str,
        "total_crops": int,
        "config": Dict
    },
    "clip_results": List[Dict],     # Same as raw_results
    "summary": {
        "total_crops": int,
        "unknown_count": int,
        "identification_rate": float,
        "main_categories": Dict,
        "confidence_distribution": Dict
    }
}
```

#### **Config Integration**:
```yaml
recognizer:
  model_type: "clip"
  label_config_path: "waste_labels_template.json"
  cache_embeddings: true
  embedding_cache_path: "clip_embeddings_cache.pkl"
  batch_size: 16

clip_results_cache_dir: "clip_results_cache"
enable_smart_cache: true
clip_threshold: 0.1  # Unknown classification threshold
```

#### **Threshold Strategy & Uncertainty Handling**:
**Current Approach**:
- **Preliminary Scoring**: CLIP provides initial rough scores, not final decisions
- **Threshold Uncertainty**: `clip_threshold` (0.1) has inherent uncertainty for unknown/background classification
- **Statistical Analysis**: Results analyzer processes CLIP scores to generate intelligent prompts
- **Gemini Integration**: Enhanced prompts merge CLIP context into final Gemini analysis

**Threshold Customization**:
```yaml
# Conservative approach - More unknowns
clip_threshold: 0.3  # Higher threshold → more "unknown" classifications

# Aggressive approach - Fewer unknowns  
clip_threshold: 0.05 # Lower threshold → fewer "unknown" classifications

# Task-specific thresholds
clip_threshold: 0.2  # Balanced for general waste classification
```

**Classification Rules**: See Unknown Classification Strategy in 6.2.2

**Result Processing Strategy**:
- **Statistical Analysis**: Dominance coefficient, variation coefficient, score gradient analysis
- **Strategy Selection**: Focus validation, dual comparison, multi-candidate, descriptive analysis
- **Prompt Generation**: Strategy-based templates based on statistical features
- **Gemini Enhancement**: CLIP context merged into structured analysis prompts

#### **Performance Optimization**:
- **Batch Processing**: Process multiple crops simultaneously
- **Embedding Caching**: Avoid recomputing text embeddings
- **Content-based Cache**: Reuse results for identical images/configs
- **Memory Management**: Image embeddings cached in memory only

**Debugging Tips**:
1. **Low Recognition Rate**: Check `clip_threshold`, verify label prompts
2. **All Unknown**: Increase `clip_threshold`, check image quality
3. **Cache Issues**: Clear `clip_embeddings_cache.pkl`, regenerate
4. **Slow Performance**: Reduce `batch_size`, enable caching
5. **Memory Issues**: Disable image embedding cache

**Quick Reference - CLIP Pipeline**:
```bash
# Test CLIP matcher
python -c "from clip_matcher import CLIPMatcher; m=CLIPMatcher('waste_labels_template.json','cpu'); print('OK')"

# Test results analyzer  
python clip_results_analyzer.py

# Generate labels
python generate_label_template.py

# Check cache
ls clip_results_cache/
```

**Key Files Generated**:
- `clip_embeddings_cache.pkl` - Text embeddings cache
- `clip_results_cache/*.json` - Classification results
- `waste_labels_template.json` - Label configuration

### 6.2.7) CLIP Usage in Adapter
**How CLIP is Called in the Main Pipeline**:

```python
# In adapter.py - Actual workflow (from lines 371-408)
def process_image(self, image_path: str, scan_id: int):
    # 1. Segmentation → Generate crops (lines 344-361)
    crops = crop_objects(image_rgb, masks, per_scan_crop_dir, ...)
    
    # 2. CLIP Classification (lines 371-378)
    batch_size = self.cfg.get("recognizer", {}).get("batch_size", 32)
    clip_results = self.clip_matcher.classify_batch(crops, batch_size=batch_size)
    
    # 3. Optional CLIP Results Saving (lines 380-392)
    if out_cfg.get("save_clip_results", False):
        result = self.clip_analyzer.integrate_with_existing_pipeline(
            image_path=image_path,
            existing_clip_results=clip_results,
            config=self.cfg,
            force_save=True
        )
    
    # 4. Prompt Building per Crop (lines 394-408)
    for crop, pred in zip(crops, clip_results):
        prompt = self.prompt_builder.build_unified_prompt(
            [pred],
            include_statistics=True,
            include_label_context=self._gemini_include_label_context,
            uncertainty_focus=self._gemini_uncertainty_focus
        )
        requests.append({"prompt": prompt, "image_path": crop.get("crop_path")})
```

**Key Integration Points**:
```python
# Initialization in _init_clip() (lines 80-94)
def _init_clip(self):
    rec_cfg = self.cfg.get("recognizer", {})
    label_config_path = rec_cfg.get("label_config_path", "waste_labels_template.json")
    # Path resolution and validation...
    self.clip_matcher = CLIPMatcher(
        label_config_path=label_config_path,
        device=self.device,
        clip_threshold=self.cfg.get("clip_threshold", 0.25),
        cache_embeddings=rec_cfg.get("cache_embeddings", True),
        embedding_cache_path=rec_cfg.get("embedding_cache_path", "clip_embeddings_cache.pkl")
    )

# CLIP Analyzer initialization (lines 48-51)
self.clip_analyzer = create_clip_results_analyzer(
    cache_dir=self.cfg.get("clip_results_cache_dir", "clip_results_cache"),
    enable_smart_cache=self.cfg.get("enable_smart_cache", True)
)
```

**Input from Segmentation**:
```python
# Crops from segmentation pipeline
crops = [
    {
        "region_id": 0,
        "crop_path": "crops/scan_123_crop001.jpg",
        "bbox": [x0, y0, x1, y1],
        "area": 1500
    }
]
```

**Output to Gemini**:
```python
# Enhanced prompt context passed to Gemini
enhanced_prompt = """
CROP 0 Analysis (ID: 0):
CLIP Statistical Context:
• Top candidate: Plastics/bottle (score: 0.85)
• Leading margin: 3.2x advantage over next candidate
• Score distribution: Concentrated pattern with clear frontrunner
...
"""
```

**Configuration Integration**:
```yaml
# config.yaml - CLIP settings used by adapter (actual config structure)
recognizer:
  model_type: "clip"
  label_config_path: "waste_labels_template.json"
  cache_embeddings: true
  embedding_cache_path: "clip_embeddings_cache.pkl"
  batch_size: 16

clip_results_cache_dir: "clip_results_cache"
enable_smart_cache: true
clip_threshold: 0.1  # Config file value, adapter default is 0.25 if not set

# Output control for CLIP results saving
output:
  save_clip_results: false  # Optional: save CLIP results to cache
```

**Actual Parameter Mapping**:
```python
# From adapter.py _init_clip() method
clip_threshold = self.cfg.get("clip_threshold", 0.25)  # Default 0.25 if not in config
batch_size = self.cfg.get("recognizer", {}).get("batch_size", 32)  # Default 32
cache_embeddings = rec_cfg.get("cache_embeddings", True)
embedding_cache_path = rec_cfg.get("embedding_cache_path", "clip_embeddings_cache.pkl")

# Note: Current config.yaml sets clip_threshold: 0.1, overriding the 0.25 default
```

### 6.2.8) CLIP Section Summary
**Complete CLIP Pipeline**:
1. **Label Configuration** (6.2.1): 9 main categories, 42 combinations, ~210 prompts
2. **Classification Engine** (6.2.2): ViT-B/32 model, batch processing, caching, unknown handling
3. **Results Analysis** (6.2.3): Statistical analysis, strategy selection, intelligent prompts
4. **Label Generation** (6.2.4): Template generation with customization examples
5. **Results Storage** (6.2.5): JSON persistence with metadata and summaries
6. **Pipeline Integration** (6.2.6): Complete data flow with types and structures
7. **Adapter Usage** (6.2.7): Real-world integration in main pipeline

**Key Strengths**:
- **Modular Design**: Each component has clear responsibilities
- **Performance Optimized**: Batch processing, caching, smart memory management
- **Highly Customizable**: Templates, thresholds, and strategies can be adapted
- **Robust Error Handling**: Unknown classification and fallback mechanisms
- **Developer Friendly**: Comprehensive documentation and testing commands
- **Seamless Integration**: Easy integration into main pipeline workflow

**Integration Points**:
- **Input**: Crop images from segmentation pipeline
- **Output**: Enhanced prompts for Gemini analysis
- **Storage**: Persistent results for caching and debugging
- **Configuration**: Flexible YAML-based parameter management

## 6.3) Gemini Modules (Detailed)

### 6.3.1) gemini_prompt_builder.py - Prompt Generation Engine
**Function**: Transforms CLIP outputs and crop images into comprehensive prompts for Gemini inference
**Key Class**: `GeminiPromptBuilder`

#### **Core Features**:
- **Unified Prompt Generation**: Single method `build_unified_prompt()` for all scenarios
- **CLIP Integration**: Incorporates CLIP statistics and classification context
- **Strategy-Based Templates**: 5 statistical template types based on CLIP analysis patterns
- **Dynamic Formatting**: Auto-detects single vs batch crop analysis
- **Label System Context**: Includes waste classification system reference
- **Confidence Grading**: A/B/C/D confidence level criteria

#### **Input/Output**:
```python
# Input: CLIP results + configuration flags
clip_results = [
    {
        "region_id": 0,
        "main_label": "Plastics",
        "sub_label": "bottle",
        "confidence": 0.85,
        "top_k": [...],
        "classification_context": {...}
    }
]

# Output: Comprehensive prompt text
prompt = prompt_builder.build_unified_prompt(
    clip_results=clip_results,
    include_statistics=True,      # Include CLIP statistical analysis
    include_label_context=True,   # Include waste classification system
    uncertainty_focus=True        # Emphasize uncertainty handling
)
```

#### **Prompt Structure**:
```
1. System Instruction
   - Single crop: "Analyze this cropped image of a potential waste object"
   - Batch: "Analyze these N cropped images of potential waste objects"

2. **NEW: Image Reference & Context**
   - Explicitly label crop vs original image for Gemini
   - Include filenames for better context understanding
   - Provide scene context from original image

3. Multi-Object Cognition Guidance
   - Primary Object identification
   - Secondary Elements detection
   - Relationship analysis

4. CLIP Statistics Context (if enabled)
   - Statistical analysis from CLIP results analyzer
   - Statistical context based on dominance coefficient, variation coefficient
- **Statistical prompt templates** (5 types: single_dominant, dual_competitive, multi_competitive, low_confidence, default)
- **Automatic template selection** based on CLIP statistical patterns

5. Label System Context (if enabled)
   - 12 main waste categories reference
   - Detailed subcategory descriptions
   - Classification system guidelines

6. Analysis Tasks Definition
   - PRIMARY OBJECT ANALYSIS: Material, shape, size, color, condition, purpose, state, visibility
   - SECONDARY ELEMENTS: Other objects, relationships, classification impact
   - OVERALL ASSESSMENT: Category determination, recyclability, confidence

7. Confidence Criteria
   - A (High): Clear features, strong confidence, complete visibility
   - B (Medium): Generally clear, moderate confidence, mostly identifiable
   - C (Low): Unclear features, low confidence, partial visibility
   - D (Unreliable): Very poor quality, very low confidence, incomplete

8. **NEW: Prompt Saving Support**
   - Optional prompt saving to files for debugging
   - Configurable save directory and naming

9. **Future Enhancement Areas**
   - Current focus: waste object analysis within crops
   - Potential improvement: explicit handling of non-waste backgrounds (roads, buildings, infrastructure)
   - Could add dedicated prompt sections for environmental context filtering

10. Output Format Guidance
    - Structured response format
    - Specific sections and requirements
```

#### **Key Parameters** (config.yaml):
```yaml
gemini:
  include_label_context: true    # Include waste classification system
  uncertainty_focus: true        # Emphasize uncertainty handling
  save_prompts: true            # Save prompts to files for debugging
  prompt_save_dir: "prompts"    # Directory for saved prompts
```

#### **CLIP Integration with Strategy Templates**:
```python
# Actual method signature in code:
def build_unified_prompt(self,
                        clip_results: List[Dict],
                        include_statistics: bool = True,
                        include_label_context: bool = True,
                        uncertainty_focus: bool = True,
                        original_image_path: Optional[str] = None,
                        crop_path: Optional[str] = None) -> str:

# Template selection based on statistical patterns:
# - single_dominant: Clear frontrunner (dominance_coef > 2.0)
# - dual_competitive: Two viable options (1.5 < dominance_coef < 2.0)  
# - multi_competitive: Multiple options (dominance_coef < 1.5)
# - low_confidence: Low scores (best_score < threshold)
# - default: Fallback template
```

#### **Debugging & Troubleshooting**:
```python
# Check prompt builder initialization (actual factory function signature)
from gemini_prompt_builder import create_prompt_builder
builder = create_prompt_builder(
    label_config_path='waste_labels_template.json',
    save_prompts=True,
    prompt_save_dir="prompts"
)
print(f"Label system loaded: {builder.label_system is not None}")

# Test prompt generation with sample CLIP results (actual structure from clip_matcher.py)
sample_clip_results = [{
    "region_id": 0,
    "crop_path": "crop_001.jpg",
    "main_label": "Plastics", 
    "sub_label": "bottle",
    "confidence": 0.85,
    "top_k": [
        {"main_label": "Plastics", "sub_label": "bottle", "score": 0.85},
        {"main_label": "Plastics", "sub_label": "container", "score": 0.12}
    ],
    "probabilities": {"Plastics|bottle": 0.85, "Plastics|container": 0.12},
    "classification_context": {...},
    "prompt_details": {...}
}]

prompt = builder.build_unified_prompt(
    sample_clip_results,
    include_statistics=True,           # Matches adapter.py line 401
    include_label_context=True,        # Matches adapter.py line 402
    uncertainty_focus=True,            # Matches adapter.py line 403
    original_image_path="/path/to/original.jpg",  # NEW: Original image context
    crop_path="crop_001.jpg"          # NEW: Crop image path for labeling
)
print(f"Generated prompt length: {len(prompt)} characters")
```

**Run Test**:
```bash
python -c "
from gemini_prompt_builder import create_prompt_builder
builder = create_prompt_builder('waste_labels_template.json')
print('Prompt builder initialized successfully')
"
```

### 6.3.2) gemini_inferencer.py - API Inference Engine
**Function**: Wrapper around Google Gemini API calls, supporting multimodal inference with structured output
**Key Class**: `GeminiInferencer`

#### **Core Features**:
- **Multimodal Input**: Text + image support with automatic file size handling
- **Structured Output**: Pydantic schema integration for consistent JSON responses
- **Multiple Inference Modes**: Single, Batch, Chunked-Batch, and Concurrent processing
- **Response Caching**: Disk-based caching to avoid redundant API calls
- **Error Handling**: Retry mechanisms with exponential backoff
- **Schema Selection**: Auto-select single-crop vs batch-crop schemas

#### **API Configuration**:
```python
# New Google AI SDK support
from google import genai  # New SDK (preferred)
# Fallback: import google.generativeai as genai  # Legacy SDK

# Model configuration
model_name = "gemini-2.5-flash"  # or "gemini-1.5-flash"
generation_config = {
    "temperature": 0.1,           # Low temperature for deterministic output
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 4096,    # Increased to reduce truncation
    "response_mime_type": "application/json",  # Structured output
    "response_schema": schema     # Pydantic schema
}
```

#### **Inference Modes & Usage**:
```python
# 1. Single Mode - Process one request at a time
result = inferencer.infer_single(
    prompt="Analyze this waste object",
    image_path="crop_001.jpg"
)

# 2. Batch Mode - Process multiple requests sequentially
requests = [
    {"prompt": "Analyze crop 1", "image_path": "crop_001.jpg"},
    {"prompt": "Analyze crop 2", "image_path": "crop_002.jpg"}
]
results = inferencer.infer_batch(requests, batch_delay=1.2)

# 3. Concurrent Mode - Process multiple requests in parallel (using infer_single)
results = inferencer._infer_requests_concurrent(requests, max_workers=3)
```

#### **Adapter.py Actual Usage Patterns**:
```python
# In adapter.py - Dynamic mode selection based on configuration
if batch_chunk_size and batch_chunk_size > 1:
    # CHUNKED-BATCH MODE: Split into chunks, process each chunk with infer_batch
    chunks = [requests[i:i + batch_chunk_size] for i in range(0, len(requests), batch_chunk_size)]
    if chunk_workers > 1:
        # Parallel chunk processing
        for chunk in chunks:
            results = inferencer.infer_batch(chunk, batch_delay=1.2)
    else:
        # Sequential chunk processing
        for chunk in chunks:
            results = inferencer.infer_batch(chunk, batch_delay=1.2)
else:
    # PER-CROP MODE: Choose between concurrent or single batch
    if max_workers > 1:
        # CONCURRENT MODE: Use infer_single with ThreadPoolExecutor
        results = inferencer._infer_requests_concurrent(requests, max_workers=3)
    else:
        # SINGLE BATCH MODE: Process all requests in one infer_batch call
        results = inferencer.infer_batch(requests, batch_delay=1.2)
```

#### **Mode Selection Logic**:
```yaml
# config.yaml - Controls which mode is used
output:
  batch_chunk_size: 3        # If > 1: Use Chunked-Batch Mode
  chunk_workers: 2           # If > 1: Parallel chunk processing
  concurrent_requests: 3     # If > 1: Use Concurrent Mode (infer_single)
  # If all are 1 or 0: Use Single Batch Mode (infer_batch)
```

#### **Performance Characteristics**:
- **Single Mode**: Best for 1-2 crops, highest reliability
- **Batch Mode**: Best for 3-10 crops, good balance of speed/reliability  
- **Chunked-Batch Mode**: Best for 10+ crops, handles large batches efficiently
- **Concurrent Mode**: Best for 3-6 crops, fastest but higher API load

#### **Input/Output**:
```python
# Input: Prompt + optional image(s)
result = inferencer.infer_single(
    prompt="Your analysis prompt...",
    image_path="crop_image.jpg",        # Primary image (crop)
    original_image_path="original.jpg"  # **NEW**: Original image for context
)

# Output: Structured result
{
    "success": True,
    "error": None,
    "response": "Raw response text",
    "parsed": {                   # Pydantic model instance
        "total_crops": 1,
        "crop_analyses": [...]
    },
    "metadata": {
        "model": "gemini-2.5-flash",
        "image_included": True,
        "original_image_included": True,  # **NEW**: Original image metadata
        "structured_output": True,
        "response_time_ms": 1250.5,
        "input_tokens": 1500,
        "output_tokens": 800,
        "total_tokens": 2300,
        "parse_status": "sdk_parsed"  # sdk_parsed/manual_parsed/fallback
    }
}
```

#### **Key Parameters** (config.yaml):
```yaml
gemini:
  api_key: "your_api_key_here"
  model_name: "gemini-2.5-flash"
  cache_responses: true
  cache_dir: "gemini_cache"
  max_retries: 5
  retry_delay: 2.0
  enable_structured_output: true
  batch_delay: 1.2  # Delay between batch calls
```

#### **Debugging & Troubleshooting**:
```python
# Test API connection and configuration
from gemini_inferencer import create_gemini_inferencer
inferencer = create_gemini_inferencer()

# Check connection
if inferencer.test_connection():
    print("✅ API connection successful")
    
    # Test structured output
    test_prompt = "Analyze this waste object: a clear plastic bottle"
    result = inferencer.infer_single(test_prompt)
    
    if result['success']:
        print(f"✅ Inference successful")
        print(f"Parse status: {result['metadata']['parse_status']}")
        print(f"Response length: {len(result['response'])}")
        if result.get('parsed'):
            print(f"✅ Structured parsing successful: {type(result['parsed']).__name__}")
        else:
            print("⚠️ No structured data parsed")
    else:
        print(f"❌ Inference failed: {result['error']}")
else:
    print("❌ API connection failed - check API key and network")

# Check cache status
stats = inferencer.get_stats()
print(f"Cache hits: {stats['cache_hits']}/{stats['total_requests']}")
```

#### **Error Handling & Fallback Mechanisms**:
```python
# 1. Structured Output Parsing Hierarchy
if hasattr(response, 'parsed') and response.parsed is not None:
    # SDK direct parsing (best case)
    result["parsed"] = response.parsed
    result["metadata"]["parse_status"] = "sdk_parsed"
elif response_text:
    # Manual JSON parsing with repair
    try:
        parsed_data = json.loads(cleaned_text)
        result["parsed"] = parsed_data
        result["metadata"]["parse_status"] = "manual_parsed"
    except:
        # Fallback structured data
        result["parsed"] = fallback_structured_data
        result["metadata"]["parse_status"] = "fallback"
        result["metadata"]["raw_preview"] = response_text[:300]

# 2. TXT Output Fallback Strategy
if parse_status in ["manual_parsed", "fallback"]:
    # Use raw Gemini text for human-readable output
    txt_content = raw_response_text
else:
    # Use structured data for formatted output
    txt_content = format_structured_data(parsed_data)
```

#### **Common Issues & Solutions**:
1. **API Key Issues**: Set `GEMINI_API_KEY` environment variable or pass `api_key` parameter
2. **Parse Failures**: Check `parse_status` in metadata, examine raw response text
3. **Rate Limiting**: Increase `batch_delay`, enable caching, check API quotas
4. **Large Images**: Verify Files API usage for images >20MB
5. **Schema Mismatch**: Ensure correct schema selection for single vs batch processing
6. **JSON Truncation**: Increase `max_output_tokens` to 4096, reduce prompt length
7. **Cache Issues**: Clear cache with `inferencer.clear_cache()`, check disk space

**Run Test**:
```bash
python -c "
from gemini_inferencer import create_gemini_inferencer
inferencer = create_gemini_inferencer()
print('Connection test:', inferencer.test_connection())
"
```

### 6.3.3) waste_analysis_schema.py - Structured Output Schemas
**Function**: Pydantic data models for waste analysis, fully aligned with Gemini prompt output format
**Key Classes**: `SingleCropAnalysis`, `BatchCropAnalysis`, `ConfidenceGrade`

#### **Schema Architecture**:
```python
# Single Crop Analysis Structure
SingleCropAnalysis:
  ├── region_id: int
  ├── crop_path: str
  ├── primary_object_analysis: PrimaryObjectAnalysis
  │   ├── material_composition: str
  │   ├── shape: Optional[str]
  │   ├── size: Optional[str]
  │   ├── color: Optional[str]
  │   ├── condition: Optional[str]
  │   ├── likely_function_or_original_purpose: Optional[str]
  │   ├── current_state: Optional[str]
  │   └── completeness_and_visibility_in_crop: Optional[str]
  ├── secondary_elements: SecondaryElements
  │   ├── other_visible_objects_or_portions: List[str]
  │   ├── relationship_to_primary_object: Optional[str]
  │   └── potential_impact_on_classification: Optional[str]
  └── overall_assessment: OverallAssessment
      ├── primary_object_category_determination: str
      ├── recyclability_assessment_if_possible: Optional[str]
      ├── confidence_level_in_assessment: ConfidenceGrade
      ├── ambiguities_or_uncertainties_observed: List[str]
      └── alternative_possibilities_if_unclear: List[str]

# Batch Analysis Structure
BatchCropAnalysis:
  ├── total_crops: int
  ├── crop_analyses: List[SingleCropAnalysis]
  └── batch_summary: Optional[str]
```

#### **Confidence Grade System**:
```python
class ConfidenceGrade(str, Enum):
    A = "A"  # High: Clear features, high confidence
    B = "B"  # Medium: Generally clear features, moderate confidence
    C = "C"  # Low: Unclear features, low confidence
    D = "D"  # Unreliable: Very poor quality/unclear, very low confidence
```

#### **Schema Factory Functions**:
```python
# Factory functions for schema creation
single_schema = create_single_crop_schema()    # Returns SingleCropAnalysis
batch_schema = create_batch_crop_schema()      # Returns BatchCropAnalysis
compat_schema = create_waste_analysis_schema() # Returns WasteObjectAnalysis (legacy)
```

#### **Format Alignment**:
- **PRIMARY OBJECT ANALYSIS**: Material, shape, size, color, condition, purpose, state, visibility
- **SECONDARY ELEMENTS**: Other objects, relationships, classification impact
- **OVERALL ASSESSMENT**: Category determination, recyclability, confidence, uncertainties

**Run Test**:
```bash
python waste_analysis_schema.py  # Built-in test with example data
```

### 6.3.4) gemini_cache/ - Response Caching
**Function**: Persistent storage for Gemini API responses to avoid redundant calls
**Structure**: Pickle files with cache keys based on content hash

#### **Cache Key Generation**:
```python
# Cache key includes prompt + image file metadata
content = prompt
if image_path:
    stat = os.stat(image_path)
    content += f"|{image_path}|{stat.st_mtime}|{stat.st_size}"
cache_key = hashlib.md5(content.encode()).hexdigest()
```

#### **Cache File Structure**:
```
gemini_cache/
├── 030ddea01d7b75f2be1a11de7ab7883f.pkl  # Cache file 1
├── 0b08500ca57ce0b9d74dd0ae55edf19b.pkl  # Cache file 2
└── 83ce656f02d493208405c66d7e772b67.pkl  # Cache file 3
```

#### **Cache Management**:
```python
# Clear cache
inferencer.clear_cache()

# Cache statistics
stats = inferencer.get_stats()
print(f"Cache hits: {stats['cache_hits']}")
print(f"Total requests: {stats['total_requests']}")
```

### 6.3.5) Gemini Pipeline Integration

#### **Detailed Data Flow**:
```
Input: Crop Images + CLIP Results
    ↓
┌─────────────────────────────────────────┐
│ 1. Prompt Building                      │
│ Input: CLIP results + configuration     │
│ ↓                                       │
│ Process:                               │
│ - Auto-detect single vs batch mode     │
│ - Include CLIP statistics context      │
│ - Apply strategy-based templates       │
│ - Add label system reference           │
│ - Apply confidence criteria            │
│ ↓                                       │
│ Output: Comprehensive prompt text       │
│ "You are an AI assistant specialized... │
│ CROP ANALYSIS APPROACH:                │
│ • Primary Object: ...                  │
│ • Secondary Elements: ...              │
│ ..."                                   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. Schema Selection                     │
│ Input: Batch size                       │
│ ↓                                       │
│ Process:                               │
│ - Single crop → SingleCropAnalysis     │
│ - Multiple crops → BatchCropAnalysis   │
│ - Update inference schema              │
│ ↓                                       │
│ Output: Configured schema              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. Gemini API Inference                 │
│ Input: Prompt + image + schema          │
│ ↓                                       │
│ Process:                               │
│ - Check cache first                    │
│ - Prepare image (Files API if >20MB)   │
│ - Select inference mode based on config │
│   • Chunked-Batch: batch_chunk_size > 1 │
│   • Concurrent: max_workers > 1        │
│   • Single Batch: default fallback     │
│ - Call Gemini API with retries         │
│ - Parse structured response            │
│ - Cache result                         │
│ ↓                                       │
│ Output: Structured analysis result      │
│ {                                       │
│   "success": true,                     │
│   "parsed": SingleCropAnalysis(...),   │
│   "metadata": {...}                    │
│ }                                       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. Result Processing                    │
│ Input: Structured results               │
│ ↓                                       │
│ Process:                               │
│ - Extract parsed data                   │
│ - Handle parse failures                 │
│ - Generate fallback structures          │
│ - Format for output                     │
│ ↓                                       │
│ Output: Final analysis results          │
└─────────────────────────────────────────┘
```

#### **Config Integration**:
```yaml
gemini:
  api_key: "your_api_key_here"
  model_name: "gemini-2.5-flash"
  cache_responses: true
  cache_dir: "gemini_cache"
  max_retries: 5
  retry_delay: 2.0
  enable_structured_output: true
  include_label_context: true
  uncertainty_focus: true
  batch_delay: 1.2

# Inference mode control (in output section)
output:
  batch_chunk_size: 3        # Chunked-Batch mode threshold
  chunk_workers: 2           # Parallel chunk processing
  concurrent_requests: 3     # Concurrent mode threshold
```

#### **Mode Selection Examples**:
```yaml
# Example 1: Single Batch Mode (default)
output:
  batch_chunk_size: 0
  concurrent_requests: 1
# Result: Uses infer_batch() for all requests

# Example 2: Concurrent Mode
output:
  batch_chunk_size: 0
  concurrent_requests: 3
# Result: Uses infer_single() with ThreadPoolExecutor

# Example 3: Chunked-Batch Mode
output:
  batch_chunk_size: 3
  chunk_workers: 2
# Result: Splits requests into chunks of 3, processes in parallel

# Example 4: Sequential Chunked-Batch
output:
  batch_chunk_size: 3
  chunk_workers: 1
# Result: Splits requests into chunks of 3, processes sequentially
```

#### **Gemini Usage in Adapter Pipeline**:
```python
# In adapter.py - Actual Gemini integration workflow
def process_image(self, image_path: str, scan_id: int):
    # 1. Initialize Gemini components (lines 96-117)
    self.prompt_builder = create_prompt_builder(label_config_path)
    self.gemini = create_gemini_inferencer(
        api_key=api_key,
        model_name="gemini-2.5-flash",  # Current config uses gemini-2.5-flash
        enable_structured_output=True,
        auto_schema_selection=True
    )
    
    # 2. Build prompts per crop (lines 397-404)
    for crop, pred in zip(crops, clip_results):
        prompt = self.prompt_builder.build_unified_prompt(
            [pred],                           # Single crop wrapped in list
            include_statistics=True,          # Uses CLIP strategy templates
            include_label_context=self._gemini_include_label_context,  # From config
            uncertainty_focus=self._gemini_uncertainty_focus          # From config
        )
        requests.append({"prompt": prompt, "image_path": crop.get("crop_path")})
    
    # 3. Schema selection and inference (lines 410-465)
    schema = self.gemini.select_schema_for_batch(len(requests))
    self.gemini.update_schema_for_inference(schema)
    
    # 4. Batch inference with error handling (lines 410-465)
    gemini_responses = self.gemini.infer_batch(requests, batch_delay=self._gemini_batch_delay)
    
    # 5. Extract and save results (lines 472-552)
    for resp in gemini_responses:
        parsed = resp.get("parsed")  # Pydantic model instance
        parse_status = resp.get("metadata", {}).get("parse_status")
        # Save to JSON/TXT files with fallback handling
```

#### **Quick Reference - Gemini Pipeline**:
```bash
# Test prompt builder
python -c "from gemini_prompt_builder import create_prompt_builder; b=create_prompt_builder(); print('OK')"

# Test inferencer
python -c "from gemini_inferencer import create_gemini_inferencer; i=create_gemini_inferencer(); print(i.test_connection())"

# Test schemas
python waste_analysis_schema.py

# Check cache
ls gemini_cache/

# Test full pipeline
python adapter.py /path/to/image.jpg --scan-id 12345
```

**Key Files Generated**:
- `gemini_cache/*.pkl` - Cached API responses
- `outputs/scan_*_result.json` - Structured analysis results
- `outputs/scan_*_result.txt` - Human-readable analysis
- `outputs/scan_*_gemini_metrics.json` - API usage statistics

**Future Enhancement Opportunities**:
- **Background Context Handling**: Current prompts focus on waste objects within crops. Future versions could include explicit instructions for handling non-waste environmental elements (roads, buildings, infrastructure, natural backgrounds) to improve analysis accuracy and reduce false positives.
- **Context-Aware Filtering**: Add dedicated prompt sections for distinguishing between actual waste objects and background environmental elements.
- **Scene Classification**: Integrate broader scene understanding to provide better context for waste object identification.

## 6.4) Integration Modules (Detailed)

### 6.4.1) adapter.py - Pipeline Orchestrator
**Function**: Main pipeline orchestrator that coordinates all modules and provides unified interface
**Key Class**: `PipelineAdapter`

#### **Core Features**:
- **Unified Interface**: Single entrypoint `process_image(image_path, scan_id)`
- **Module Coordination**: Manages segmentation, CLIP, prompt building, and Gemini inference
- **Output Management**: Saves all artifacts and returns absolute file paths
- **Error Handling**: Comprehensive error handling with fallback mechanisms
- **Configuration Integration**: Reads all settings from `config.yaml`

#### **Prerequisites & Environment**:
```bash
# Required environment variables
export GEMINI_API_KEY="your_api_key_here"

# Or set in config.yaml
gemini:
  api_key: "your_api_key_here"

# Check dependencies
python -c "import torch, cv2, PIL; print('Dependencies OK')"
python -c "from google import genai; print('Gemini SDK OK')"
```

#### **Input/Output**:
```python
# Input
result = adapter.process_image(
    image_path="/path/to/image.jpg",  # Absolute path to input image
    scan_id=12345,                    # Unique scan identifier
    output_dir="/custom/output/path"  # Optional: custom output directory (default: pipeline/outputs)
)

# Note: output_dir parameter controls where result files are saved:
# - If specified: saves to output_dir/scan_<id>_result.*
# - If None: saves to pipeline/outputs/scan_<id>_result.*
# - Crops are always saved to crops/scan_<id>/ regardless of output_dir

# Output
{
    "success": True,
    "scan_id": 12345,
    "num_crops": 3,
    "outputs": [
        {
            "region_id": 0,
            "crop_path": "/abs/path/to/crop_001.jpg",
            "gemini_text": "Raw Gemini response...",
            "gemini_parsed": SingleCropAnalysis(...)
        }
    ],
    "files": {                        # Absolute paths to all generated files
        "result_json": "/abs/path/to/scan_12345_result.json",
        "result_txt": "/abs/path/to/scan_12345_result.txt",
        "gemini_metrics": "/abs/path/to/scan_12345_gemini_metrics.json",
        "crop_visualization": "/abs/path/to/scan_12345_crops_visualization.jpg",
        "crops_dir": "/abs/path/to/crops/scan_12345/",
        "original_image": "/abs/path/to/image.jpg"
    }
}
```

#### **Pipeline Workflow**:
```python
def process_image(self, image_path: str, scan_id: int):
    # 1. Multi-scale Segmentation (lines 343-348)
    scales = [0.5, 1.0, 1.5]  # From config
    masks = self.advanced_segmenter.multi_scale_segmentation(image_rgb, scales)
    
    # 2. High-resolution Cropping (lines 350-361)
    crops = crop_objects(image_rgb, masks, per_scan_crop_dir, ...)
    
    # 3. CLIP Classification (lines 371-378)
    clip_results = self.clip_matcher.classify_batch(crops, batch_size=32)
    
    # 4. Prompt Building (lines 394-408)
    for crop, pred in zip(crops, clip_results):
        prompt = self.prompt_builder.build_unified_prompt([pred], ...)
        requests.append({"prompt": prompt, "image_path": crop.get("crop_path")})
    
    # 5. Gemini Inference (lines 410-465)
    # Dynamic mode selection based on config
    if batch_chunk_size > 1:
        # Chunked-Batch Mode
    elif max_workers > 1:
        # Concurrent Mode
    else:
        # Single Batch Mode
    gemini_responses = self.gemini.infer_batch(requests, ...)
    
    # 6. Result Processing & Saving (lines 472-570)
    # Extract structured data, handle parse failures, save files
```

#### **Key Configuration Parameters**:
```yaml
# Segmentation
segmenter:
  multi_scale_scales: [0.5, 1.0, 1.5]
  min_mask_area: 3600

# CLIP
recognizer:
  batch_size: 32
  clip_threshold: 0.25

# Gemini
gemini:
  model_name: "gemini-1.5-flash"
  batch_delay: 1.2
  include_label_context: true
  uncertainty_focus: true

# Output control
output:
  batch_chunk_size: 3        # Chunked-Batch mode
  concurrent_requests: 3     # Concurrent mode
  save_clip_results: false   # Optional CLIP cache saving
```

#### **Generated Files Structure**:
```
outputs/
├── scan_12345_result.json              # Structured analysis results
├── scan_12345_result.txt               # Human-readable analysis
├── scan_12345_gemini_metrics.json      # API usage statistics
└── scan_12345_crops_visualization.jpg  # Crop regions visualization

crops/scan_12345/
├── crop_001.jpg                        # Individual crop images
├── crop_002.jpg
└── crop_003.jpg
```

#### **Error Handling & Fallback**:
```python
# 1. Segmentation failures
if not masks:
    return {"success": False, "error": "no_masks"}

# 2. Cropping failures  
if not crops:
    return {"success": False, "error": "no_crops"}

# 3. Gemini parse failures
if parse_status in ["manual_parsed", "fallback"]:
    # Use raw Gemini text for TXT output
    txt_content = raw_response_text
else:
    # Use structured data for formatted output
    txt_content = format_structured_data(parsed_data)
```

#### **Programmatic Usage**:
```python
# Import and initialize
from adapter import create_pipeline_adapter

# Create adapter instance
adapter = create_pipeline_adapter("config.yaml")

# Process single image
result = adapter.process_image("/path/to/image.jpg", scan_id=12345)

# Check results
if result["success"]:
    print(f"Processed {result['num_crops']} crops")
    
    # Access structured analysis
    for output in result["outputs"]:
        parsed = output.get("gemini_parsed")
        if parsed:
            print(f"Crop {output['region_id']}: {parsed.overall_assessment.primary_object_category_determination}")
    
    # Access file paths
    files = result.get("files", {})
    print(f"Results saved to: {files.get('result_txt')}")
    print(f"Crops saved to: {files.get('crops_dir')}")
else:
    print(f"Processing failed: {result.get('error')}")

# Batch processing
import os
image_dir = "/path/to/images"
for i, filename in enumerate(os.listdir(image_dir)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, filename)
        result = adapter.process_image(image_path, scan_id=1000 + i, output_dir="/batch/outputs")
        print(f"{filename}: {result['success']}")
```

**Run Test**:
```bash
# Single image processing
python adapter.py /path/to/image.jpg --scan-id 12345 --config config.yaml

# Check outputs
ls outputs/scan_12345_*
ls crops/scan_12345/
```

### 6.4.2) batch_test_adapter.py - Batch Testing Utility
**Function**: Automated batch testing tool for evaluating pipeline performance across multiple images
**Key Features**: CSV/JSON summaries, performance metrics, artifact organization

#### **Core Features**:
- **Batch Processing**: Process multiple images or entire directories
- **Performance Metrics**: Detailed timing and API usage statistics
- **Artifact Organization**: Consolidates all outputs into organized test runs
- **Mode Detection**: Automatically detects and reports inference mode
- **Flexible Input**: Supports single images or directories

#### **Usage Examples**:
```bash
# Process single image
python batch_test_adapter.py /path/to/image.jpg --scan-base 1000

# Process directory with limit
python batch_test_adapter.py /data/images --limit 5 --suffix test_run

# Full directory processing
python batch_test_adapter.py /data/images --config config.yaml --sleep 1.0

# Include CLIP cache files
python batch_test_adapter.py /data/images --include-clip-cache --suffix debug
```

#### **Command Line Arguments**:
```bash
positional arguments:
  input                 Input directory or single image file

optional arguments:
  --config CONFIG       Optional config.yaml path
  --scan-base BASE      Base scan id (default: current timestamp)
  --limit LIMIT         Limit number of images (0 = all)
  --sleep SLEEP         Sleep seconds between images
  --suffix SUFFIX       Suffix tag for summary filenames
  --outdir OUTDIR       Directory to store test outputs
  --include-clip-cache  Copy related CLIP cache files per scan
```

#### **Output Organization**:
```
outputs_tests/run_20241201_143022/
├── batch_summary.csv                   # CSV summary with metrics
├── batch_summary.json                  # JSON summary with details
└── scan_1001/                          # Per-scan organized outputs
    ├── scan_1001_result.json
    ├── scan_1001_result.txt
    ├── scan_1001_gemini_metrics.json
    ├── scan_1001_crops_visualization.jpg
    ├── crops/                           # Crop images
    │   ├── crop_001.jpg
    │   └── crop_002.jpg
    └── clip_results_cache/             # CLIP cache (if --include-clip-cache)
        └── scan_1001_clip_results.json
```

#### **CSV Summary Fields (updated)**
```csv
idx,scan_id,image,success,num_crops,elapsed_sec,mode,json_exists,txt_exists,
# Per-image Gemini metrics (preferred)
gm_api_calls_image,gm_input_tokens_image,gm_output_tokens_image,gm_total_tokens_image,gm_avg_response_time_ms_image,
gm_errors_image,gm_structured_parse_success_image,gm_structured_parse_failures_image,
# Cumulative Gemini metrics (session-wide, for reference)
gm_total_input_tokens,gm_total_output_tokens,gm_total_tokens,gm_avg_response_time_ms,
gm_api_calls,gm_cache_hits,gm_errors,gm_structured_parse_success,gm_structured_parse_failures,
# Phase timings (seconds)
segmentation_sec,crop_generation_sec,clip_classification_sec,prompt_building_sec,gemini_inference_sec,
error
```

Notes:
- Per-image metrics are computed by differencing Gemini stats before/after each image.
- `gm_avg_response_time_ms_image` is averaged over API calls for that image; display in seconds (s) in UIs.

#### **Mode Detection**
```python
# Adapter logic (simplified)
if batch_chunk_size and batch_chunk_size > 1:
    mode = f"chunked(size={batch_chunk_size}, workers={chunk_workers})"  # infer_batch per chunk; optional parallel chunks
else:
    # per-crop path; if concurrent_requests>1 uses ThreadPoolExecutor over infer_single
    mode = f"per-crop(concurrency={concurrent_requests})"
```

Examples:
- `concurrent_requests: 6, batch_chunk_size: 4, chunk_workers: 2` → mode: `chunked(size=4, workers=2)` (Parallel Chunked-Batch Mode)
- `concurrent_requests: 4, batch_chunk_size: 0` → mode: `per-crop(concurrency=4)` (Concurrent Per-Crop Mode)

#### **Performance Monitoring (per-image metrics)**
```python
{
    "success": True,
    "num_crops": 19,
    "elapsed_sec": 118.636,
    "mode": "chunked(size=4, workers=2)",
    # Per-image metrics
    "gm_api_calls_image": 19,
    "gm_input_tokens_image": 29908,
    "gm_output_tokens_image": 9933,
    "gm_total_tokens_image": 52795,
    "gm_avg_response_time_ms_image": 9732.765,  # display as 9.733s
    # Phase timings (s)
    "segmentation_sec": 3.939,
    "crop_generation_sec": 0.177,
    "clip_classification_sec": 1.483,
    "prompt_building_sec": 0.001,
    "gemini_inference_sec": 112.800
}
```

Display conventions:
- Show times in seconds (s). Convert `gm_avg_response_time_ms_image` to seconds in frontends.
- Tokens are counts; API calls are integers; mode is a string.

#### **Metrics definitions**
- **elapsed_sec (s)**: Total pipeline time per image (end-to-end).
- **segmentation_sec, crop_generation_sec, clip_classification_sec, prompt_building_sec, gemini_inference_sec (s)**: Phase-wise times per image.
- **gm_api_calls_image**: Number of Gemini API calls for this image (equals number of crop requests in per-crop mode; equals number of requests inside all chunks in chunked mode).
- **gm_input_tokens_image / gm_output_tokens_image / gm_total_tokens_image**: Token usage attributable to this image only.
- **gm_avg_response_time_ms_image**: Average Gemini API latency per request for this image (ms). For display, divide by 1000 to show in seconds.
- Cumulative fields (`gm_total_*`, `gm_api_calls`, etc.) are session totals and provided for context only.

#### **Current optimal configuration (v2)**
```yaml
output:
  concurrent_requests: 6
  batch_chunk_size: 4
  chunk_workers: 2
gemini:
  batch_delay: 1.2
recognizer:
  batch_size: 16   # CLIP batch size on CPU
```
Rationale:
- Parallel chunk processing (`chunk_workers: 2`) significantly reduces wall time versus sequential chunking.
- `batch_chunk_size: 4` balances Gemini throughput and response parsing stability.
- `concurrent_requests` is used only if chunking is disabled; with chunking enabled, throughput comes from `chunk_workers`.

#### **CLIP batch_size effectiveness**
- `recognizer.batch_size` controls CLIP embedding batch size in `classify_batch()`; if crop count ≤ batch_size, processed in one CLIP pass; if > batch_size, processed in multiple passes of size `batch_size`.
- On CPU, `batch_size: 16` is a safe upper bound; larger values may increase memory pressure with diminishing returns.

#### **Error Handling**:
```python
# Graceful error handling with detailed logging
try:
    result = adapter.process_image(img_path, scan_id)
    # Process successful result
except Exception as e:
    # Log error details and continue with next image
    print(f"[{idx}/{len(all_files)}] {name} -> FAILED: {e}")
    # Add error to summary
```

**Run Test**:
```bash
# Quick smoke test
python batch_test_adapter.py /data/images --limit 2 --suffix smoke

# Full test run
python batch_test_adapter.py /data/images --config config.yaml --sleep 0.5

# Debug mode with CLIP cache
python batch_test_adapter.py /data/images --include-clip-cache --suffix debug
```

### 6.4.3) Integration Pipeline Summary

#### **Complete Data Flow**:
```
Input Image → Segmentation → Crops → CLIP → Prompts → Gemini → Results → Files
     ↓              ↓         ↓      ↓       ↓        ↓        ↓       ↓
  image.jpg    masks[]    crops[]  preds[]  prompts  responses  JSON   outputs/
     ↓              ↓         ↓      ↓       ↓        ↓        ↓       ↓
  scan_id      multi-scale  high-res  batch   unified  structured  TXT   crops/
     ↓              ↓         ↓      ↓       ↓        ↓        ↓       ↓
  unique_id    [0.5,1.0,1.5]  per-scan 32     strategy  A/B/C/D   vis   scan_*/
```

#### **Key Integration Points**:
1. **Configuration**: All modules read from unified `config.yaml`
2. **Error Propagation**: Failures at any stage are handled gracefully
3. **File Management**: Consistent naming and organization across modules
4. **Mode Selection**: Dynamic inference mode based on configuration
5. **Caching**: CLIP and Gemini responses are cached for efficiency

#### **Performance Characteristics**:
- **Single Image**: 2-5 seconds (depending on crop count and mode)
- **Batch Processing**: Linear scaling with configurable concurrency
- **Memory Usage**: Moderate (crops and embeddings cached)
- **API Efficiency**: Caching reduces redundant calls

#### **Debugging Workflow**:
1. **Single Image Test**: `python adapter.py image.jpg --scan-id 12345`
2. **Check Outputs**: Inspect `outputs/scan_12345_*` files
3. **Batch Test**: `python batch_test_adapter.py /data --limit 3`
4. **Analyze Metrics**: Review CSV summary for performance patterns
5. **Tune Configuration**: Adjust batch sizes, delays, and concurrency

Data flow:
image → segmenter → refined masks → crops → CLIP → prompt → Gemini → save outputs (JSON/TXT/visualization/metrics) → return paths

#### **Common Issues & Quick Fixes**:
1. **No masks found**: Increase `min_mask_area`, adjust `multi_scale_scales`, check image quality
2. **Poor segmentation**: Try different scales `[0.3, 0.8, 1.2]`, increase `max_objects`
3. **CLIP classification issues**: Verify `waste_labels_template.json` exists, check `clip_threshold`
4. **Gemini parse failures**: Check `parse_status` in metrics, increase `max_output_tokens`
5. **API rate limits**: Increase `batch_delay`, reduce `concurrent_requests`, enable caching
6. **Memory issues**: Reduce `batch_chunk_size`, clear caches, process fewer crops

#### **Performance Tuning Guide**:
```yaml
# For fast processing (3-6 crops)
output:
  batch_chunk_size: 0
  concurrent_requests: 3
  gemini:
    batch_delay: 0.5

# For large batches (10+ crops)  
output:
  batch_chunk_size: 4
  chunk_workers: 2
  gemini:
    batch_delay: 1.5

# For maximum reliability
output:
  batch_chunk_size: 0
  concurrent_requests: 1
  gemini:
    batch_delay: 2.0
```

## 7) Testing & Latest Improvements

### 7.1) Quick Tests
Gemini connectivity:
```bash
python -c "from gemini_inferencer import create_gemini_inferencer; g=create_gemini_inferencer(); print(g.test_connection())"
```
Generate labels:
```bash
python generate_label_template.py
```

Single image test:
```bash
python adapter.py test_super_u_images/Super_U_20250624_085602.jpg --scan-id 8888 --config config.yaml
```

### 7.2) Comprehensive Testing (7 Super_U Images)

**Complete Test Results**:
```bash
# Complete 7-image test with improved fusion algorithm
python batch_test_adapter.py test_super_u_images/ --suffix "super_u_final_improved" --sleep 1.0
```

**Key Improvements Validated**:
- ✅ **IoU-based Mask Fusion**: Average 60% reduction in redundant masks
- ✅ **Containment Removal**: Eliminated crops fully contained within others
- ✅ **Relaxed CLIP Rules**: Reduced false "unknown" classifications
- ✅ **Original Image Context**: Gemini receives both crop and scene context
- ✅ **Prompt Saving**: All prompts saved for debugging analysis


**Output Structure** (per image):
```
outputs_tests/run_super_u_final_improved/
├── scan_XXXXXXX/
│   ├── scan_XXXXXXX_result.json         # Structured analysis
│   ├── scan_XXXXXXX_result.txt          # Human-readable results
│   ├── scan_XXXXXXX_gemini_metrics.json # API usage stats
│   ├── scan_XXXXXXX_crops_visualization.jpg # Visual overview
│   └── crops/                           # Individual crop images
│       ├── scan_XXXXXXX_crop000.jpg
│       └── ...
└── batch_summary_super_u_final_improved.csv # Complete test summary
```

### 7.3) Batch Testing Examples

Smoke test (2 images):
```bash
python batch_test_adapter.py /data/images --config ./config.yaml --limit 2 --suffix smoke
```

Full directory test:
```bash
python batch_test_adapter.py test_super_u_images/ --config config.yaml --sleep 1.0
```

Debug mode with prompt saving:
```bash
# Enable prompt saving in config.yaml first:
# gemini:
#   save_prompts: true
#   prompt_save_dir: "prompts"
python batch_test_adapter.py test_super_u_images/ --limit 3 --suffix debug
```

Performance optimization test:
```bash
# Test with different batch configurations
python batch_test_adapter.py test_super_u_images/ --limit 5 --suffix chunked_batch
```

### 7.4) Recent Improvements Summary

- **IoU-based segmentation fusion**: Reduced redundant crops by ~60%
- **Optimized CLIP classification**: Fewer false "unknown" results  
- **Original image context**: Gemini receives scene context for improved analysis
- **Configurable post-processing**: `use_postprocess` control for performance

### 7.6) Visualization (Reports)

Two report generators are available under `visualization/`:

- `fixed_layout_batch_report_generator.py` (recommended):
  - Stable table (rows do not expand); details show below table in a two-column layout.
  - Left: original image, crops visualization, per-image stats, phase timings.
  - Right: scrollable analysis text.
  - Time displayed in seconds; "平均API响应时间" uses per-image metric (`gm_avg_response_time_ms_image / 1000`).
  - Usage:
    ```bash
    python visualization/fixed_layout_batch_report_generator.py outputs_tests/<run>/batch_summary_*.csv
    ```

- `interactive_batch_report_generator.py` (compact, expandable rows):
  - Click-to-expand row with per-image details inside the table.
  - Same metric display rules as above.

### 7.5) Configuration for Optimal Performance

**For VM Deployment** (recommended):
```yaml
output:
  batch_chunk_size: 3      # Chunked-batch mode for stability
  chunk_workers: 1         # Sequential processing
  
gemini:
  batch_delay: 1.2         # Stable API timing
  max_output_tokens: 4096  # Prevent JSON truncation
  
segmenter:
  use_postprocess: false   # Better performance
  multi_scale_scales: [0.5, 1.0, 1.5]  # Balanced detection
```

## 8) Troubleshooting

### 8.1) Common Issues & Solutions

**Segmentation Issues**:
- **No masks found**: Reduce `min_mask_area` from 3600 to 1000, check image quality
- **Too many small masks**: Increase `min_mask_area`, enable `use_postprocess: true`
- **Missing objects**: Add smaller scales like `[0.3, 0.5, 1.0, 1.5]`
- **Over-segmentation**: Current fusion algorithm should handle this automatically

**CLIP Classification Issues**:
- **All unknown**: Check `clip_threshold` (default 0.1), verify `waste_labels_template.json` exists
- **False unknowns**: Rules have been relaxed, but adjust thresholds in `clip_matcher.py` if needed
- **Poor performance**: Clear `clip_embeddings_cache.pkl` and regenerate

**Gemini API Issues**:
- **API key errors**: Set `GEMINI_API_KEY` environment variable or in `config.yaml`
- **Parse failures**: Check `parse_status` in `scan_*_gemini_metrics.json`
- **JSON truncation**: Increase `max_output_tokens` to 4096, use TXT fallback
- **Rate limiting**: Increase `batch_delay` to 1.5s, reduce `concurrent_requests`

**Performance Issues**:
- **Slow processing**: Disable `use_postprocess`, reduce `multi_scale_scales`
- **Memory issues**: Reduce `batch_chunk_size`, clear caches
- **API quota**: Enable caching, reduce batch sizes

### 8.2) Debug Commands

Check configuration:
```bash
python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['segmenter'])"
```

Test individual components:
```bash
# Test segmentation only
python -c "from segmenter import Segmenter; s=Segmenter('fastsam','models/FastSAM-s.pt','cpu'); print('OK')"

# Test CLIP
python -c "from clip_matcher import CLIPMatcher; m=CLIPMatcher('waste_labels_template.json','cpu'); print('OK')"

# Test Gemini
python -c "from gemini_inferencer import create_gemini_inferencer; g=create_gemini_inferencer(); print(g.test_connection())"
```

Clear all caches:
```bash
rm -rf gemini_cache/* clip_results_cache/* crops/* outputs/*
```

## 9) VM Deployment Guide

### 9.1) Upload Bundle to VM
```bash
# Option 1: scp (initial upload)
scp -i ~/.ssh/<key> -r upload_vm_bundle <user>@<host>:~/ImageAPI/

# Option 2: rsync (incremental updates - faster)
rsync -avz -e "ssh -i ~/.ssh/<key>" upload_vm_bundle/ <user>@<host>:~/ImageAPI/pipeline/
```

### 9.2) VM Environment Setup
```bash
# Connect to VM
ssh -i ~/.ssh/<key> <user>@<host>

# Navigate to pipeline directory
cd ~/ImageAPI/pipeline

# Create/activate virtual environment
python3 -m venv ~/wcr_env
source ~/wcr_env/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Set API key (required)
export GEMINI_API_KEY="<YOUR_API_KEY>"
```

### 9.3) VM Testing Commands
```bash
# Test individual components
python -c "from gemini_inferencer import create_gemini_inferencer; print(create_gemini_inferencer().test_connection())"

# Single image test
python adapter.py test_super_u_images/Super_U_20250624_085602.jpg --scan-id 9999 --config config.yaml

# Batch test (3 images)
python batch_test_adapter.py test_super_u_images/ --limit 3 --suffix vm_test

# Check outputs
ls outputs_tests/run_vm_test/
```

### 9.4) VM Production Integration
```python
# For integration with app.py or other services
from adapter import create_pipeline_adapter

# Initialize once
adapter = create_pipeline_adapter("config.yaml")

# Process images (with optional custom output directory)
result = adapter.process_image("/path/to/image.jpg", scan_id=12345, output_dir="/custom/outputs")

# Use returned file paths
files = result.get("files", {})
json_path = files.get("result_json")
txt_path = files.get("result_txt")
vis_path = files.get("crop_visualization")
```

### 9.5) VM Maintenance
```bash
# Clear caches periodically
rm -rf gemini_cache/* clip_results_cache/* crops/* outputs/*

# Update pipeline code
rsync -avz -e "ssh -i ~/.ssh/<key>" upload_vm_bundle/ <user>@<host>:~/ImageAPI/pipeline/

# Monitor disk usage
df -h
du -sh gemini_cache/ clip_results_cache/ crops/ outputs/
```

## 10) License
Follows the licenses of integrated SDKs and models.
