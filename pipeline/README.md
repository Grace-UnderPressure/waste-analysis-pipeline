# Waste Analysis Pipeline — VM Developer Guide (upload_vm_bundle)

Production-ready bundle for VM deployment:
- **Multi-scale segmentation** → automated crop generation → CLIP classification → Gemini inference
- **Original image context** for Gemini analysis
- **Robust fallback mechanisms** with parse_status/raw_preview
- **60% reduction** in redundant crops through IoU-based mask fusion

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Installation & Setup](#2-installation--setup)
3. [Models and Data](#3-models-and-data)
4. [Quickstart](#4-quickstart)
5. [Config (config.yaml)](#5-config-configyaml)
6. [Tips](#6-tips)
7. [Core Modules](#7-core-modules)
   - 7.1 [Segmentation](#71-segmentation-modules)
   - 7.2 [CLIP Classification](#72-clip-modules)
   - 7.3 [Gemini Inference](#73-gemini-modules)
   - 7.4 [Integration](#74-integration-modules)
8. [Configuration Reference](#8-configuration-reference)
9. [Usage Examples](#9-usage-examples)
10. [Testing & Performance](#10-testing--performance)
11. [Troubleshooting](#11-troubleshooting)
12. [VM Deployment](#12-vm-deployment)
13. [License](#13-license)

## 1) Quick Start
- Python 3.10+
- GPU optional (CUDA recommended)
- Google Gemini API Key

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

## 2) Installation & Setup

### Prerequisites
- Python 3.10+
- GPU optional (CUDA recommended)
- Google Gemini API Key

### Installation
```bash
# Install dependencies
python -m pip install -r requirements.txt

# Set environment variable
export GEMINI_API_KEY="<YOUR_KEY>"
```

### Key Files
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
- `gemini`: `API_key` (or ENV), `model_name`, structured output flags, prompt saving options
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

## 7) Core Modules

### 7.1) Segmentation Modules

#### 7.1.1) segmenter.py - Core Segmentation Engine
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

#### 7.1.2) mask_postprocess.py - Mask Refinement Pipeline
**Function**: Multi-stage mask post-processing and filtering
**Key Function**: `mask_postprocess()`

**Integration**: Called automatically in `adapter.py` via multi-scale segmentation. Can be controlled via config (`use_postprocess: false` for better performance).

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

**Key Parameters**:
- `min_area`: From config `min_mask_area` (default: 3600)
- `max_masks`: From config `segmenter.max_objects` (default: 20)
- Other parameters are optimized and hardcoded for best performance

**Debug Output**: Each step prints mask count for monitoring

#### 7.1.3) advanced_segmentation_methods.py - Multi-scale Strategies
**Function**: Multi-scale segmentation and IoU-based fusion
**Key Class**: `AdvancedSegmenter`

**Key Methods**:

**`multi_scale_segmentation()`** - Main method used in pipeline:
- **Purpose**: Run segmentation at multiple scales and fuse results
- **Config scales**: `[0.5, 1.0, 1.5]` (50%, 100%, 150% of original size)
- **Workflow**: Scale image → Run FastSAM → Resize masks back → Collect all scales

**`_smart_mask_fusion()`** - IoU-based mask fusion:
- **Purpose**: Handle over-segmentation and overlaps
- **Strategy**: Multi-criteria overlap detection with IoU threshold 0.3
- **Improvements**: Containment removal, reduced false positives

**Integration**: Used automatically in `adapter.py` with configurable scales and post-processing options.

**Run Test**:
```bash
python advanced_segmentation_methods.py  # Has built-in test
```

#### 7.1.4) crop_generator.py - Object Cropping
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

#### 7.1.5) Segmentation Pipeline Integration
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

**Performance Tips**:
- Use `device: "cuda"` for GPU acceleration
- Adjust `input_resize` for memory constraints
- Modify scales `[0.5, 1.0, 1.5]` for speed vs quality trade-off

**Quick Test**:
```bash
python adapter.py test.jpg --scan-id 12345 --config config.yaml
```

**Output Files**:
- `outputs/scan_<id>_result.json` - Structured results
- `outputs/scan_<id>_result.txt` - Human-readable results  
- `outputs/scan_<id>_crops_visualization.jpg` - Visual overview
- `outputs/scan_<id>_gemini_metrics.json` - Performance metrics
- `crops/scan_<id>/` - Individual crop files

**Key Config Parameters**:
```yaml
segmenter:
  model_path: "models/FastSAM-s.pt"
  multi_scale_scales: [0.5, 1.0, 1.5]
  use_postprocess: false
min_mask_area: 3600
device: "cpu"
```

### 7.2) CLIP Modules

#### 7.2.0) CLIP Design Philosophy & Customization
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

#### 7.2.1) waste_labels_template.json - Label Configuration
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

#### 7.2.2) clip_matcher.py - CLIP Classification Engine
**Function**: CLIP-based image-text matching for waste categorization
**Key Class**: `CLIPMatcher`

**Core Features**:
- **Batch Processing**: Efficient batch encoding of multiple crops
- **Embedding Caching**: Text embeddings cached to disk, image embeddings in memory
- **Unknown Fallback**: Multi-criteria classification with confidence thresholds
- **Multi-Prompt Aggregation**: Combines scores from multiple prompts per category

**Input/Output**:
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

#### 7.2.3) clip_results_analyzer.py - Results Processing & Analysis
**Function**: Unified CLIP results management, statistical analysis, and strategy-based prompt generation
**Key Class**: `CLIPResultsAnalyzer`

**Three-Layer Architecture**:

1. **Data Management**: Save/load CLIP results to JSON with smart caching
2. **Statistical Analysis**: Dominance coefficient, variation coefficient, score gradient
3. **Formatting**: Strategy-based prompt generation and template selection

**Analysis Strategies**:
- **descriptive_analysis**: Very low confidence (< 0.03)
- **focus_validation**: Strong dominance (≥ 4.0)
- **dual_comparison**: Moderate competition (2.0-4.0)
- **multi_candidate**: High competition (< 2.0)

**Key Methods**:
- `analyze_and_format_for_gemini()`: One-stop analysis with caching
- Returns enhanced prompt context and statistical analysis

**Cache System**:
- **Smart Cache**: Content-aware (image hash + config hash)
- **Simple Cache**: Filename-based fallback

**Run Test**:
```bash
python clip_results_analyzer.py  # Has built-in test
```

#### 7.2.4) generate_label_template.py - Label Generator
**Function**: Generate waste classification label templates
**Output**: `waste_labels_template.json`

**Design Principles**:
1. **Objectivity**: Focus on object type, avoid subjective states
2. **Brevity**: Max 5 prompts per category to reduce cost
3. **Hierarchy**: Preserve main-sub relationship
4. **Main Class Presence**: Every prompt mentions the main class
5. **Other Handling**: Catch-all prompts for "other" subclass

**Prompt Generation Logic**:
- **Regular categories**: 5 generic, balanced prompts per main/sub combination
- **"Other" subclass**: 5 catch-all prompts for unspecified items

**Customization Examples**:
- **Medical waste**: Add context awareness (medical, sterile, clinical)
- **Size-sensitive**: Add dimensional attributes (large, small, compact)
- **Material-focused**: Emphasize material properties (transparent, flexible, rigid)

**Customization Workflow**:
1. Modify categories in `generate_label_template.py`
2. Adjust prompts for specific attributes
3. Update thresholds in config
4. Test with domain-specific data

**Run Test**:
```bash
python generate_label_template.py  # Generates waste_labels_template.json
```

#### 7.2.5) clip_results_cache/ - Results Storage
**Function**: Persistent storage for CLIP classification results
**Structure**: JSON files with metadata and results

**File Naming**: `{image_name}_{image_hash}_{config_hash}_clip_results.json`

**File Structure**: JSON files with metadata, CLIP results, and summary statistics

#### 7.2.6) CLIP Pipeline Integration

**Data Flow**:
1. **CLIPMatcher.classify_batch()**: Process crop images → raw CLIP results
2. **CLIPResultsAnalyzer.analyze_and_format()**: Statistical analysis → enhanced context
3. **Save to cache**: Store results with metadata
4. **Pass to Gemini**: Enhanced prompt context for final analysis

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

#### 7.2.7) CLIP Usage in Adapter
**Integration Workflow**:
1. **Segmentation** → Generate crops from masks
2. **CLIP Classification** → Batch process crops with configurable batch size
3. **Results Saving** → Optional cache integration for debugging
4. **Prompt Building** → Transform CLIP results into Gemini prompts

**Key Configuration**:
- **Batch Size**: `recognizer.batch_size` (default: 32, config: 16)
- **Threshold**: `clip_threshold` (default: 0.25, config: 0.1)
- **Caching**: `cache_embeddings` and `embedding_cache_path`
- **Results Cache**: `clip_results_cache_dir` for persistent storage

**Data Flow**:
- **Input**: Crop images from segmentation pipeline
- **Processing**: Batch classification with statistical analysis
- **Output**: Enhanced prompts with CLIP context for Gemini
- **Storage**: Optional JSON results for debugging and analysis

#### 7.2.8) CLIP Section Summary
**Complete CLIP Pipeline**:
1. **Label Configuration** (3.2.1): 9 main categories, 42 combinations, ~210 prompts
2. **Classification Engine** (3.2.2): ViT-B/32 model, batch processing, caching, unknown handling
3. **Results Analysis** (3.2.3): Statistical analysis, strategy selection, intelligent prompts
4. **Label Generation** (3.2.4): Template generation with customization examples
5. **Results Storage** (3.2.5): JSON persistence with metadata and summaries
6. **Pipeline Integration** (3.2.6): Complete data flow with types and structures
7. **Adapter Usage** (3.2.7): Real-world integration in main pipeline

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

### 7.3) Gemini Modules

#### 7.3.1) gemini_prompt_builder.py - Prompt Generation Engine
**Function**: Transforms CLIP outputs and crop images into comprehensive prompts for Gemini inference
**Key Class**: `GeminiPromptBuilder`

**Core Features**:
- **Unified Prompt Generation**: Single method `build_unified_prompt()` for all scenarios
- **CLIP Integration**: Incorporates CLIP statistics and classification context
- **Strategy-Based Templates**: 5 statistical template types based on CLIP analysis patterns
- **Dynamic Formatting**: Auto-detects single vs batch crop analysis
- **Label System Context**: Includes waste classification system reference
- **Confidence Grading**: A/B/C/D confidence level criteria

**Input/Output**:
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

**Prompt Structure**:
1. **System Instruction**: Analyze cropped waste objects
2. **Image Context**: Crop vs original image labeling
3. **CLIP Statistics**: Statistical analysis and template selection
4. **Label System**: Waste classification categories reference
5. **Analysis Tasks**: Primary object, secondary elements, overall assessment
6. **Confidence Criteria**: A/B/C/D grading system
7. **Output Format**: Structured response requirements

**Key Parameters** (config.yaml):
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

#### 7.3.2) gemini_inferencer.py - API Inference Engine
**Function**: Wrapper around Google Gemini API calls, supporting multimodal inference with structured output
**Key Class**: `GeminiInferencer`

**Core Features**:
- **Multimodal Input**: Text + image support with automatic file size handling
- **Structured Output**: Pydantic schema integration for consistent JSON responses
- **Multiple Inference Modes**: Single, Batch, Chunked-Batch, and Concurrent processing
- **Response Caching**: Disk-based caching to avoid redundant API calls
- **Error Handling**: Retry mechanisms with exponential backoff
- **Schema Selection**: Auto-select single-crop vs batch-crop schemas

**API Configuration**:
- **SDK**: Uses `google-genai` (new SDK)
- **Model**: `gemini-2.5-flash` or `gemini-1.5-flash`
- **Output**: Structured JSON with Pydantic schemas
- **Temperature**: 0.1 for deterministic output

**Inference Modes**:
- **Single Mode**: Process one request at a time
- **Batch Mode**: Process multiple requests sequentially
- **Concurrent Mode**: Process multiple requests in parallel
- **Chunked-Batch Mode**: Split large batches into chunks

**Usage in Adapter**:
- **Chunked-Batch Mode**: Split requests into chunks, process in parallel
- **Concurrent Mode**: Use ThreadPoolExecutor for parallel processing
- **Single Batch Mode**: Process all requests in one call

**Mode Selection**:
- **Chunked-Batch**: `batch_chunk_size > 1`
- **Concurrent**: `concurrent_requests > 1`
- **Single Batch**: Default fallback

**Performance Characteristics**:
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
  API_key: "your_API_key_here"
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
1. **API Key Issues**: Set `GEMINI_API_KEY` environment variable or pass `API_key` parameter
2. **Parse Failures**: Check `parse_status` in metadata, examine raw response text
3. **Rate Limiting**: Increase `batch_delay`, enable caching, check API quotas
4. **Large Images**: Verify Files API usage for images >20MB
5. **Schema Mismatch**: Ensure correct schema selection for single vs batch processing
6. **JSON Truncation**: Increase `max_output_tokens` to 4096, reduce prompt length
7. **Cache Issues**: Clear cache with `inferencer.clear_cache()`, check disk space

**Run Test**:
```bash
python -c "from gemini_inferencer import create_gemini_inferencer; print(create_gemini_inferencer().test_connection())"
```

#### 7.3.3) waste_analysis_schema.py - Structured Output Schemas
**Function**: Pydantic data models for waste analysis, fully aligned with Gemini prompt output format
**Key Classes**: `SingleCropAnalysis`, `BatchCropAnalysis`, `ConfidenceGrade`

**Schema Architecture**:
- **SingleCropAnalysis**: Individual crop analysis with primary object, secondary elements, and overall assessment
- **BatchCropAnalysis**: Multiple crop analysis with batch summary
- **ConfidenceGrade**: A/B/C/D confidence levels

**Confidence Grade System**:
- **A (High)**: Clear features, high confidence
- **B (Medium)**: Generally clear features, moderate confidence
- **C (Low)**: Unclear features, low confidence
- **D (Unreliable)**: Very poor quality/unclear, very low confidence

**Run Test**:
```bash
python waste_analysis_schema.py  # Built-in test with example data
```

#### 7.3.4) gemini_cache/ - Response Caching
**Function**: Persistent storage for Gemini API responses to avoid redundant calls
**Structure**: Pickle files with cache keys based on content hash

**Cache Key Generation**: Based on prompt + image file metadata hash

**Cache Management**:
- **Clear cache**: `inferencer.clear_cache()`
- **Cache statistics**: `inferencer.get_stats()`

#### 7.3.5) Gemini Pipeline Integration

**Data Flow**:
1. **Prompt Building**: CLIP results → comprehensive prompts
2. **Schema Selection**: Auto-select single vs batch schemas
3. **Gemini API Inference**: Process with caching and retries
4. **Result Processing**: Extract and format structured results

**Config Integration**:
```yaml
gemini:
  API_key: "your_API_key_here"
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
        API_key=API_key,
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
# Test components
python -c "from gemini_prompt_builder import create_prompt_builder; print('Prompt builder OK')"
python -c "from gemini_inferencer import create_gemini_inferencer; print('Inferencer OK:', create_gemini_inferencer().test_connection())"
python waste_analysis_schema.py

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

### 7.4) Integration Modules

#### 7.4.1) adapter.py - Pipeline Orchestrator
**Function**: Main pipeline orchestrator that coordinates all modules and provides unified interface
**Key Class**: `PipelineAdapter`

**Core Features**:
- **Unified Interface**: Single entrypoint `process_image(image_path, scan_id)`
- **Module Coordination**: Manages segmentation, CLIP, prompt building, and Gemini inference
- **Output Management**: Saves all artifacts and returns absolute file paths
- **Error Handling**: Comprehensive error handling with fallback mechanisms
- **Configuration Integration**: Reads all settings from `config.yaml`

**Prerequisites & Environment**:
- **API Key**: Set `GEMINI_API_KEY` environment variable or in `config.yaml`
- **Dependencies**: Check with `python -c "import torch, cv2, PIL; print('Dependencies OK')"`

**Input/Output**:
- **Input**: `image_path`, `scan_id`, optional `output_dir`
- **Output**: Success status, crop count, structured analysis, file paths

**Pipeline Workflow**:
1. **Multi-scale Segmentation**: Generate masks at multiple scales
2. **High-resolution Cropping**: Create crop images from masks
3. **CLIP Classification**: Classify crops with batch processing
4. **Prompt Building**: Generate comprehensive prompts
5. **Gemini Inference**: Process with dynamic mode selection
6. **Result Processing**: Extract and save structured results

**Key Configuration Parameters**:
- **Segmentation**: `multi_scale_scales`, `min_mask_area`
- **CLIP**: `batch_size`, `clip_threshold`
- **Gemini**: `model_name`, `batch_delay`, `include_label_context`
- **Output**: `batch_chunk_size`, `concurrent_requests`

**Generated Files Structure**:
- **outputs/**: JSON results, TXT analysis, metrics, visualization
- **crops/scan_<id>/**: Individual crop images

**Error Handling & Fallback**:
- **Segmentation failures**: Return error if no masks found
- **Cropping failures**: Return error if no crops generated
- **Gemini parse failures**: Use raw text as fallback

**Programmatic Usage**:
```python
from adapter import create_pipeline_adapter

# Initialize and process
adapter = create_pipeline_adapter("config.yaml")
result = adapter.process_image("/path/to/image.jpg", scan_id=12345)

# Check results
if result["success"]:
    print(f"Processed {result['num_crops']} crops")
    files = result.get("files", {})
    print(f"Results: {files.get('result_txt')}")
```

**Run Test**:
```bash
# Single image processing
python adapter.py /path/to/image.jpg --scan-id 12345 --config config.yaml

# Check outputs
ls outputs/scan_12345_*
ls crops/scan_12345/
```

#### 7.4.2) batch_test_adapter.py - Batch Testing Utility
**Function**: Automated batch testing tool for evaluating pipeline performance across multiple images
**Key Features**: CSV/JSON summaries, performance metrics, artifact organization

**Core Features**:
- **Batch Processing**: Process multiple images or entire directories
- **Performance Metrics**: Detailed timing and API usage statistics
- **Artifact Organization**: Consolidates all outputs into organized test runs
- **Mode Detection**: Automatically detects and reports inference mode
- **Flexible Input**: Supports single images or directories

**Usage Examples**:
```bash
# Process single image
python batch_test_adapter.py /path/to/image.jpg --scan-base 1000

# Process directory with limit
python batch_test_adapter.py /data/images --limit 5 --suffix test_run

# Full directory processing
python batch_test_adapter.py /data/images --config config.yaml --sleep 1.0
```

**Command Line Arguments**:
- **input**: Input directory or single image file
- **--config**: Optional config.yaml path
- **--limit**: Limit number of images (0 = all)
- **--suffix**: Suffix tag for summary filenames
- **--outdir**: Directory to store test outputs

**Output Organization**:
- **batch_summary.csv**: CSV summary with metrics
- **batch_summary.json**: JSON summary with details
- **scan_<id>/**: Per-scan organized outputs with crops and results

**CSV Summary Fields**:
- **Basic info**: scan_id, image, success, num_crops, elapsed_sec, mode
- **Per-image metrics**: API calls, tokens, response time
- **Phase timings**: segmentation, crop generation, CLIP, prompt building, Gemini inference

**Mode Detection**:
- **Chunked-Batch**: `batch_chunk_size > 1` → `chunked(size=X, workers=Y)`
- **Concurrent**: `concurrent_requests > 1` → `per-crop(concurrency=X)`

**Performance Monitoring**:
- **Per-image metrics**: API calls, tokens, response time
- **Phase timings**: Segmentation, crop generation, CLIP, prompt building, Gemini inference
- **Display**: Times in seconds, tokens as counts, mode as string

**Metrics Definitions**:
- **elapsed_sec**: Total pipeline time per image (end-to-end)
- **Phase timings**: Segmentation, crop generation, CLIP, prompt building, Gemini inference
- **gm_api_calls_image**: Number of Gemini API calls for this image
- **Token usage**: Input/output/total tokens attributable to this image only
- **gm_avg_response_time_ms_image**: Average Gemini API latency per request (ms)

**Current Optimal Configuration (v2)**:
- **output**: `concurrent_requests: 6`, `batch_chunk_size: 4`, `chunk_workers: 2`
- **gemini**: `batch_delay: 1.2`
- **recognizer**: `batch_size: 16` (CLIP batch size on CPU)

**CLIP Batch Size Effectiveness**:
- Controls CLIP embedding batch size in `classify_batch()`
- On CPU, `batch_size: 16` is a safe upper bound

**Error Handling**:
- Error handling with detailed logging
- Continue processing next image on failure
- Add error details to summary

**Run Test**:
```bash
# Quick smoke test
python batch_test_adapter.py /data/images --limit 2 --suffix smoke

# Full test run
python batch_test_adapter.py /data/images --config config.yaml --sleep 0.5
```

#### 7.4.3) Integration Pipeline Summary

**Complete Data Flow**:
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           WASTE ANALYSIS PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────────┘

Input Image (JPG/PNG)
         ↓
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Segmentation  │    │   Multi-Scale   │    │   Mask Fusion   │
│   (FastSAM)     │───▶│   Processing    │───▶│   (IoU-based)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ↓
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Crop Objects  │    │   High-Res      │    │   Quality       │
│   Generation    │───▶│   Cropping      │───▶│   Filtering     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ↓
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLIP Batch    │    │   Statistical   │    │   Strategy      │
│   Classification│───▶│   Analysis      │───▶│   Selection     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ↓
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prompt        │    │   Gemini API    │    │   Structured    │
│   Building      │───▶│   Inference     │───▶│   Parsing       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ↓
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   JSON Results  │    │   TXT Analysis  │    │   Visualization │
│   (Structured)  │    │   (Human-Read)  │    │   (Crops)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘

Output Files:
├── scan_<id>_result.json          # Structured analysis
├── scan_<id>_result.txt           # Human-readable results
├── scan_<id>_crops_visualization.jpg  # Visual overview
├── scan_<id>_gemini_metrics.json  # API usage stats
└── crops/scan_<id>/               # Individual crop images
    ├── crop_001.jpg
    ├── crop_002.jpg
    └── ...
```

**Key Integration Points**:
- **Configuration**: All modules read from unified `config.yaml`
- **Error Propagation**: Failures at any stage are handled
- **File Management**: Consistent naming and organization across modules
- **Mode Selection**: Dynamic inference mode based on configuration
- **Caching**: CLIP and Gemini responses are cached for efficiency

**Performance Characteristics**:
- **Single Image**: 30-120 seconds (depending on crop count and API response time)
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

## 8) Configuration Reference

### 8.1) Configuration Parameters

**Segmentation**:
- `segmenter.model_type`: "fastsam"
- `segmenter.model_path`: "models/FastSAM-s.pt"
- `segmenter.multi_scale_scales`: [0.5, 1.0, 1.5]
- `segmenter.min_mask_area`: 3600
- `segmenter.use_postprocess`: false

**CLIP Classification**:
- `recognizer.model_type`: "clip"
- `recognizer.label_config_path`: "waste_labels_template.json"
- `recognizer.batch_size`: 16
- `clip_threshold`: 0.1

**Gemini Inference**:
- `gemini.API_key`: "your_API_key_here"
- `gemini.model_name`: "gemini-2.5-flash"
- `gemini.batch_delay`: 1.2
- `gemini.include_label_context`: true
- `gemini.uncertainty_focus`: true

**Output Control**:
- `output.batch_chunk_size`: 4
- `output.chunk_workers`: 2
- `output.concurrent_requests`: 6

## 9) Usage Examples

**Single Image Processing**:
```bash
python adapter.py /path/to/image.jpg --scan-id 12345 --config config.yaml
```

**Batch Processing**:
```bash
python batch_test_adapter.py /data/images --limit 5 --suffix test_run
```

**Programmatic Usage**:
```python
from adapter import create_pipeline_adapter

adapter = create_pipeline_adapter("config.yaml")
result = adapter.process_image("/path/to/image.jpg", scan_id=12345)

if result["success"]:
    print(f"Processed {result['num_crops']} crops")
    files = result.get("files", {})
    print(f"Results: {files.get('result_txt')}")
```

## 10) Testing & Performance

**Quick Tests**:
```bash
# Test Gemini connectivity
python -c "from gemini_inferencer import create_gemini_inferencer; print('Gemini OK:', create_gemini_inferencer().test_connection())"

# Generate labels
python generate_label_template.py

# Single image test
python adapter.py test_super_u_images/Super_U_20250624_085602.jpg --scan-id 8888 --config config.yaml
```

**Batch Testing**:
```bash
# Smoke test (2 images)
python batch_test_adapter.py /data/images --limit 2 --suffix smoke

# Full directory test
python batch_test_adapter.py test_super_u_images/ --config config.yaml --sleep 1.0
```

**Performance Characteristics**:
- **Single Image**: 30-120 seconds (depending on crop count and API response time)
- **Batch Processing**: Linear scaling with configurable concurrency
- **Memory Usage**: Moderate (crops and embeddings cached)
- **API Efficiency**: Caching reduces redundant calls

## 11) Troubleshooting

**Common Issues & Solutions**:

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
- **API Key errors**: Set `GEMINI_API_KEY` environment variable or in `config.yaml`
- **Parse failures**: Check `parse_status` in `scan_*_gemini_metrics.json`
- **JSON truncation**: Increase `max_output_tokens` to 4096, use TXT fallback
- **Rate limiting**: Increase `batch_delay` to 1.5s, reduce `concurrent_requests`

**Performance Issues**:
- **Slow processing**: Disable `use_postprocess`, reduce `multi_scale_scales`
- **Memory issues**: Reduce `batch_chunk_size`, clear caches
- **API quota**: Enable caching, reduce batch sizes

**Debug Commands**:
```bash
# Check configuration
python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['segmenter'])"

# Test individual components
python -c "from segmenter import Segmenter; print('Segmentation OK')"
python -c "from clip_matcher import CLIPMatcher; print('CLIP OK')"
python -c "from gemini_inferencer import create_gemini_inferencer; print('Gemini OK:', create_gemini_inferencer().test_connection())"

# Clear all caches
rm -rf gemini_cache/* clip_results_cache/* crops/* outputs/*
```

## 12) VM Deployment

### 12.1) Upload Bundle to VM
```bash
# Option 1: scp (initial upload)
scp -i ~/.ssh/<key> -r upload_vm_bundle <user>@<host>:~/ImageAPI/

# Option 2: rsync (incremental updates - faster)
rsync -avz -e "ssh -i ~/.ssh/<key>" upload_vm_bundle/ <user>@<host>:~/ImageAPI/pipeline/
```

### 12.2) VM Environment Setup
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

### 12.3) VM Testing Commands
```bash
# Test individual components
python -c "from gemini_inferencer import create_gemini_inferencer; print('Gemini OK:', create_gemini_inferencer().test_connection())"

# Single image test
python adapter.py test_super_u_images/Super_U_20250624_085602.jpg --scan-id 9999 --config config.yaml

# Batch test (3 images)
python batch_test_adapter.py test_super_u_images/ --limit 3 --suffix vm_test

# Check outputs
ls outputs_tests/run_vm_test/
```

### 12.4) VM Production Integration
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

### 12.5) VM Maintenance
```bash
# Clear caches periodically
rm -rf gemini_cache/* clip_results_cache/* crops/* outputs/*

# Update pipeline code
rsync -avz -e "ssh -i ~/.ssh/<key>" upload_vm_bundle/ <user>@<host>:~/ImageAPI/pipeline/

# Monitor disk usage
df -h
du -sh gemini_cache/ clip_results_cache/ crops/ outputs/
```

## 13) License
Follows the licenses of integrated SDKs and models.
