# Visualization Module

Interactive HTML report generator for batch test results.

## Quick Start

```bash
python batch_report_generator.py <csv_file> [--output-dir DIR] [--model NAME]
```

**Example:**
```bash
python batch_report_generator.py \
    ../demo_results/gemini-2.5-flash/outputs/batch_summary_visualization_demo.csv \
    --output-dir reports \
    --model "Gemini 2.5 Flash"
```

**Output:** `reports/batch_report_YYYYMMDD_HHMMSS.html` (~30-40 MB)

## Directory Structure

```
visualization/
├── README.md                      # This file
├── batch_report_generator.py      # Main generator
├── test_report_generator.py       # Test script
├── examples/
│   └── generate_report_example.sh # Batch generation script
└── reports/                       # Output directory
```

## Features

- **Interactive Visualization**: Click to expand/collapse details
- **Image Display**:
  - Segmented Regions with BBoxes
  - Individual Crops (clickable)
- **Performance Metrics**: Processing time, API calls, token usage
- **Self-contained HTML**: Base64-embedded images, single-file shareable

## Input Requirements

### CSV File

**Required columns:**
- `image`, `scan_id`, `success`, `num_crops`, `elapsed_sec`
- `mode`, `gm_api_calls_image`, `gm_total_tokens_image`
- `gm_input_tokens_image`, `gm_output_tokens_image`
- `gm_avg_response_time_ms_image`

**Optional columns:**
- `segmentation_sec`, `crop_generation_sec`
- `clip_classification_sec`, `prompt_building_sec`
- `gemini_inference_sec`

### File Structure

```
outputs_directory/
├── batch_summary_*.csv
└── scan_<id>/
    ├── scan_<id>_crops_visualization.jpg
    ├── scan_<id>_result.txt
    └── crops/
        └── scan_<id>_crop*.jpg
```

## Output Structure

### HTML Report Contains:

1. **Summary Cards** (top)
   - Total images, objects, processing time
   - Average time per image, success rate
   - Average tokens per object

2. **Results Table**
   - Image name, status, object count
   - Processing time, API calls, tokens/object
   - Processing mode

3. **Expandable Details** (click table row)
   - **Left Panel**:
     - Segmented Regions with BBoxes
     - Individual Crops list (clickable)
   - **Right Panel**:
     - Detailed Analysis Results (620px scrollable)
     - Test Results Statistics
     - Phase Time Distribution

### Interactive Features:
- Click table row → Expand/collapse details
- Click image → View in modal (zoom/pan with mouse)
- Click crop button → View individual crop
- ESC key → Close modal/collapse

## Arguments

- `csv_path` - Path to batch summary CSV file
- `--output-dir DIR` - Output directory (default: `reports`)
- `--model NAME` - Model name to display in report

## Notes

- Requires modern browser (Chrome 80+, Firefox 75+, Safari 13+)
- Report size: ~30-40 MB (Base64-embedded images)
- For large datasets (>50 images), split into multiple CSVs
