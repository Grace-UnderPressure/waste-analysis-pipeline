# 🗂️ Waste Analysis Pipeline

A production-ready computer vision pipeline for automated waste object detection, classification, and analysis using multi-scale segmentation, vision-language models, and multimodal large language models.

## ✨ Features

- **🔍 Multi-scale Segmentation**: FastSAM with IoU-based mask fusion
- **🏷️ CLIP Classification**: Preliminary waste categorization  
- **🧠 Gemini Analysis**: Detailed multimodal understanding with original image context
- **📊 Structured Output**: JSON + human-readable TXT reports
- **🚀 VM Ready**: Ready for cloud deployment
- **🛠️ Developer Friendly**: Comprehensive documentation and testing tools

## 🏗️ Architecture

```
Image → FastSAM Segmentation → Crop Generation → CLIP Classification → Gemini Analysis → Results
```

## 📁 Project Structure

```
pipeline/
├── README.md                          # Detailed developer documentation
├── adapter.py                         # Main pipeline orchestrator
├── config.yaml                        # Configuration settings
├── requirements.txt                   # Python dependencies
├── segmenter.py                       # FastSAM integration
├── advanced_segmentation_methods.py   # Mask fusion algorithms
├── clip_matcher.py                    # CLIP classification engine
├── clip_results_analyzer.py           # CLIP results analysis
├── gemini_inferencer.py               # Gemini API integration
├── gemini_prompt_builder.py           # Prompt construction
├── waste_analysis_schema.py           # Structured output schemas
├── batch_test_adapter.py              # Batch testing utilities
├── demo_sample_images/                # Demo test images (8 images)
├── demo_results/                      # Demo results with HTML reports
│   ├── gemini-2.5-flash/
│   └── gemini-2.5-flash-lite/
├── visualization/                     # HTML report generation tool
└── WT_experiment_dataset_picked/      # Test dataset (29 images)
```

## 🚀 Quick Start

### Installation
```bash
cd pipeline/
pip install -r requirements.txt
```

### Configuration
```bash
# Set your Gemini API key as environment variable
export GEMINI_API_KEY="your_api_key_here"

# Or edit config.yaml to add your API key directly
# Note: Designed for CPU environments (tested on GCP VM)
```

### Basic Usage
```python
from adapter import create_pipeline_adapter

adapter = create_pipeline_adapter("config.yaml")
result = adapter.process_image("path/to/image.jpg", scan_id=12345)
print(f"Analysis saved to: {result['files']['result_json']}")
print(f"Crops saved to: {result['files']['crops_dir']}")
```

### Batch Testing
```bash
python batch_test_adapter.py test_super_u_images/ --scan-base 1000 --suffix test_run
```

### CLI Usage
```bash
# Single image processing
python adapter.py path/to/image.jpg --scan-id 12345 --config config.yaml
```

## 📊 Performance

- **Processing Speed**: 0.5-2 minutes per image on CPU (varies by model version and crop count)
- **Segmentation**: Multi-scale (0.5x, 1.0x, 1.5x) with IoU-based mask fusion, ~60% reduction in redundant crops
- **Classification**: CLIP + statistical analysis with relaxed unknown rules
- **Analysis**: Gemini Flash with original image context and structured output
- **API Efficiency**: Parallel chunked-batch mode for optimal throughput
- **Output**: Structured JSON + human-readable TXT + visualization reports

## 🔧 Key Improvements

- **Mask Fusion**: IoU + containment analysis for crop quality, ~60% reduction in redundant crops
- **Original Image Context**: Full scene understanding for Gemini analysis
- **Relaxed CLIP Rules**: Reduced false "unknown" classifications
- **API Optimization**: Parallel chunked-batch mode with configurable concurrency
- **Phase-wise Timing**: Detailed performance breakdown per processing stage
- **Interactive Reports**: HTML visualization with per-image metrics and phase analysis
- **VM Ready**: Cloud deployment with comprehensive testing

## 📖 Documentation

See `pipeline/README.md` for comprehensive developer documentation including:
- Detailed module descriptions
- Data flow diagrams
- Configuration options
- Testing procedures
- VM deployment guide

## 🎯 Live Demo

Interactive HTML reports with results from Gemini 2.5 Flash and Flash Lite models:

**View Reports Online:**
- [Gemini 2.5 Flash Report](https://htmlpreview.github.io/?https://raw.githubusercontent.com/Grace-UnderPressure/waste-analysis-pipeline/main/pipeline/demo_results/gemini-2.5-flash/report.html)
- [Gemini 2.5 Flash Lite Report](https://htmlpreview.github.io/?https://raw.githubusercontent.com/Grace-UnderPressure/waste-analysis-pipeline/main/pipeline/demo_results/gemini-2.5-flash-lite/report.html)

> ⚠️ Reports are ~39 MB each. Initial load may take 20-30 seconds.

**Demo Results:**

| Model | Avg Time/Image | Avg Tokens/Object | Objects | Images |
|-------|----------------|-------------------|---------|--------|
| Gemini 2.5 Flash | 127.4s | 3,120 | 111 | 8 |
| Gemini 2.5 Flash Lite | 37.7s | 2,041 | 111 | 8 |

**Demo Structure:**
- `demo_sample_images/` - 8 test images (ALTRIANE, Bricocash, Super U)
- `demo_results/` - Complete test results with interactive HTML reports
- `visualization/` - Report generation tool

**Generate Your Own Reports:**
```bash
cd pipeline/visualization
python batch_report_generator.py \
    ../demo_results/gemini-2.5-flash/outputs/batch_summary_*.csv \
    --model "Gemini 2.5 Flash"
```

## 🧪 Testing

The repository includes:
- **29 test images** from multiple scenarios
- **Batch testing utilities** with per-image metrics and phase timing
- **Interactive HTML reports** for result visualization
- **Performance optimization** with configurable API concurrency modes
- **Comprehensive test results** and analysis
- **VM deployment validation**

## 🤝 Contributing

This is a personal research project. For questions or suggestions, please open an issue.

## 📄 License

MIT License - See LICENSE file for details.

---

**Author**: Grace-UnderPressure  
**Email**: yulong.ma23@gmail.com  
**Latest Update**: Oct 2025
