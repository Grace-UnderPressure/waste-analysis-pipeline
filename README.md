# 🗂️ Waste Analysis Pipeline

A comprehensive AI-powered waste classification system using FastSAM + CLIP + Gemini for multimodal analysis.

## ✨ Features

- **🔍 Multi-scale Segmentation**: FastSAM with intelligent mask fusion
- **🏷️ CLIP Classification**: Preliminary waste categorization  
- **🧠 Gemini Analysis**: Detailed multimodal understanding with original image context
- **📊 Structured Output**: JSON + human-readable TXT reports
- **🚀 VM Ready**: Optimized for cloud deployment
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
├── advanced_segmentation_methods.py   # Enhanced mask fusion algorithms
├── clip_matcher.py                    # CLIP classification engine
├── clip_results_analyzer.py           # CLIP results analysis
├── gemini_inferencer.py               # Gemini API integration
├── gemini_prompt_builder.py           # Intelligent prompt construction
├── waste_analysis_schema.py           # Structured output schemas
├── batch_test_adapter.py              # Batch testing utilities
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
# Edit config.yaml with your API keys
cp config.yaml config_local.yaml
# Add your Gemini API key to config_local.yaml
# Note: Optimized for CPU environments (tested on GCP VM)
```

### Basic Usage
```python
from adapter import WasteAnalysisAdapter

adapter = WasteAnalysisAdapter("config.yaml")
result = adapter.process_image("path/to/image.jpg", scan_id=12345)
print(f"Analysis saved to: {result['files']['json_path']}")
```

### Batch Testing
```bash
python batch_test_adapter.py --input_dir test_super_u_images/ --output outputs_tests/run_test --scan_base 1000
```

## 📊 Performance

- **Processing Speed**: ~1-2 minutes per image on CPU (6-19 crops, depending on API response time)
- **Segmentation**: Multi-scale (0.5x, 1.0x, 1.5x) with IoU-based mask fusion, ~60% reduction in redundant crops
- **Classification**: CLIP + statistical analysis with relaxed unknown rules
- **Analysis**: Gemini 2.5 Flash with original image context and structured output
- **API Efficiency**: Parallel chunked-batch mode for optimal throughput
- **Output**: Structured JSON + human-readable TXT + visualization reports

## 🔧 Key Improvements

- **Enhanced Mask Fusion**: IoU + containment analysis for better crop quality, ~60% reduction in redundant crops
- **Original Image Context**: Full scene understanding for Gemini analysis
- **Relaxed CLIP Rules**: Reduced false "unknown" classifications
- **API Optimization**: Parallel chunked-batch mode with configurable concurrency
- **Phase-wise Timing**: Detailed performance breakdown per processing stage
- **Interactive Reports**: HTML visualization with per-image metrics and phase analysis
- **VM Optimized**: Ready for cloud deployment with comprehensive testing

## 📖 Documentation

See `pipeline/README.md` for comprehensive developer documentation including:
- Detailed module descriptions
- Data flow diagrams
- Configuration options
- Testing procedures
- VM deployment guide

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
**Latest Update**: December 2024
