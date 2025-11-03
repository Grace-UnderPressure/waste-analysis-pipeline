# Demo: Waste Analysis Pipeline Visualization

Interactive HTML reports showcasing waste analysis results from Gemini 2.5 Flash and Flash Lite models.

## Structure

```
├── demo_sample_images/     # 8 test images (ALTRIANE, Bricocash, Super U)
├── demo_results/           # Test results (~136 MB total)
│   ├── gemini-2.5-flash/
│   │   ├── report.html     # Interactive report (~39 MB)
│   │   └── outputs/        # Raw results (CSV + scans)
│   └── gemini-2.5-flash-lite/
│       ├── report.html
│       └── outputs/
└── visualization/          # Report generation tool
```

## Quick Start

**View Reports:**
- Open `demo_results/gemini-2.5-flash/report.html` in browser
- Requires Chrome 80+, Firefox 75+, or Safari 13+

**Generate Reports:**
```bash
cd visualization
python batch_report_generator.py \
    ../demo_results/gemini-2.5-flash/outputs/batch_summary_*.csv \
    --model "Gemini 2.5 Flash"
```

## Features

- **Interactive Visualization**: Click rows to expand, zoom images, view individual crops
- **Performance Metrics**: Processing time, API calls, token usage per image/object
- **Model Comparison**: Side-by-side evaluation of different Gemini models
- **Self-contained**: All assets embedded, no external dependencies

## Results

| Model | Avg Time/Image | Avg Tokens/Object | Images |
|-------|----------------|-------------------|--------|
| Gemini 2.5 Flash | ~60-90s | ~2,500 | 8 |
| Gemini 2.5 Flash Lite | ~50-80s | ~2,400 | 8 |

**Note**: Reports are ~39 MB each due to embedded Base64 images. Consider Git LFS for version control.

---

See `visualization/README.md` for tool documentation.

