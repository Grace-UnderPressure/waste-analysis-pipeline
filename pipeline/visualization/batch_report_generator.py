#!/usr/bin/env python3
"""
Batch Test Results Interactive HTML Report Generator

Generates self-contained, interactive HTML reports from batch test CSV data.
Reports include visualization images, analysis results, and performance statistics.

Usage:
    python batch_report_generator.py <csv_file> [--output-dir <dir>] [--model <name>]

Example:
    python batch_report_generator.py \\
        ../outputs_tests/visualization_demo/batch_summary_visualization_demo.csv \\
        --output-dir reports \\
        --model "Gemini 2.5 Flash"

Input:
    - CSV file with batch test summary data
    - Related scan directories with visualization images and results

Output:
    - Self-contained HTML file (~30-40 MB, Base64-embedded images)
    - Filename: batch_report_YYYYMMDD_HHMMSS.html

Author: Waste Classification Research Team
Version: 1.2
Last Updated: 2025-11-03
"""

import os
import csv
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse


class BatchReportGenerator:
    """
    Generator for interactive batch test result HTML reports.
    
    This class handles:
    - Loading CSV data and related files
    - Encoding images to Base64
    - Calculating summary statistics
    - Generating self-contained HTML with embedded assets
    
    Attributes:
        csv_path (Path): Path to the input CSV file
        output_dir (Path): Directory for output reports
        model_name (str): Model name to display in report
        summary_data (List[Dict]): Loaded CSV data
        image_data (Dict): Metadata for related image files
    """
    
    def __init__(self, csv_path: str, output_dir: str = None, model_name: str = None):
        """
        Initialize the batch report generator.
        
        Args:
            csv_path (str): Path to batch_summary CSV file
            output_dir (str, optional): Output directory for reports. 
                                       Defaults to 'reports'.
            model_name (str, optional): Model name to display in report.
                                       Defaults to 'Unknown Model'.
        
        Example:
            >>> generator = BatchReportGenerator(
            ...     "outputs/batch_summary.csv",
            ...     output_dir="reports",
            ...     model_name="Gemini 2.5 Flash"
            ... )
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir) if output_dir else Path("visualization/reports/html")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name or "Unknown Model"
        
        # Data storage
        self.summary_data = []
        self.image_data = {}
        
    def load_csv_data(self):
        """Load CSV data"""
        print(f"📊 Loading CSV data: {self.csv_path}")
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.summary_data = list(reader)
        
        print(f"✅ Loaded {len(self.summary_data)} records")
        
    def find_related_files(self):
        """Find related files for each image"""
        print("🔍 Finding related files...")
        
        csv_dir = self.csv_path.parent
        
        for row in self.summary_data:
            scan_id = row['scan_id']
            image_name = row['image']
            
            # Find crops visualization image
            crops_viz_path = None
            for pattern in [f"scan_{scan_id}_crops_visualization.jpg", 
                           f"scan_{scan_id}/scan_{scan_id}_crops_visualization.jpg"]:
                path = csv_dir / pattern
                if path.exists():
                    crops_viz_path = path
                    break
            
            # Find result.txt file
            result_txt_path = None
            for pattern in [f"scan_{scan_id}_result.txt",
                           f"scan_{scan_id}/scan_{scan_id}_result.txt"]:
                path = csv_dir / pattern
                if path.exists():
                    result_txt_path = path
                    break
            
            self.image_data[scan_id] = {
                'image_name': image_name,
                'crops_viz_path': crops_viz_path,
                'result_txt_path': result_txt_path,
                'row_data': row
            }
            
            print(f"  📷 {image_name}:")
            print(f"    - Crops Visualization: {'✅' if crops_viz_path else '❌'}")
            print(f"    - Analysis Results: {'✅' if result_txt_path else '❌'}")
    
    def encode_image_to_base64(self, image_path: Path) -> Optional[str]:
        """Encode image to base64 data URI"""
        if not image_path or not image_path.exists():
            return None
            
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')
                ext = image_path.suffix.lower()
                mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
                return f"data:{mime_type};base64,{base64_data}"
        except Exception as e:
            print(f"⚠️ Failed to encode image {image_path}: {e}")
            return None
    
    def read_text_file(self, file_path: Path) -> Optional[str]:
        """Read text file content"""
        if not file_path or not file_path.exists():
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"⚠️ Failed to read file {file_path}: {e}")
            return None
    
    def calculate_summary_stats(self) -> Dict:
        """Calculate summary statistics"""
        if not self.summary_data:
            return {}
        
        total_images = len(self.summary_data)
        total_crops = sum(int(row.get('num_crops', 0)) for row in self.summary_data)
        total_time = sum(float(row.get('elapsed_sec', 0)) for row in self.summary_data)
        success_count = sum(1 for row in self.summary_data if row.get('success', '').lower() == 'true')
        total_tokens = sum(int(row.get('gm_total_tokens_image', 0) or 0) for row in self.summary_data)
        
        return {
            'total_images': total_images,
            'total_crops': total_crops,
            'total_time': total_time,
            'avg_time_per_image': total_time / total_images if total_images > 0 else 0,
            'success_rate': success_count / total_images * 100 if total_images > 0 else 0,
            'total_tokens': total_tokens,
            'avg_tokens_per_crop': total_tokens / total_crops if total_crops > 0 else 0
        }
    
    def generate_html_report(self) -> str:
        """Generate fixed layout HTML report"""
        print("📝 Generating fixed layout HTML report...")
        
        stats = self.calculate_summary_stats()
        csv_dir = self.csv_path.parent
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fixed Layout Interactive Batch Test Results Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 16px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 1.5em;
            font-weight: 300;
        }}
        
        .header p {{
            margin: 4px 0 0 0;
            opacity: 0.9;
            font-size: 0.85em;
        }}
        
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 10px;
            padding: 12px;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        
        .stat-number {{
            font-size: 1.4em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 3px;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.8em;
        }}
        
        .image-section {{
            padding: 15px;
            border-top: 1px solid #eee;
        }}
        
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .summary-table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 500;
        }}
        
        .summary-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
            vertical-align: middle;
        }}
        
        .summary-table tr:hover {{
            background: #f8f9fa;
            cursor: pointer;
        }}
        
        .summary-table tr.active {{
            background: #e3f2fd;
        }}
        
        .image-name {{
            font-weight: bold;
            color: #333;
        }}
        
        .status-badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        
        .status-success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status-failed {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .metric-value {{
            font-weight: bold;
            color: #667eea;
        }}
        
        .expandable-content {{
            display: none;
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .expandable-content.show {{
            display: block;
        }}
        
        .detail-grid {{
            display: flex;
            gap: 15px;
            align-items: flex-start;
        }}
        
        .left-panel {{
            flex: 1 1 50%;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        
        .image-stack {{
            display: flex;
            flex-direction: column;
            gap: 12px;
            flex-shrink: 0;
        }}
        
        .image-container {{
            text-align: center;
            max-height: 700px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .image-container img {{
            max-width: 100%;
            max-height: 650px;
            width: auto;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            object-fit: contain;
            cursor: pointer;
            transition: transform 0.2s ease;
        }}
        
        .image-container img:hover {{
            transform: scale(1.02);
        }}
        
        /* Modal styles */
        .image-modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.8);
            cursor: pointer;
        }}
        
        .modal-content {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            cursor: grab;
            transition: transform 0.1s ease;
        }}
        
        .modal-content:active {{
            cursor: grabbing;
        }}
        
        .modal-content.zoomed {{
            cursor: grab;
        }}
        
        .modal-close {{
            position: absolute;
            top: 15px;
            right: 25px;
            color: white;
            font-size: 35px;
            font-weight: bold;
            cursor: pointer;
            z-index: 1001;
        }}
        
        .modal-close:hover {{
            color: #ccc;
        }}
        
        /* Zoom hint */
        .zoom-hint {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            background: rgba(0,0,0,0.7);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        
        .zoom-hint.show {{
            opacity: 1;
        }}
        
        .image-caption {{
            margin-top: 10px;
            font-weight: bold;
            color: #333;
            font-size: 0.9em;
        }}
        
        .crops-list {{
            background: white;
            padding: 12px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-height: 200px;
            overflow-y: auto;
        }}
        
        .crops-list h4 {{
            margin: 0 0 8px 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 4px;
            font-size: 0.95em;
        }}
        
        .crop-items {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
            gap: 6px;
        }}
        
        .crop-item {{
            padding: 6px 10px;
            background: #f8f9fa;
            border-radius: 4px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 0.8em;
            font-weight: 500;
            color: #555;
        }}
        
        .crop-item:hover {{
            background: #667eea;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 2px 6px rgba(102, 126, 234, 0.3);
        }}
        
        .stats-panel {{
            background: white;
            padding: 12px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .stats-panel h3, .stats-panel h4 {{
            margin: 0 0 8px 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 4px;
            font-size: 1em;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
            margin-bottom: 10px;
        }}
        
        .stat-item {{
            display: flex;
            justify-content: space-between;
            padding: 5px;
            background: #f8f9fa;
            border-radius: 3px;
            font-size: 0.85em;
        }}
        
        .stat-label {{
            font-weight: bold;
            color: #666;
        }}
        
        .stat-value {{
            font-weight: bold;
            color: #667eea;
        }}
        
        .phase-chart {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        
        .phase-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.8em;
            padding: 4px 6px;
            background: #f8f9fa;
            border-radius: 3px;
            border-left: 2px solid transparent;
        }}
        
        .phase-item:nth-child(1) {{ border-left-color: #667eea; }}
        .phase-item:nth-child(2) {{ border-left-color: #764ba2; }}
        .phase-item:nth-child(3) {{ border-left-color: #f093fb; }}
        .phase-item:nth-child(4) {{ border-left-color: #f5576c; }}
        .phase-item:nth-child(5) {{ border-left-color: #4facfe; }}
        
        .phase-name {{
            min-width: 70px;
            font-weight: bold;
            color: #333;
            font-size: 0.8em;
        }}
        
        .phase-bar-mini {{
            flex: 1;
            height: 16px;
            background: #e9ecef;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
        }}
        
        .phase-fill-mini {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
            border-radius: 8px;
        }}
        
        .phase-time {{
            min-width: 85px;
            text-align: right;
            font-size: 0.75em;
            color: #666;
        }}
        
        .phase-percentage {{
            font-weight: bold;
            color: #667eea;
            margin-left: 5px;
        }}
        
        .right-panel {{
            flex: 1 1 50%;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        
        .right-panel h3 {{
            margin: 0 0 8px 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 4px;
            flex-shrink: 0;
            font-size: 1em;
        }}
        
        .analysis-section {{
            flex: 0 0 auto;
            display: flex;
            flex-direction: column;
        }}
        
        .analysis-content {{
            background: white;
            padding: 12px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-y: auto;
            height: 620px;
            max-height: 620px;
        }}
        
        .analysis-content pre {{
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            line-height: 1.4;
            margin: 0;
        }}
        
        .right-stats-panel {{
            flex: 0 0 auto;
            background: white;
            padding: 12px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .no-data {{
            text-align: center;
            color: #999;
            font-style: italic;
            padding: 40px;
        }}
        
        .timestamp {{
            text-align: center;
            color: #999;
            font-size: 0.9em;
            padding: 20px;
            border-top: 1px solid #eee;
        }}
        
        .click-hint {{
            font-size: 0.8em;
            color: #666;
            font-style: italic;
            margin-top: 10px;
        }}
        
        @media (max-width: 768px) {{
            .detail-grid {{
                grid-template-columns: 1fr;
                align-items: stretch;
                min-height: auto;
            }}
            
            .left-panel, .right-panel {{
                height: auto;
            }}
            
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
            
            .summary-table {{
                font-size: 0.9em;
            }}
            
            .summary-table th,
            .summary-table td {{
                padding: 8px;
            }}
            
            .phase-item {{
                flex-direction: column;
                align-items: flex-start;
                gap: 5px;
            }}
            
            .phase-name {{
                min-width: auto;
            }}
            
            .phase-time {{
                min-width: auto;
                text-align: left;
            }}
            
            .analysis-content {{
                max-height: 400px;
                height: 400px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Waste Analysis Pipeline - Batch Test Report</h1>
            <p>Click any row to view detailed analysis results</p>
            <p style="margin-top:6px;font-size:0.95em;opacity:0.95;">Model: {self.model_name}</p>
        </div>
        
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-number">{stats.get('total_images', 0)}</div>
                <div class="stat-label">Processed Images</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats.get('total_crops', 0)}</div>
                <div class="stat-label">Detected Objects</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats.get('total_time', 0):.1f}s</div>
                <div class="stat-label">Total Processing Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats.get('avg_time_per_image', 0):.1f}s</div>
                <div class="stat-label">Avg Time/Image</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats.get('success_rate', 0):.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats.get('avg_tokens_per_crop', 0):.0f}</div>
                <div class="stat-label">Avg Tokens/Object</div>
            </div>
        </div>
        
        <div class="image-section">
            <h2>📊 Detailed Test Results</h2>
            <div class="click-hint">💡 Click table rows to view corresponding images and analysis results</div>
            
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>Image Name</th>
                        <th>Status</th>
                        <th>Objects</th>
                        <th>Processing Time</th>
                        <th>API Calls</th>
                        <th>Tokens/Object</th>
                        <th>Mode</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add table rows
        for i, (scan_id, data) in enumerate(self.image_data.items()):
            row = data['row_data']
            image_name = data['image_name']
            success = row.get('success', '').lower() == 'true'
            
            # Get per-image metrics
            tokens = row.get('gm_total_tokens_image', 0)
            num_crops = int(row.get('num_crops', 0))
            tokens_per_crop = int(tokens) / num_crops if num_crops > 0 else 0
            api_calls = row.get('gm_api_calls_image', 0)
            
            html_content += f"""
                    <tr onclick="toggleRow({i})">
                        <td class="image-name">{image_name}</td>
                        <td>
                            <span class="status-badge {'status-success' if success else 'status-failed'}">
                                {'✅ Success' if success else '❌ Failed'}
                            </span>
                        </td>
                        <td class="metric-value">{row.get('num_crops', 'N/A')}</td>
                        <td class="metric-value">{row.get('elapsed_sec', 'N/A')}s</td>
                        <td class="metric-value">{api_calls}</td>
                        <td class="metric-value">{tokens_per_crop:.0f}</td>
                        <td>{row.get('mode', 'N/A')}</td>
                    </tr>
"""
        
        html_content += """
                </tbody>
            </table>
            
            <!-- Expandable content area -->
"""
        
        # Add expandable content
        for i, (scan_id, data) in enumerate(self.image_data.items()):
            row = data['row_data']
            image_name = data['image_name']
            
            # Encode images
            crops_viz_b64 = self.encode_image_to_base64(data['crops_viz_path'])
            
            # Find individual crop images
            crops_dir = csv_dir / f"scan_{scan_id}" / "crops"
            crop_images = []
            if crops_dir.exists():
                for crop_file in sorted(crops_dir.glob(f"scan_{scan_id}_crop*.jpg")):
                    crop_b64 = self.encode_image_to_base64(crop_file)
                    if crop_b64:
                        crop_images.append({
                            'name': crop_file.stem.replace(f'scan_{scan_id}_', ''),
                            'b64': crop_b64
                        })
            
            # Read analysis results
            analysis_text = self.read_text_file(data['result_txt_path'])
            
            # Calculate phase timing data
            phase_times = {
                'segmentation': float(row.get('segmentation_sec', 0)),
                'crop_generation': float(row.get('crop_generation_sec', 0)),
                'clip_classification': float(row.get('clip_classification_sec', 0)),
                'prompt_building': float(row.get('prompt_building_sec', 0)),
                'gemini_inference': float(row.get('gemini_inference_sec', 0))
            }
            total_phase_time = sum(phase_times.values())
            
            # Get per-image metrics
            tokens = row.get('gm_total_tokens_image', 0)
            api_calls = row.get('gm_api_calls_image', 0)
            response_time = row.get('gm_avg_response_time_ms_image', 0)
            
            html_content += f"""
            <div class="expandable-content" id="content-{i}">
                <div class="detail-grid">
                    <div class="left-panel">
                        <div class="image-stack">
                            <div class="image-container">
                                <img src="{crops_viz_b64 or 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIEltYWdlPC90ZXh0Pjwvc3ZnPg=='}" alt="Segmented Regions with BBoxes">
                                <div class="image-caption">Segmented Regions with BBoxes ({row.get('num_crops', 'N/A')} objects)</div>
                            </div>
                        </div>
                        
                        <div class="crops-list">
                            <h4>📦 Individual Crops ({len(crop_images)})</h4>
                            <div class="crop-items">
"""
            
            # Add crop list items
            for crop_info in crop_images:
                html_content += f"""
                                <div class="crop-item" onclick="showCropImage('{crop_info['b64']}', '{crop_info['name']}')">
                                    {crop_info['name']}
                                </div>
"""
            
            html_content += """
                            </div>
                        </div>
                    </div>
                    
                    <div class="right-panel">
                        <div class="analysis-section">
                            <h3>🔍 Detailed Analysis Results</h3>
                            <div class="analysis-content">
                                <pre>"""
            
            # Add analysis results
            html_content += f"""{analysis_text}""" if analysis_text else """📝 Analysis results file not found"""
            
            html_content += f"""</pre>
                            </div>
                        </div>
                        
                        <div class="right-stats-panel">
                            <h3>📊 Test Results Statistics</h3>
                            <div class="stats-grid">
                                <div class="stat-item">
                                    <span class="stat-label">Processing Time:</span>
                                    <span class="stat-value">{row.get('elapsed_sec', 'N/A')}s</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">API Calls:</span>
                                    <span class="stat-value">{api_calls}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Input Tokens:</span>
                                    <span class="stat-value">{int(row.get('gm_input_tokens_image', 0)):,}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Output Tokens:</span>
                                    <span class="stat-value">{int(row.get('gm_output_tokens_image', 0)):,}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Total Tokens:</span>
                                    <span class="stat-value">{int(tokens):,}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Avg API Response Time:</span>
                                    <span class="stat-value">{float(response_time)/1000:.3f}s</span>
                                </div>
                            </div>
                            
                            <h4>⏱️ Phase Time Distribution</h4>
                            <div class="phase-chart">
"""
            
            # Add phase timing chart
            phase_labels = {
                'segmentation': 'Segmentation',
                'crop_generation': 'Crop Generation', 
                'clip_classification': 'CLIP Classification',
                'prompt_building': 'Prompt Building',
                'gemini_inference': 'Gemini Inference'
            }
            
            for phase_name, time_val in phase_times.items():
                percentage = (time_val / total_phase_time * 100) if total_phase_time > 0 else 0
                html_content += f"""
                                <div class="phase-item">
                                    <span class="phase-name">{phase_labels[phase_name]}</span>
                                    <div class="phase-bar-mini">
                                        <div class="phase-fill-mini" style="width: {percentage:.1f}%"></div>
                                    </div>
                                    <span class="phase-time">{time_val:.3f}s</span>
                                    <span class="phase-percentage">({percentage:.1f}%)</span>
                                </div>
"""
            
            html_content += """
                            </div>
                        </div>
                    </div>
                </div>
            </div>
"""
        
        # Add timestamp and JavaScript
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_content += f"""
        </div>
        
        <div class="timestamp">
            Report Generated: {timestamp}
        </div>
    </div>
    
    <!-- Image Modal -->
    <div id="imageModal" class="image-modal">
        <span class="modal-close">&times;</span>
        <img class="modal-content" id="modalImage" src="" alt="">
        <div class="zoom-hint" id="zoomHint">Hover and scroll to zoom • Drag to move • Double-click to reset</div>
    </div>
    
    <script>
        function toggleRow(index) {{
            const currentRow = document.querySelectorAll('.summary-table tr')[index + 1]; // +1 for header
            const content = document.getElementById('content-' + index);
            
            // Check if current row is already active
            const isCurrentlyActive = currentRow.classList.contains('active');
            
            // Remove active state from all rows
            document.querySelectorAll('.summary-table tr').forEach(row => {{
                row.classList.remove('active');
            }});
            
            // Hide all expandable content
            document.querySelectorAll('.expandable-content').forEach(content => {{
                content.classList.remove('show');
            }});
            
            // Activate current row if it wasn't active before
            if (!isCurrentlyActive) {{
                currentRow.classList.add('active');
                if (content) {{
                    content.classList.add('show');
                }}
            }}
        }}
        
        function showCropImage(imageData, cropName) {{
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            modalImg.src = imageData;
            modalImg.alt = cropName;
            modal.style.display = 'block';
            
            // Reset zoom state
            modalImg.style.transform = 'translate(-50%, -50%) scale(1)';
            modalImg.classList.remove('zoomed');
            
            // Show hint
            const hint = document.getElementById('zoomHint');
            hint.classList.add('show');
            setTimeout(() => hint.classList.remove('show'), 3000);
        }}
        
        // Keyboard support
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                document.querySelectorAll('.summary-table tr').forEach(row => {{
                    row.classList.remove('active');
                }});
                document.querySelectorAll('.expandable-content').forEach(content => {{
                    content.classList.remove('show');
                }});
                // Close image modal
                document.getElementById('imageModal').style.display = 'none';
            }}
        }});
        
        // Image click event
        document.addEventListener('click', function(e) {{
            if (e.target.tagName === 'IMG' && e.target.closest('.image-container')) {{
                const modal = document.getElementById('imageModal');
                const modalImg = document.getElementById('modalImage');
                modalImg.src = e.target.src;
                modalImg.alt = e.target.alt;
                modal.style.display = 'block';
                
                // Reset zoom state
                modalImg.style.transform = 'translate(-50%, -50%) scale(1)';
                modalImg.classList.remove('zoomed');
                
                // Show hint
                const hint = document.getElementById('zoomHint');
                hint.classList.add('show');
                setTimeout(() => hint.classList.remove('show'), 3000);
            }}
        }});
        
        // Close modal
        document.getElementById('imageModal').addEventListener('click', function(e) {{
            if (e.target === this || e.target.classList.contains('modal-close')) {{
                this.style.display = 'none';
            }}
        }});
        
        // Image zoom and drag functionality
        let scale = 1;
        let isDragging = false;
        let startX, startY, translateX = 0, translateY = 0;
        
        const modalImg = document.getElementById('modalImage');
        
        // Mouse wheel zoom - centered on mouse position
        modalImg.addEventListener('wheel', function(e) {{
            e.preventDefault();
            
            // Get image position and mouse position
            const rect = this.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;
            
            // Calculate mouse position relative to image center
            const mouseX = e.clientX - centerX;
            const mouseY = e.clientY - centerY;
            
            // Set zoom origin to mouse position
            this.style.transformOrigin = `${{50 + (mouseX / rect.width) * 100}}% ${{50 + (mouseY / rect.height) * 100}}%`;
            
            // Zoom
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            scale = Math.max(0.5, Math.min(3, scale * delta));
            
            // Apply zoom
            this.style.transform = `translate(-50%, -50%) scale(${{scale}})`;
            this.classList.add('zoomed');
        }});
        
        // Double-click to reset
        modalImg.addEventListener('dblclick', function(e) {{
            e.preventDefault();
            scale = 1;
            translateX = 0;
            translateY = 0;
            this.style.transform = 'translate(-50%, -50%) scale(1)';
            this.style.transformOrigin = 'center center';
            this.classList.remove('zoomed');
        }});
        
        // Drag start
        modalImg.addEventListener('mousedown', function(e) {{
            if (scale > 1) {{
                isDragging = true;
                startX = e.clientX;
                startY = e.clientY;
                this.style.cursor = 'grabbing';
            }}
        }});
        
        // Dragging
        document.addEventListener('mousemove', function(e) {{
            if (isDragging) {{
                e.preventDefault();
                const deltaX = e.clientX - startX;
                const deltaY = e.clientY - startY;
                
                translateX += deltaX;
                translateY += deltaY;
                
                // Apply drag
                modalImg.style.transform = `translate(-50%, -50%) scale(${{scale}}) translate(${{translateX}}px, ${{translateY}}px)`;
                
                startX = e.clientX;
                startY = e.clientY;
            }}
        }});
        
        // Drag end
        document.addEventListener('mouseup', function() {{
            if (isDragging) {{
                isDragging = false;
                modalImg.style.cursor = 'grab';
            }}
        }});
    </script>
</body>
</html>
"""
        
        return html_content
    
    def save_report(self, html_content: str) -> Path:
        """Save HTML report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"batch_report_{timestamp}.html"
        report_path = self.output_dir / report_name
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Fixed layout HTML report saved: {report_path}")
        return report_path
    
    def generate_report(self) -> Path:
        """Generate complete report"""
        print("🚀 Starting fixed layout batch test report generation...")
        
        # 1. Load data
        self.load_csv_data()
        
        # 2. Find related files
        self.find_related_files()
        
        # 3. Generate HTML
        html_content = self.generate_html_report()
        
        # 4. Save report
        report_path = self.save_report(html_content)
        
        print(f"🎉 Fixed layout report generation completed!")
        print(f"📁 Report location: {report_path}")
        print(f"🌐 Can be opened directly in browser")
        print(f"💡 Click table rows to view details, press ESC to close expanded content")
        
        return report_path


def main():
    parser = argparse.ArgumentParser(description='Generate fixed layout HTML report for batch test results')
    parser.add_argument('csv_path', help='Path to batch_summary CSV file')
    parser.add_argument('--output-dir', help='Output directory', default='visualization/reports/html')
    parser.add_argument('--model', help='Model name to display, e.g., Gemini 2.5 Flash Lite', default=None)
    
    args = parser.parse_args()
    
    generator = BatchReportGenerator(args.csv_path, args.output_dir, model_name=args.model)
    generator.generate_report()


if __name__ == "__main__":
    main()
