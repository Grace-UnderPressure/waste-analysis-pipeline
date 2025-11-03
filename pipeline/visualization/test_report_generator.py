#!/usr/bin/env python3
"""
Test script for batch report generator
"""

import subprocess
import sys
import os
from pathlib import Path

def test_batch_report_generator():
    """Test batch report generator"""
    print("🧪 Testing batch report generator...")
    
    # Change to correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Test single image report
    print("\n📷 Testing single image report...")
    try:
        result = subprocess.run([
            sys.executable, 
            "batch_report_generator.py",
            "../outputs_tests/test_fixed_timing_v2/batch_summary_fixed_test_v2.csv"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Single image report generated successfully")
        else:
            print(f"❌ Single image report generation failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Single image test exception: {e}")
    
    # Test multiple images
    print("\n📷 Testing multiple images report...")
    try:
        result = subprocess.run([
            sys.executable,
            "batch_report_generator.py", 
            "../outputs_tests/run_super_u_optimal_v2/batch_summary_super_u_optimal_v2.csv"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Multiple images report generated successfully")
        else:
            print(f"❌ Multiple images report generation failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Multiple images test exception: {e}")
    
    # Check generated files
    print("\n📁 Checking generated files...")
    reports_dir = Path("reports/html")
    if reports_dir.exists():
        html_files = list(reports_dir.glob("batch_report_*.html"))
        print(f"✅ Found {len(html_files)} batch report files")
        for f in html_files[-2:]:  # Show latest 2
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   📄 {f.name}: {size_mb:.1f}MB")
    else:
        print("❌ No generated files found")

if __name__ == "__main__":
    import os
    test_batch_report_generator()
