#!/bin/bash
# Example script to generate visualization reports

set -e  # Exit on error

echo "============================================"
echo "Visualization Report Generation Examples"
echo "============================================"
echo

# Navigate to visualization directory
cd "$(dirname "$0")/.."

# Example 1: Generate report from Gemini 2.5 Flash demo
echo "Example 1: Generate report from Gemini 2.5 Flash"
echo "----------------------------------------"

python batch_report_generator.py \
    ../demo_results/gemini-2.5-flash/outputs/batch_summary_visualization_demo.csv \
    --output-dir reports \
    --model "Gemini 2.5 Flash"

echo "✅ Report generated in: reports/"
echo
echo

# Example 2: Generate report from Gemini 2.5 Flash Lite demo
echo "Example 2: Generate report from Gemini 2.5 Flash Lite"
echo "----------------------------------------"

python batch_report_generator.py \
    ../demo_results/gemini-2.5-flash-lite/outputs/batch_summary_visualization_demo.csv \
    --output-dir reports \
    --model "Gemini 2.5 Flash Lite"

echo "✅ Report generated in: reports/"

echo
echo "============================================"
echo "All reports generated successfully!"
echo "============================================"
