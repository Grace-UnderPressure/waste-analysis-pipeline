"""
Gemini Prompt Builder Module
Transforms CLIP outputs and crop images into prompts suitable for Gemini inference.

Features:
1) Build attribute-understanding prompts
2) Build multi-turn Q&A prompts
3) Integrate CLIP classification results as inference context
4) Support multiple prompt templates
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from clip_results_analyzer import CLIPResultsAnalyzer, create_clip_results_analyzer
"""
Note: Unused helpers removed for VM bundle. Adapter uses build_unified_prompt only.
"""

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiPromptBuilder:
    """Gemini Prompt builder integrated with CLIP results manager"""
    
    def __init__(self, 
                 label_config_path: Optional[str] = None,
                 clip_results_manager: Optional[CLIPResultsAnalyzer] = None,
                 save_prompts: bool = False,
                 prompt_save_dir: str = "prompts"):
        """
        Initialize the prompt builder

        Args:
            label_config_path: Path to label config for classification system context
            clip_results_manager: CLIP results analyzer instance
            save_prompts: Enable prompt saving to files
            prompt_save_dir: Directory to save prompts
        """
        self.label_config_path = label_config_path
        self.label_system = None
        self.clip_analyzer = clip_results_manager or create_clip_results_analyzer()
        self.save_prompts = save_prompts
        self.prompt_save_dir = prompt_save_dir
        
        # Context for prompt file naming
        self._current_scan_id = None
        self._current_crop_id = None
        
        # Load label system (if provided)
        if label_config_path and os.path.exists(label_config_path):
            self._load_label_system()
        
        # Setup prompt saving
        if self.save_prompts:
            os.makedirs(self.prompt_save_dir, exist_ok=True)
            logger.info(f"✅ Prompt saving enabled: {self.prompt_save_dir}")
    
    def _load_label_system(self):
        """Load label classification system"""
        try:
            with open(self.label_config_path, 'r', encoding='utf-8') as f:
                self.label_system = json.load(f)
            logger.info(f"✅ Label system loaded: {len(self.label_system)} categories")
        except Exception as e:
            logger.error(f"⚠️  Failed to load label system: {e}")
            self.label_system = None
    
    def build_unified_prompt(self,
                               clip_results: List[Dict],
                               include_statistics: bool = True,
                               include_label_context: bool = True,
                               uncertainty_focus: bool = True,
                               original_image_path: Optional[str] = None,
                               crop_path: Optional[str] = None) -> str:
        """
        Unified prompt builder that integrates all features

        Args:
            clip_results: List of CLIP results (single or multiple)
            include_statistics: Include CLIP statistics analysis
            include_label_context: Include label-system context
            uncertainty_focus: Emphasize uncertainty
            original_image_path: Path to original image for context
            crop_path: Path to crop image for reference

        Returns:
            str: Full prompt text
        """
        try:
            # Auto-detect mode
            is_single_crop = len(clip_results) == 1
            prompt_parts = []
            
            # 1. System instruction
            if is_single_crop:
                prompt_parts.append(
                    "You are an AI assistant specialized in waste classification and analysis. "
                    "Please analyze this cropped image of a potential waste object."
                )
            else:
                prompt_parts.append(
                    "You are an AI assistant specialized in waste classification and analysis. "
                    f"Please analyze these {len(clip_results)} cropped images of potential waste objects."
                )
            
            # 1.5. Image reference and context (if original image provided)
            if original_image_path:
                crop_filename = os.path.basename(crop_path) if crop_path else "crop_image"
                original_filename = os.path.basename(original_image_path)
                
                prompt_parts.append(
                    f"\nIMAGE REFERENCE:\n"
                    f"• First image: CROP from {crop_filename} - The specific object/region to analyze\n"
                    f"• Second image: ORIGINAL from {original_filename} - Full scene context for reference\n"
                    f"\nCONTEXT: You are analyzing crops from this original image: {original_image_path}\n"
                    "Please consider the overall scene context, lighting conditions, and scale "
                    "when analyzing each crop. The original image provides important context "
                    "for understanding the objects' environment and relationships.\n"
                    "Focus your analysis on the first image (crop) while using the second image (original) for context."
                )
            
            # 2. Multi-object cognition guidance
            if is_single_crop:
                prompt_parts.append(
                    "\nCROP ANALYSIS APPROACH:\n"
                    "This crop may contain multiple elements. Please identify:\n"
                    "• **Primary Object**: The main, most complete, or most classifiable item\n"
                    "• **Secondary Elements**: Any partial objects, background items, or additional waste\n"
                    "Your classification should focus on the primary object while noting secondary elements."
                )
            else:
                prompt_parts.append(
                    "\nBATCH ANALYSIS APPROACH:\n"
                    f"You will analyze {len(clip_results)} crop images. For each crop:\n"
                    "• Identify the PRIMARY OBJECT (main classification target)\n"
                    "• Note any SECONDARY ELEMENTS (partial objects, background items)\n"
                    "• Provide structured analysis for the primary object\n"
                    "• Consider relationships between crops if relevant\n"
                    "\nPlease analyze each crop systematically and provide consistent analysis structure."
                )
            
            # 3. CLIP statistics context (if enabled)
            if include_statistics:
                if is_single_crop:
                    # Single-crop statistics
                    statistics = self.clip_analyzer.analyze_single_result(clip_results[0])
                    enhanced_context = self.clip_analyzer._generate_single_crop_enhanced_context(
                        clip_results[0], statistics, 0
                    )
                    prompt_parts.append(enhanced_context)
                else:
                    # Batch statistics
                    statistics_list = self.clip_analyzer.analyze_statistics_batch(clip_results)
                    enhanced_context = self.clip_analyzer.generate_enhanced_prompt_context(
                        clip_results, statistics_list
                    )
                    prompt_parts.append(enhanced_context)
            else:
                # Basic CLIP result description
                if is_single_crop:
                    clip_description = self.clip_analyzer.format_for_gemini_prompt([clip_results[0]])
                else:
                    clip_description = self.clip_analyzer.format_for_gemini_prompt(clip_results)
                prompt_parts.append(clip_description)
            
            # 4. Label-system context
            if include_label_context:
                context = self._build_label_context_summary()
                if context:
                    prompt_parts.append(f"\nFor reference, the waste classification system includes:{context}")
            
            # 5. Analysis tasks definition
            if is_single_crop:
                # Single-crop analysis tasks
                prompt_parts.append(
                    "\nPlease provide structured analysis including:"
                    "\n**PRIMARY OBJECT ANALYSIS**:"
                    "\n- Material composition (plastic, metal, paper, glass, etc.)"
                    "\n- Physical characteristics (shape, size, color, condition)"
                    "\n- Likely function or original purpose"
                    "\n- Current state (clean, dirty, damaged, intact)"
                    "\n- Completeness and visibility in the crop"
                    "\n"
                    "\n**SECONDARY ELEMENTS** (if present):"
                    "\n- Other visible objects or portions"
                    "\n- Their relationship to the primary object"
                    "\n- Potential impact on classification"
                    "\n"
                    "\n**OVERALL ASSESSMENT**:"
                    "\n- Primary object category determination"
                    "\n- Recyclability assessment if possible"
                )
            else:
                # Batch analysis tasks
                prompt_parts.append(
                    "\nBATCH ANALYSIS REQUIREMENTS:\n"
                    f"For each of the {len(clip_results)} crops, provide:\n"
                    "\n**CROP [X] - PRIMARY OBJECT ANALYSIS**:"
                    "\n• Material Analysis: What material(s) is this object made of?"
                    "\n• Physical Characteristics: Shape, size, condition, and notable features"
                    "\n• Category Assessment: Based on visual evidence and CLIP context, what waste category?"
                    "\n• Completeness: How much of the object is visible and analyzable?"
                    "\n"
                    "\n**CROP [X] - SECONDARY ELEMENTS**:"
                    "\n• Note any other visible objects/portions"
                    "\n• Assess their relationship to the primary object"
                    "\n• Consider their impact on classification"
                    "\n"
                    "\n**CROP [X] - OVERALL ASSESSMENT**:"
                    "\n• Primary object category determination"
                    "\n• Confidence in classification"
                    "\n• Recyclability assessment if possible"
                )
            
            # 6. Uncertainty focus and confidence criteria
            if uncertainty_focus:
                if is_single_crop:
                    prompt_parts.append(
                        "\n- Confidence level in your assessment"
                        "\n- Any ambiguities or uncertainties you observe"
                        "\n- Alternative possibilities if classification is unclear"
                    )
                else:
                    prompt_parts.append(
                        "\n• Any uncertainties or ambiguities observed"
                        "\n• Alternative classification possibilities"
                    )
                
                # Confidence grade criteria
                prompt_parts.append(
                    "\nCONFIDENCE GRADE CRITERIA:\n"
                    "A (High): Clear visual features, strong classification confidence, consistent with scene context, complete object visible\n"
                    "B (Medium): Generally clear features, moderate confidence, mostly consistent context, object mostly identifiable\n"
                    "C (Low): Unclear features or poor quality, low confidence, some context inconsistencies, partial object visibility\n"
                    "D (Unreliable): Very poor quality/unclear, very low confidence, inconsistent context, incomplete/occluded object"
                )
            
            # 7. Output format guidance - standardized
            if is_single_crop:
                prompt_parts.append(
                    "\nOUTPUT FORMAT:\n"
                    "Structure your response with clear sections:\n"
                    "**PRIMARY OBJECT ANALYSIS**: Focus on the main classifiable item\n"
                    "**SECONDARY ELEMENTS**: Note other visible objects/portions (if present)\n"
                    "**OVERALL ASSESSMENT**: Category determination and confidence\n"
                    "\nBe specific about visual observations and honest about any uncertainties."
                )
            else:
                prompt_parts.append(
                    "\nOUTPUT FORMAT:\n"
                    f"Analyze all {len(clip_results)} crops systematically with this structure:\n"
                    "• **CROP [X] - PRIMARY OBJECT ANALYSIS**: Main classification subject\n"
                    "• **CROP [X] - SECONDARY ELEMENTS**: Other visible objects/portions (if present)\n"
                    "• **CROP [X] - OVERALL ASSESSMENT**: Category and confidence determination\n"
                    "\nMaintain consistent structure for each crop. Be specific about visual details and explicit about uncertainties."
                )
            
            # Build final prompt
            final_prompt = "\n".join(prompt_parts)
            
            # Save prompt if enabled
            if self.save_prompts:
                self._save_prompt_to_file(final_prompt)
            
            return final_prompt
            
        except Exception as e:
            logger.error(f"Unified prompt build failed: {e}")
            return ""
    
    
    def _build_label_context_summary(self) -> str:
        """Build a simplified label-system context summary"""
        # Use the latest detailed classification standard as the reference space
        return """
MAIN WASTE CATEGORIES (Reference Classification System):
1. Organic/Putrescible Waste - Food scraps, garden waste, biodegradable materials
2. Paper - Paper packaging, newspapers, magazines, office paper, other paper products
3. Cardboard - Flat and corrugated cardboard packaging, other cardboard materials
4. Composite Materials - Multi-material packaging (milk cartons, juice boxes), small electrical appliances
5. Textiles - Clothing, fabric materials, sanitary textiles (diapers), soiled paper textiles
6. Plastics:
   • Plastic Bags: Garbage bags, shopping bags, food storage bags
   • Plastic Films: Food packaging films, product wrapping, protective materials
   • PET Bottles: Water bottles, soft drink bottles, clear food containers
   • Other Plastic Containers: Milk jugs (HDPE), yogurt containers (PP), squeeze bottles (LDPE)
   • Rigid Plastic Packaging: Food trays, blister packs, plastic boxes
   • Other Plastic Items: Toys, disposable utensils, furniture items
7. Glass - Clear glass packaging, colored glass packaging, other glass items
8. Metals - Steel cans (ferrous), aluminum cans and foil, other ferrous and non-ferrous metals
9. Combustible Materials - Wood packaging, footwear, leather goods, other combustible items
10. Non-Combustible Materials - Non-combustible packaging, other non-combustible items
11. Hazardous Household Waste - Batteries, fluorescent lamps, medical waste, oils, chemicals, gas cylinders
12. Fine Elements - Small particles and debris (under 20mm), very fine elements (under 8mm)

Please use this classification system as your reference when analyzing and categorizing waste objects."""
    
    def _save_prompt_to_file(self, prompt_text: str):
        """Save prompt to file for debugging"""
        try:
            if not self._current_scan_id or self._current_crop_id is None:
                logger.warning("Cannot save prompt: scan_id or crop_id not set")
                return
            
            # Generate filename
            timestamp = int(time.time())
            filename = f"scan_{self._current_scan_id}_crop_{self._current_crop_id:03d}_unified_{timestamp}.txt"
            filepath = os.path.join(self.prompt_save_dir, filename)
            
            # Write prompt with metadata
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Gemini Prompt - Scan {self._current_scan_id}\n")
                f.write(f"# Crop ID: {self._current_crop_id}\n")
                f.write(f"# Type: unified\n")
                f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Length: {len(prompt_text)} characters\n")
                f.write("\n" + "="*80 + "\n\n")
                f.write(prompt_text)
            
            logger.debug(f"Prompt saved: {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save prompt: {e}")


def create_prompt_builder(label_config_path: Optional[str] = None,
                         clip_results_manager: Optional[CLIPResultsAnalyzer] = None,
                         save_prompts: bool = False,
                         prompt_save_dir: str = "prompts") -> GeminiPromptBuilder:
    """
    Factory function to create GeminiPromptBuilder
    
    Args:
        label_config_path: path to label config file
        clip_results_manager: CLIP results analyzer instance
        save_prompts: enable prompt saving
        prompt_save_dir: directory to save prompts
        
    Returns:
        GeminiPromptBuilder: builder instance
    """
    return GeminiPromptBuilder(label_config_path, clip_results_manager, save_prompts, prompt_save_dir)
