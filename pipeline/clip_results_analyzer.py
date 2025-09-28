"""
CLIP Results Analyzer - Unified Version
Integrates result management, statistical analysis, and intelligent formatting.

Responsibilities:
1. Data Management Layer: Save, load, and cache CLIP results
2. Statistical Analysis Layer: Compute multi-dimensional statistical features
3. Formatting Layer: Generate intelligent prompts based on statistics

Key improvements:
- Unified data flow
- Automatic caching of statistical features
- Intelligent prompt strategies based on statistics
- Full compatibility with existing code
"""

import json
import os
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CLIPStatistics:
    """Data class for CLIP statistical features"""
    # Core classification info
    best_label: Tuple[str, str]  # (main, sub)
    best_score: float
    confidence_level: str  # high/medium/low/very-low
    
    # Competition and distribution
    top_candidates: List[Dict[str, Any]]
    dominance_coef: float  # dominance coefficient
    effective_vc: float  # effective variation coefficient
    score_gradient: float  # score gradient
    
    # Confidence metrics
    abs_confidence: float  # absolute confidence
    rel_confidence: float  # relative confidence
    
    # Analysis strategy
    analysis_strategy: str  # focus_validation/dual_comparison/multi_candidate/descriptive_analysis
    prompt_template: str  # corresponding prompt template type


class CLIPResultsAnalyzer:
    """
    CLIP Results Analyzer - Unified version
    Integrates data management, statistical analysis, and intelligent formatting
    """
    
    def __init__(self, 
                 cache_dir: str = "clip_results_cache",
                 enable_smart_cache: bool = True,
                 enable_statistics_cache: bool = True,
                 clip_threshold: float = 0.25):
        """
        Initialize CLIP results analyzer
        
        Args:
            cache_dir: cache directory
            enable_smart_cache: enable content-aware caching
            enable_statistics_cache: enable statistics caching
            clip_threshold: CLIP recognition threshold
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.enable_smart_cache = enable_smart_cache
        self.enable_statistics_cache = enable_statistics_cache
        self.clip_threshold = clip_threshold
        
        # Statistics cache
        self.statistics_cache = {}
        
        # Prompt strategy templates
        self._load_prompt_templates()
    
    # ========================================
    # Core API - One-stop analysis and formatting
    # ========================================
    
    def analyze_and_format_for_gemini(self, 
                                    image_path: str,
                                    clip_results: List[Dict],
                                    config: Dict = None,
                                    force_save: bool = False) -> Dict:
        """
        One-stop analysis and formatting, returning all data needed by Gemini
        
        Args:
            image_path: original image path
            clip_results: CLIP classification results
            config: configuration
            force_save: force saving regardless of cache
            
        Returns:
            Dict: all analysis results and formatted data
        """
        # 1. Save/load CLIP results
        integration_result = self.integrate_with_existing_pipeline(
            image_path, clip_results, config, force_save
        )
        
        raw_results = integration_result["gemini_ready_data"]["clip_results_raw"]
        
        # 2. Statistical analysis
        statistics_list = self.analyze_statistics_batch(raw_results)
        
        # 3. Intelligent formatting
        enhanced_prompt = self.generate_enhanced_prompt_context(raw_results, statistics_list)
        
        # 4. Strategy summary
        strategies = [stat.analysis_strategy for stat in statistics_list]
        dominant_strategy = max(set(strategies), key=strategies.count)
        
        return {
            "json_path": integration_result["json_path"],
            "cache_hit": integration_result["cache_hit"],
            "raw_results": raw_results,
            "statistics": [self._statistics_to_dict(stat) for stat in statistics_list],
            "enhanced_prompt": enhanced_prompt,
            "dominant_strategy": dominant_strategy,
            "summary": integration_result["gemini_ready_data"]["summary"],
            "metadata": integration_result["gemini_ready_data"]["metadata"]
        }
    
    # ========================================
    # Data management layer - migrated from clip_results_manager
    # ========================================
    
    def save_clip_results(self, 
                         image_path: str,
                         clip_results: List[Dict],
                         config: Dict = None) -> str:
        """
        Save CLIP results to JSON file
        
        Args:
            image_path: original image path
            clip_results: list of CLIP results
            config: config dict
            
        Returns:
            str: path to the saved JSON file
        """
        # Generate unique filename
        image_name = Path(image_path).stem
        results_hash = self._generate_results_hash(clip_results, image_path, config)
        json_filename = f"{image_name}_{results_hash}_clip_results.json"
        json_path = self.cache_dir / json_filename
        
        # Build payload
        save_data = {
            "metadata": {
                "image_path": image_path,
                "image_name": image_name,
                "total_crops": len(clip_results),
                "generation_time": None,
                "config": config or {}
            },
            "clip_results": self._format_results_for_storage(clip_results),
            "summary": self._generate_results_summary(clip_results)
        }
        
        # Save JSON
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ CLIP results saved: {json_path}")
            return str(json_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save CLIP results: {e}")
            raise
    
    def load_clip_results(self, json_path: str) -> Dict:
        """
        Load CLIP results from JSON
        
        Args:
            json_path: JSON path
            
        Returns:
            Dict: data containing metadata and clip_results
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"🔄 CLIP results loaded: {json_path}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to load CLIP results: {e}")
            raise
    
    def find_existing_results(self, image_path: str, config: Dict = None) -> Optional[str]:
        """
        Find existing CLIP results file (supports smart cache)
        
        Args:
            image_path: image path
            config: configuration (for cache matching)
            
        Returns:
            Optional[str]: JSON path if found
        """
        if self.enable_smart_cache:
            return self._find_smart_cached_results(image_path, config)
        else:
            return self._find_simple_cached_results(image_path)
    
    def integrate_with_existing_pipeline(self, 
                                        image_path: str,
                                        existing_clip_results: List[Dict],
                                        config: Dict = None,
                                        force_save: bool = False) -> Dict:
        """
        Convenient method to integrate with existing pipeline (compatibility)
        
        Args:
            image_path: image path
            existing_clip_results: existing CLIP results
            config: configuration
            force_save: force save (ignore cache)
            
        Returns:
            Dict: includes save path and Gemini-formatted data
        """
        json_path = None
        cache_hit = False
        
        if not force_save:
            # Check cache
            json_path = self.find_existing_results(image_path, config)
            cache_hit = json_path is not None
        
        if json_path is None or force_save:
            # Save new results (always save when force_save)
            json_path = self.save_clip_results(image_path, existing_clip_results, config)
            cache_hit = False
        
        # Get Gemini-formatted data
        gemini_data = self.get_results_for_gemini(json_path)
        
        return {
            "json_path": json_path,
            "gemini_ready_data": gemini_data,
            "cache_hit": cache_hit
        }
    
    def get_results_for_gemini(self, json_path: str) -> Dict:
        """
        Get CLIP results formatted for Gemini input (compatibility)
        
        Args:
            json_path: CLIP results JSON path
            
        Returns:
            Dict: formatted data for Gemini
        """
        data = self.load_clip_results(json_path)
        clip_results = data["clip_results"]
        
        return {
            "clip_results_raw": clip_results,
            "clip_description": self.format_for_gemini_prompt(clip_results),
            "summary": data.get("summary", {}),
            "metadata": data.get("metadata", {})
        }
    
    # ========================================
    # Statistical analysis layer - migrated and optimized
    # ========================================
    
    def analyze_statistics_batch(self, clip_results: List[Dict]) -> List[CLIPStatistics]:
        """
        Analyze statistical features for a batch of CLIP results
        
        Args:
            clip_results: list of CLIP results
            
        Returns:
            List[CLIPStatistics]: list of statistical features
        """
        statistics_list = []
        
        for result in clip_results:
            # Check statistics cache
            cache_key = self._get_statistics_cache_key(result)
            
            if self.enable_statistics_cache and cache_key in self.statistics_cache:
                statistics = self.statistics_cache[cache_key]
            else:
                statistics = self.analyze_single_result(result)
                if self.enable_statistics_cache:
                    self.statistics_cache[cache_key] = statistics
            
            statistics_list.append(statistics)
        
        return statistics_list
    
    def analyze_single_result(self, clip_result: Dict) -> CLIPStatistics:
        """
        Analyze statistical features for a single CLIP result
        
        Args:
            clip_result: single CLIP classification result
            
        Returns:
            CLIPStatistics: statistics object
        """
        try:
            # Extract base data
            main_label = clip_result.get('main_label', 'unknown')
            sub_label = clip_result.get('sub_label', 'unknown')
            confidence = clip_result.get('confidence', 0.0)
            top_k = clip_result.get('top_k', [])
            
            # 1. Core classification info
            best_label = (main_label, sub_label)
            best_score = confidence
            confidence_level = self._classify_confidence_level(confidence)
            
            # 2. Effective candidates (top-5)
            top_candidates = top_k[:5] if len(top_k) >= 5 else top_k
            
            # 3. Competition metrics
            dominance_coef = self._calculate_dominance_coefficient(top_k)
            effective_vc = self._calculate_effective_variation_coefficient(top_candidates)
            score_gradient = self._calculate_score_gradient(top_candidates)
            
            # 4. Confidence metrics
            abs_confidence = best_score
            rel_confidence = best_score / self.clip_threshold if self.clip_threshold > 0 else 0.0
            
            # 5. Determine analysis strategy
            analysis_strategy = self._determine_analysis_strategy(
                dominance_coef, abs_confidence, effective_vc
            )
            
            # 6. Select prompt template
            prompt_template = self._select_prompt_template(analysis_strategy)
            
            return CLIPStatistics(
                best_label=best_label,
                best_score=best_score,
                confidence_level=confidence_level,
                top_candidates=top_candidates,
                dominance_coef=dominance_coef,
                effective_vc=effective_vc,
                score_gradient=score_gradient,
                abs_confidence=abs_confidence,
                rel_confidence=rel_confidence,
                analysis_strategy=analysis_strategy,
                prompt_template=prompt_template
            )
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return self._create_default_statistics()
    
    def _calculate_dominance_coefficient(self, top_k: List[Dict]) -> float:
        """Calculate dominance coefficient"""
        if len(top_k) < 2:
            return 1.0
        
        first_score = top_k[0]['score']
        second_score = top_k[1]['score']
        
        if second_score == 0:
            return float('inf')
        
        return first_score / second_score
    
    def _calculate_effective_variation_coefficient(self, top_candidates: List[Dict]) -> float:
        """Calculate effective variation coefficient (based on top-K candidates)"""
        if len(top_candidates) < 2:
            return 0.0
        
        scores = [item['score'] for item in top_candidates]
        mean_score = sum(scores) / len(scores)
        
        if mean_score == 0:
            return 0.0
        
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_score = variance ** 0.5
        
        return std_score / mean_score
    
    def _calculate_score_gradient(self, top_candidates: List[Dict]) -> float:
        """Calculate score gradient"""
        if len(top_candidates) < 2:
            return 0.0
        
        first_score = top_candidates[0]['score']
        last_score = top_candidates[-1]['score']
        
        return (first_score - last_score) / (len(top_candidates) - 1)
    
    def _classify_confidence_level(self, confidence: float) -> str:
        """Simple confidence classification"""
        return f"{confidence:.3f}"  # return numeric string directly
    
    def _determine_analysis_strategy(self, dominance_coef: float, abs_confidence: float, effective_vc: float) -> str:
        """Determine analysis strategy based on statistical features"""
        # Very low confidence - descriptive analysis
        if abs_confidence < 0.03:
            return "descriptive_analysis"
        
        # Strong dominance - focus on top candidate
        if dominance_coef >= 4.0:
            return "focus_validation"
        
        # Moderate competition - compare top candidates  
        if dominance_coef >= 2.0:
            return "dual_comparison"
        
        # High competition - open analysis
        return "multi_candidate"
    
    def _select_prompt_template(self, strategy: str) -> str:
        """Select prompt template type"""
        template_mapping = {
            "focus_validation": "single_dominant",
            "dual_comparison": "dual_competitive", 
            "multi_candidate": "multi_competitive",
            "descriptive_analysis": "low_confidence"
        }
        return template_mapping.get(strategy, "low_confidence")
    
    # ========================================
    # Formatting layer - Enhanced intelligent prompt generation
    # ========================================
    
    def generate_enhanced_prompt_context(self, 
                                       clip_results: List[Dict], 
                                       statistics_list: List[CLIPStatistics]) -> str:
        """
        Generate enhanced prompt context based on statistics
        
        Args:
            clip_results: CLIP results
            statistics_list: list of statistical features
            
        Returns:
            str: enhanced prompt context
        """
        descriptions = []
        
        for i, (result, stats) in enumerate(zip(clip_results, statistics_list)):
            # Generate enhanced description for single crop
            crop_description = self._generate_single_crop_enhanced_context(result, stats, i)
            descriptions.append(crop_description)
        
        return "\n\n".join(descriptions)
    
    def _generate_single_crop_enhanced_context(self, 
                                             clip_result: Dict, 
                                             statistics: CLIPStatistics, 
                                             crop_index: int) -> str:
        """Generate enhanced context for a single crop"""
        template_type = statistics.prompt_template
        template = self.prompt_templates.get(template_type, self.prompt_templates["default"])
        
        # Prepare template params
        params = self._prepare_template_params(clip_result, statistics, crop_index)
        
        try:
            return template.format(**params)
        except Exception as e:
            logger.error(f"Prompt template formatting failed: {e}")
            return self._generate_fallback_context(clip_result, statistics, crop_index)
    
    def _prepare_template_params(self, clip_result: Dict, statistics: CLIPStatistics, crop_index: int) -> Dict:
        """Prepare template parameters"""
        region_id = clip_result.get('region_id', crop_index)
        
        params = {
            'crop_index': crop_index,
            'region_id': region_id,
            'main_label': statistics.best_label[0],
            'sub_label': statistics.best_label[1],
            'best_score': statistics.best_score,
            'dominance_coef': statistics.dominance_coef,
            'abs_confidence': statistics.abs_confidence,
            'rel_confidence': statistics.rel_confidence,
            'effective_vc': statistics.effective_vc,
            'score_gradient': statistics.score_gradient,
            'confidence_level': statistics.confidence_level,
            'analysis_strategy': statistics.analysis_strategy
        }
        
        # Add candidate info
        top_candidates = statistics.top_candidates
        if len(top_candidates) >= 1:
            params.update({
                'first_main': top_candidates[0].get('main_label', 'unknown'),
                'first_sub': top_candidates[0].get('sub_label', 'unknown'),
                'first_score': top_candidates[0].get('score', 0.0)
            })
        
        if len(top_candidates) >= 2:
            params.update({
                'second_main': top_candidates[1].get('main_label', 'unknown'),
                'second_sub': top_candidates[1].get('sub_label', 'unknown'),
                'second_score': top_candidates[1].get('score', 0.0)
            })
        
        if len(top_candidates) >= 3:
            params.update({
                'third_main': top_candidates[2].get('main_label', 'unknown'),
                'third_sub': top_candidates[2].get('sub_label', 'unknown'),
                'third_score': top_candidates[2].get('score', 0.0)
            })
        
        return params
    
    def format_for_gemini_prompt(self, clip_results: List[Dict]) -> str:
        """
        Compatibility method: format CLIP results as a Gemini prompt (kept interface)
        
        Args:
            clip_results: list of CLIP results
            
        Returns:
            str: formatted natural language description
        """
        descriptions = []
        
        for i, result in enumerate(clip_results):
            region_id = result.get('region_id', i)
            main_label = result.get('main_label', 'unknown')
            sub_label = result.get('sub_label', 'unknown')
            confidence = result.get('confidence', 0.0)
            
            # Base classification info
            desc_lines = [f"CROP {region_id}:"]
            desc_lines.append(f"  Primary Classification: {main_label}/{sub_label}")
            desc_lines.append(f"  Confidence Score: {confidence:.3f}")
            
            # Confidence level description
            conf_level = self._get_confidence_level_description(confidence)
            desc_lines.append(f"  Confidence Level: {conf_level}")
            
            # Top-K candidates (if present)
            top_k = result.get('top_k', [])
            if len(top_k) > 1:
                desc_lines.append(f"  Alternative Classifications:")
                for j, alt in enumerate(top_k[1:3]):  # show top-2 alternatives
                    alt_main = alt.get('main_label', 'unknown')
                    alt_sub = alt.get('sub_label', 'unknown')
                    alt_score = alt.get('score', 0.0)
                    desc_lines.append(f"    {j+2}. {alt_main}/{alt_sub} (score: {alt_score:.3f})")
            
            # Unknown case extra explanation
            if main_label == "unknown":
                unknown_info = result.get('unknown_info', {})
                if unknown_info:
                    best_attempt = unknown_info.get('best_attempted_class', ['unknown', 'unknown'])
                    best_score = unknown_info.get('best_attempted_score', 0.0)
                    reason = unknown_info.get('reason', 'unknown')
                    
                    desc_lines.append(f"  Classification Issue:")
                    desc_lines.append(f"    Best Attempt: {best_attempt[0]}/{best_attempt[1]} ({best_score:.3f})")
                    desc_lines.append(f"    Reason for Unknown: {reason}")
            
            descriptions.append("\n".join(desc_lines))
        
        return "\n\n".join(descriptions)
    
    # ========================================
    # Helper methods and templates
    # ========================================
    
    def _load_prompt_templates(self):
        """Load prompt strategy templates"""
        self.prompt_templates = {
            "single_dominant": """
CROP {crop_index} Analysis (ID: {region_id}):
CLIP Statistical Context:
• Top candidate: {main_label}/{sub_label} (score: {best_score:.3f})
• Leading margin: {dominance_coef:.2f}x advantage over next candidate
• Score distribution: Concentrated pattern with clear frontrunner
• Analytical consideration: {main_label} features merit primary attention
• Alternative pathway: Secondary options show minimal relevance

Statistical Background: Concentrated score pattern suggests focused classification pathway.
The leading candidate demonstrates substantial statistical separation from alternatives.
            """,
            
            "dual_competitive": """
CROP {crop_index} Analysis (ID: {region_id}):
CLIP Statistical Context:
• Leading candidate: {first_main}/{first_sub} (score: {first_score:.3f})
• Secondary candidate: {second_main}/{second_sub} (score: {second_score:.3f})
• Competition level: {dominance_coef:.2f}x separation between top options
• Score pattern: Moderate differentiation with competing possibilities
• Comparative consideration: Both {first_main} and {second_main} warrant evaluation

Statistical Background: Moderate score separation indicates two viable classification pathways.
The leading option shows measurable advantage while secondary pathway remains statistically relevant.
            """,
            
            "multi_competitive": """
CROP {crop_index} Analysis (ID: {region_id}):
CLIP Statistical Context:
• Multiple candidates: {first_main}/{first_sub}, {second_main}/{second_sub}, {third_main}/{third_sub}
• Score proximity: Close competition ({dominance_coef:.2f}x minimal separation)
• Distribution pattern: Dispersed scores across categories
• Open assessment: Multiple classification pathways remain viable
• Comprehensive evaluation: Consider material, morphological, and functional evidence

Statistical Background: Dispersed score distribution indicates no dominant classification pathway.
Multiple categories demonstrate comparable statistical relevance requiring comprehensive evaluation.
            """,
            
            "low_confidence": """
CROP {crop_index} Analysis (ID: {region_id}):
CLIP Statistical Context:
• Score ceiling: {best_score:.3f} (below reliable threshold)
• Pattern: No dominant classification emerged
• Analytical approach: Visual characteristics take precedence over automated suggestions
• Independent assessment: Rely on direct morphological observation

Statistical Background: Low score ceiling indicates limited automated classification guidance available.
Visual analysis should proceed independently based on direct morphological characteristics.
            """,
            
            "default": """
CROP {crop_index} Analysis (ID: {region_id}):
• Classification result: {main_label}/{sub_label} (score: {best_score:.3f})
• Confidence level: {confidence_level}

Please conduct visual analysis based on the image and refer to the CLIP guidance above for classification judgment.
            """
        }
    
    def _generate_fallback_context(self, clip_result: Dict, statistics: CLIPStatistics, crop_index: int) -> str:
        """Generate fallback context"""
        region_id = clip_result.get('region_id', crop_index)
        return f"""
CROP {crop_index} Analysis (ID: {region_id}):
• Classification: {statistics.best_label[0]}/{statistics.best_label[1]} (score: {statistics.best_score:.3f})
• Confidence: {statistics.confidence_level}
• Analysis Strategy: {statistics.analysis_strategy}

Please perform visual analysis based on the image and refer to the CLIP hints above.
        """.strip()
    
    def _statistics_to_dict(self, statistics: CLIPStatistics) -> Dict:
        """Convert statistics object to dict"""
        return {
            "best_label": statistics.best_label,
            "best_score": statistics.best_score,
            "confidence_level": statistics.confidence_level,
            "dominance_coef": statistics.dominance_coef,
            "effective_vc": statistics.effective_vc,
            "score_gradient": statistics.score_gradient,
            "abs_confidence": statistics.abs_confidence,
            "rel_confidence": statistics.rel_confidence,
            "analysis_strategy": statistics.analysis_strategy,
            "prompt_template": statistics.prompt_template
        }
    
    def _create_default_statistics(self) -> CLIPStatistics:
        """Create default statistics object"""
        return CLIPStatistics(
            best_label=("unknown", "unknown"),
            best_score=0.0,
            confidence_level="very-low",
            top_candidates=[],
            dominance_coef=1.0,
            effective_vc=0.0,
            score_gradient=0.0,
            abs_confidence=0.0,
            rel_confidence=0.0,
            analysis_strategy="descriptive_analysis",
            prompt_template="default"
        )
    
    def _get_statistics_cache_key(self, clip_result: Dict) -> str:
        """Generate statistics cache key"""
        key_data = [
            str(clip_result.get('region_id', 0)),
            str(clip_result.get('confidence', 0)),
            str(len(clip_result.get('top_k', [])))
        ]
        return hashlib.md5("|".join(key_data).encode()).hexdigest()[:8]
    
    # ========================================
    # Data management helpers (migrated from original clip_results_manager)
    # ========================================
    
    def _format_results_for_storage(self, clip_results: List[Dict]) -> List[Dict]:
        """Format results for storage"""
        simplified = []
        for result in clip_results:
            simplified_result = {
                "region_id": result.get("region_id"),
                "crop_path": result.get("crop_path"),
                "main_label": result.get("main_label"),
                "sub_label": result.get("sub_label"),
                "confidence": result.get("confidence"),
                "top_k": result.get("top_k", [])[:3],  # keep top-3 sub-label candidates
            }
            
            # Extract main category probabilities
            top_k_main_probs = self._extract_main_category_probabilities(
                result.get("top_k", [])
            )
            simplified_result["top_main_categories"] = top_k_main_probs
            
            # Keep classification context
            classification_context = result.get("classification_context", {})
            if classification_context:
                simplified_result["classification_summary"] = {
                    "confidence_level": classification_context.get("classification_confidence", "unknown"),
                    "top_main_classes": classification_context.get("top_main_classes", [])[:3]
                }
            
            # Keep unknown info
            if result.get("main_label") == "unknown":
                simplified_result["unknown_info"] = result.get("unknown_info", {})
            
            # Keep bbox if present
            if "bbox" in result:
                simplified_result["bbox"] = result["bbox"]
            
            simplified.append(simplified_result)
        
        return simplified
    
    def _extract_main_category_probabilities(self, top_k: List[Dict]) -> List[Dict]:
        """Extract main-category probabilities from top_k"""
        main_category_scores = {}
        
        for item in top_k:
            main_cat = item.get("main_label", "unknown")
            score = item.get("score", 0.0)
            
            if main_cat not in main_category_scores:
                main_category_scores[main_cat] = score
            else:
                main_category_scores[main_cat] = max(main_category_scores[main_cat], score)
        
        sorted_main_cats = sorted(
            main_category_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        return [
            {"main_category": main_cat, "max_score": score}
            for main_cat, score in sorted_main_cats
        ]
    
    def _generate_results_summary(self, clip_results: List[Dict]) -> Dict:
        """Generate results summary statistics"""
        total_crops = len(clip_results)
        unknown_count = sum(1 for r in clip_results if r.get('main_label') == 'unknown')
        
        main_categories = {}
        confidence_distribution = {"high": 0, "medium": 0, "low": 0}
        
        for result in clip_results:
            main_label = result.get('main_label', 'unknown')
            confidence = result.get('confidence', 0.0)
            
            main_categories[main_label] = main_categories.get(main_label, 0) + 1
            
            if confidence > 0.7:
                confidence_distribution["high"] += 1
            elif confidence > 0.3:
                confidence_distribution["medium"] += 1
            else:
                confidence_distribution["low"] += 1
        
        return {
            "total_crops": total_crops,
            "unknown_count": unknown_count,
            "identification_rate": (total_crops - unknown_count) / total_crops if total_crops > 0 else 0,
            "main_categories": main_categories,
            "confidence_distribution": confidence_distribution
        }
    
    def _generate_results_hash(self, clip_results: List[Dict], image_path: str = None, config: Dict = None) -> str:
        """Generate unique hash for results"""
        if self.enable_smart_cache and image_path:
            try:
                with open(image_path, 'rb') as f:
                    image_content = f.read()
                image_hash = hashlib.md5(image_content).hexdigest()[:12]
                
                config_str = json.dumps(config or {}, sort_keys=True)
                config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
                
                return f"{image_hash}_{config_hash}"
            except Exception as e:
                logger.warning(f"Smart hash generation failed, falling back: {e}")
        
        # Simple mode
        hash_data = []
        for result in clip_results:
            hash_data.append(f"{result.get('region_id', 0)}_{result.get('main_label', '')}_{result.get('confidence', 0)}")
        
        hash_string = "|".join(hash_data)
        return hashlib.md5(hash_string.encode()).hexdigest()[:8]
    
    def _get_confidence_level_description(self, confidence: float) -> str:
        """Convert confidence numeric value to a level description"""
        if confidence > 0.7:
            return "High"
        elif confidence > 0.4:
            return "Medium" 
        elif confidence > 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def _find_simple_cached_results(self, image_path: str) -> Optional[str]:
        """Simple cache lookup based on filename"""
        image_name = Path(image_path).stem
        pattern = f"{image_name}_*_clip_results.json"
        
        matching_files = list(self.cache_dir.glob(pattern))
        
        if matching_files:
            latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"🔍 Found existing CLIP results: {latest_file}")
            return str(latest_file)
        
        return None
    
    def _find_smart_cached_results(self, image_path: str, config: Dict = None) -> Optional[str]:
        """Content- and config-aware cache lookup"""
        try:
            with open(image_path, 'rb') as f:
                image_content = f.read()
            image_hash = hashlib.md5(image_content).hexdigest()[:12]
            
            config_str = json.dumps(config or {}, sort_keys=True)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
            
            pattern = f"*_{image_hash}_{config_hash}_clip_results.json"
            matching_files = list(self.cache_dir.glob(pattern))
            
            if matching_files:
                latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
                logger.info(f"🎯 Smart cache hit: {latest_file}")
                return str(latest_file)
            
        except Exception as e:
            logger.warning(f"Smart cache lookup failed, downgrade to simple: {e}")
            return self._find_simple_cached_results(image_path)
        
        return None


# ========================================
# Factory functions - compatibility
# ========================================

def create_clip_results_manager(cache_dir: str = "clip_results_cache", 
                               enable_smart_cache: bool = True) -> CLIPResultsAnalyzer:
    """
    Factory function for CLIP results manager (compatibility interface)
    
    Args:
        cache_dir: cache directory
        enable_smart_cache: enable smart cache
        
    Returns:
        CLIPResultsAnalyzer: instance
    """
    return CLIPResultsAnalyzer(cache_dir, enable_smart_cache)


def create_clip_results_analyzer(cache_dir: str = "clip_results_cache",
                                enable_smart_cache: bool = True,
                                enable_statistics_cache: bool = True,
                                clip_threshold: float = 0.25) -> CLIPResultsAnalyzer:
    """
    Factory function for CLIPResultsAnalyzer
    
    Args:
        cache_dir: cache directory
        enable_smart_cache: enable smart cache
        enable_statistics_cache: enable statistics cache
        clip_threshold: CLIP threshold
        
    Returns:
        CLIPResultsAnalyzer: instance
    """
    return CLIPResultsAnalyzer(cache_dir, enable_smart_cache, enable_statistics_cache, clip_threshold)


# ========================================
# Test code
# ========================================

if __name__ == "__main__":
    # Create analyzer
    analyzer = create_clip_results_analyzer()
    
    # Mock CLIP results
    sample_clip_results = [
        {
            "region_id": 0,
            "crop_path": "test_crop_0.jpg",
            "main_label": "unknown",
            "sub_label": "unknown",
            "confidence": 0.146,
            "top_k": [
                {"main_label": "Plastics", "sub_label": "bottle", "score": 0.146},
                {"main_label": "Plastics", "sub_label": "container", "score": 0.036},
                {"main_label": "Plastics", "sub_label": "bag", "score": 0.025}
            ],
            "unknown_info": {
                "best_attempted_class": ["Plastics", "bottle"],
                "best_attempted_score": 0.146,
                "threshold": 0.25,
                "reason": "below_threshold"
            }
        }
    ]
    
    print("🧪 Unified CLIP Results Analyzer Test")
    print("=" * 50)
    
    # Test statistics
    statistics_list = analyzer.analyze_statistics_batch(sample_clip_results)
    print("📊 Statistics:")
    for i, stats in enumerate(statistics_list):
        print(f"  Crop {i}: {stats.analysis_strategy}, dominance={stats.dominance_coef:.2f}")
    
    # Test enhanced prompt generation
    enhanced_prompt = analyzer.generate_enhanced_prompt_context(sample_clip_results, statistics_list)
    print("\n🚀 Enhanced Prompt:")
    print(enhanced_prompt)
    
    print("\n✅ Unified version test done!")