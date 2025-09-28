"""
Gemini API inference module
Wrapper around Google Gemini API calls, supporting multimodal (image + text) inference.

Features:
1) Gemini API call wrapper
2) Image + text multimodal inputs
3) Batch inference helper
4) Error handling and retries
5) Optional response caching
"""

import os
import time
import json
import base64
import hashlib
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from PIL import Image
import io

# Import the structured response schemas
from waste_analysis_schema import (
    SingleCropAnalysis, BatchCropAnalysis, 
    create_single_crop_schema, create_batch_crop_schema
)

# Try importing Google AI SDK (new version)
try:
    from google import genai
    GEMINI_AVAILABLE = True
    SDK_VERSION = "new"  # New SDK
except ImportError:
    try:
        import google.generativeai as genai
        GEMINI_AVAILABLE = True
        SDK_VERSION = "legacy"  # Legacy SDK
        print("⚠️  Using legacy SDK, please upgrade: pip install --upgrade google-genai")
    except ImportError:
        GEMINI_AVAILABLE = False
        SDK_VERSION = None
        print("⚠️  Google GenerativeAI SDK not installed. Please run: pip install google-genai")

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiInferencer:
    """Gemini API inferencer"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-1.5-flash",
                 cache_responses: bool = True,
                 cache_dir: str = "gemini_cache",
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 enable_structured_output: bool = False,
                 response_schema: Optional[type] = None,
                 auto_schema_selection: bool = True):
        """
        Initialize the Gemini inferencer
        
        Args:
            api_key: Gemini API key; if None, read from environment
            model_name: Gemini model name to use
            cache_responses: Enable response caching
            cache_dir: Cache directory
            max_retries: Max retry attempts
            retry_delay: Retry delay in seconds
            enable_structured_output: Enable structured output
            response_schema: Pydantic schema class for structured output
            auto_schema_selection: Auto-select schema (single-crop vs batch)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenerativeAI SDK未安装")
        
        self.model_name = model_name
        self.cache_responses = cache_responses
        self.cache_dir = Path(cache_dir)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_structured_output = enable_structured_output
        self.response_schema = response_schema
        self.auto_schema_selection = auto_schema_selection
        
        # If structured output is enabled without an explicit schema, use default
        if self.enable_structured_output and not self.response_schema:
            if self.auto_schema_selection:
                # Default to single-crop schema; can be updated at inference time
                self.response_schema = create_single_crop_schema()
                logger.info("✅ Auto-selected default schema: SingleCropAnalysis")
            else:
                logger.warning("⚠️ Structured output enabled but no schema specified")
        
        # Create cache directory
        if self.cache_responses:
            self.cache_dir.mkdir(exist_ok=True)
        
        # Configure API
        self._setup_api(api_key)
        
        # Initialize model
        self._setup_model()
        
        # Runtime statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "errors": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "avg_response_time_ms": 0,
            "structured_parse_success": 0,
            "structured_parse_failures": 0
        }
    
    def _setup_api(self, api_key: Optional[str]):
        """Configure Gemini API client"""
        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY env var or pass api_key")
        
        # Configure API (new SDK)
        self.client = genai.Client(api_key=self.api_key)
        logger.info("✅ Gemini API configured")
    
    def _setup_model(self):
        """Initialize Gemini model configuration"""
        try:
            # Generation parameters
            self.generation_config = {
                "temperature": 0.1,  # Low temperature for more deterministic output
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 4096,
            }
            
            # Structured output configuration
            if self.enable_structured_output and self.response_schema:
                self.generation_config.update({
                    "response_mime_type": "application/json",
                    "response_schema": self.response_schema
                })
                logger.info(f"✅ Structured output enabled: {self.response_schema.__name__}")
            
            # Safety settings
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT", 
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            # New SDK does not need model pre-initialization
            self.model = None
            logger.info(f"✅ Gemini configuration ready: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Gemini model initialization failed: {e}")
            raise
    
    def _get_cache_key(self, prompt: str, image_path: Optional[str] = None, original_image_path: Optional[str] = None) -> str:
        """Generate cache key"""
        content = prompt
        if image_path:
            # Include image file mtime and size
            try:
                stat = os.stat(image_path)
                content += f"|{image_path}|{stat.st_mtime}|{stat.st_size}"
            except:
                content += f"|{image_path}"
        
        if original_image_path:
            # Include original image file mtime and size
            try:
                stat = os.stat(original_image_path)
                content += f"|{original_image_path}|{stat.st_mtime}|{stat.st_size}"
            except:
                content += f"|{original_image_path}"
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load response from cache"""
        if not self.cache_responses:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.debug(f"🔄 Cache hit: {cache_key[:8]}...")
                self.stats["cache_hits"] += 1
                return cached_data
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, response_data: Dict):
        """Save response to cache"""
        if not self.cache_responses:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(response_data, f)
            logger.debug(f"💾 Response cached: {cache_key[:8]}...")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def _prepare_image(self, image_path: str) -> Any:
        """Prepare image input, with Files API support for large files"""
        try:
            # Check file size
            file_size = os.path.getsize(image_path)
            use_files_api = file_size > 20 * 1024 * 1024  # 20MB threshold
            
            if use_files_api:
                # Use Files API for large files
                logger.info(f"File size {file_size/1024/1024:.1f}MB, using Files API")
                uploaded_file = self.client.files.upload(file=image_path)
                logger.debug(f"File uploaded: {uploaded_file.name}")
                return uploaded_file
            else:
                # Use PIL image directly
                image = Image.open(image_path)
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                return image
            
        except Exception as e:
            logger.error(f"Image preparation failed {image_path}: {e}")
            raise
    
    def infer_single(self, 
                    prompt: str, 
                    image_path: Optional[str] = None,
                    original_image_path: Optional[str] = None,
                    **kwargs) -> Dict:
        """
        Single inference call
        
        Args:
            prompt: Text prompt
            image_path: Optional crop image file path
            original_image_path: Optional original image file path for context
            **kwargs: Additional generation parameters
            
        Returns:
            Dict: Result with response text, parsed data (if any), and metadata
        """
        self.stats["total_requests"] += 1
        
        # Check cache (include original image in cache key)
        cache_key = self._get_cache_key(prompt, image_path, original_image_path)
        cached_response = self._load_from_cache(cache_key)
        if cached_response:
            return cached_response
        
        # Prepare inputs
        inputs = [prompt]
        if image_path:
            try:
                image = self._prepare_image(image_path)
                inputs.append(image)
            except Exception as e:
                logger.error(f"Image processing failed: {e}")
                return {
                    "success": False,
                    "error": f"Image processing failed: {e}",
                    "response": None,
                    "metadata": {"image_path": image_path}
                }
        
        # Add original image if provided and different from crop
        if original_image_path and original_image_path != image_path:
            try:
                original_image = self._prepare_image(original_image_path)
                inputs.append(original_image)
            except Exception as e:
                logger.warning(f"Original image processing failed: {e}")
                # Continue without original image
        
        # Execute inference with retries
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Sending request to Gemini (attempt {attempt + 1}/{self.max_retries})")
                
                # Record start time
                start_time = time.time()
                
                # Call Gemini API (new SDK)
                if len(inputs) == 1:
                    # Text-only
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=inputs[0],
                        config=self.generation_config
                    )
                else:
                    # Multimodal input
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=inputs,
                        config=self.generation_config
                    )
                
                # Measure response time
                response_time_ms = (time.time() - start_time) * 1000
                
                # Validate response
                if hasattr(response, 'text') and response.text:
                    response_text = response.text
                    
                    # Token usage
                    input_tokens = 0
                    output_tokens = 0
                    total_tokens = 0
                    
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        usage = response.usage_metadata
                        input_tokens = getattr(usage, 'prompt_token_count', 0)
                        output_tokens = getattr(usage, 'candidates_token_count', 0)
                        total_tokens = getattr(usage, 'total_token_count', input_tokens + output_tokens)
                    
                    # Update stats
                    self.stats["total_input_tokens"] += input_tokens
                    self.stats["total_output_tokens"] += output_tokens
                    self.stats["total_tokens"] += total_tokens
                    
                    # Update average response time
                    if self.stats["api_calls"] > 0:
                        self.stats["avg_response_time_ms"] = (
                            (self.stats["avg_response_time_ms"] * (self.stats["api_calls"] - 1) + response_time_ms) 
                            / self.stats["api_calls"]
                        )
                    else:
                        self.stats["avg_response_time_ms"] = response_time_ms
                    
                    result = {
                        "success": True,
                        "error": None,
                        "response": response_text,
                        "metadata": {
                            "model": self.model_name,
                            "image_included": image_path is not None,
                            "original_image_included": original_image_path is not None and original_image_path != image_path,
                            "structured_output": self.enable_structured_output,
                            "response_time_ms": round(response_time_ms, 2),
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                            "parse_status": "none"
                        }
                    }
                    
                    # Structured output parsing
                    if self.enable_structured_output:
                        if hasattr(response, 'parsed') and response.parsed is not None:
                            result["parsed"] = response.parsed
                            self.stats["structured_parse_success"] += 1
                            logger.info(f"✅ Structured parse success: {type(response.parsed).__name__}")
                            # Mark parse source
                            try:
                                result["metadata"]["parse_status"] = "sdk_parsed"
                            except Exception:
                                pass
                        else:
                            logger.warning("Structured output enabled but no parsed data returned")
                            # Try manual JSON parsing from response text
                            try:
                                import json
                                import re
                                
                                # Clean/repair JSON text
                                cleaned_text = response_text.strip()
                                
                                # Try to find a complete JSON object
                                if cleaned_text.startswith('{'):
                                    # Find matching closing brace
                                    brace_count = 0
                                    end_pos = 0
                                    for i, char in enumerate(cleaned_text):
                                        if char == '{':
                                            brace_count += 1
                                        elif char == '}':
                                            brace_count -= 1
                                            if brace_count == 0:
                                                end_pos = i + 1
                                                break
                                    
                                    if end_pos > 0:
                                        cleaned_text = cleaned_text[:end_pos]
                                
                                # Parse the cleaned JSON
                                parsed_data = json.loads(cleaned_text)
                                result["parsed"] = parsed_data
                                self.stats["structured_parse_success"] += 1
                                logger.info("✅ Manual JSON parse success")
                                try:
                                    result["metadata"]["parse_status"] = "manual_parsed"
                                except Exception:
                                    pass
                            except Exception as e:
                                logger.warning(f"Manual JSON parse failed: {e}")
                                self.stats["structured_parse_failures"] += 1
                                # Fallback minimal structure
                                try:
                                    # Minimal placeholder structure
                                    fallback_data = {
                                        "total_crops": 1,
                                        "crop_analyses": [{
                                            "region_id": 0,
                                            "crop_path": "unknown",
                                            "primary_object_analysis": {
                                                "material_composition": "Unknown",
                                                "shape": "Unknown",
                                                "size": "Unknown",
                                                "color": "Unknown",
                                                "condition": "Unknown",
                                                "likely_function_or_original_purpose": "Unknown",
                                                "current_state": "Unknown",
                                                "completeness_and_visibility_in_crop": "Unknown"
                                            },
                                            "secondary_elements": {
                                                "other_visible_objects_or_portions": [],
                                                "relationship_to_primary_object": "Unknown",
                                                "potential_impact_on_classification": "Unknown"
                                            },
                                            "overall_assessment": {
                                                "primary_object_category_determination": "Unknown",
                                                "recyclability_assessment_if_possible": "Unknown",
                                                "confidence_level_in_assessment": "D",
                                                "ambiguities_or_uncertainties_observed": ["JSON parsing failed"],
                                                "alternative_possibilities_if_unclear": []
                                            }
                                        }],
                                        "batch_summary": None
                                    }
                                    result["parsed"] = fallback_data
                                    logger.info("✅ Used fallback structured data")
                                    try:
                                        result["metadata"]["parse_status"] = "fallback"
                                        # Keep a short preview for display when needed
                                        result["metadata"]["raw_preview"] = (response_text or "")[:300]
                                    except Exception:
                                        pass
                                except Exception as e2:
                                    logger.warning(f"Fallback also failed: {e2}")
                                    result["parsed"] = None
                    logger.info(f"✅ Gemini inference success (length: {len(response_text)})")
                    self.stats["api_calls"] += 1
                else:
                    error_msg = "No valid response returned"
                    logger.error(error_msg)
                    result = {
                        "success": False,
                        "error": error_msg,
                        "response": None,
                        "metadata": {"raw_response": str(response)}
                    }
                
                # Cache result
                self._save_to_cache(cache_key, result)
                return result
                
            except Exception as e:
                logger.error(f"Gemini API call failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    # Final attempt failed
                    self.stats["errors"] += 1
                    result = {
                        "success": False,
                        "error": f"API call failed: {e}",
                        "response": None,
                        "metadata": {"attempts": self.max_retries}
                    }
                    return result
    
    def infer_batch(self, 
                   requests: List[Dict],
                   batch_delay: float = 0.5) -> List[Dict]:
        """
        Batch helper that iterates over requests and calls infer_single
        
        Args:
            requests: List of {prompt, image_path}
            batch_delay: Delay between calls
            
        Returns:
            List[Dict]: Results list
        """
        results = []
        
        logger.info(f"🔄 Starting batch inference: {len(requests)} requests")
        
        for i, request in enumerate(requests):
            prompt = request.get('prompt', '')
            image_path = request.get('image_path')
            original_image_path = request.get('original_image_path')
            
            logger.debug(f"Processing request {i+1}/{len(requests)}")
            
            result = self.infer_single(prompt, image_path, original_image_path)
            result['request_index'] = i
            results.append(result)
            
            # Delay to avoid rate limiting
            if i < len(requests) - 1 and batch_delay > 0:
                time.sleep(batch_delay)
        
        # Summarize success count
        success_count = sum(1 for r in results if r['success'])
        logger.info(f"✅ Batch inference finished: {success_count}/{len(requests)} success")
        
        return results
    
    def get_stats(self) -> Dict:
        """Get runtime statistics"""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear on-disk cache"""
        if self.cache_responses and self.cache_dir.exists():
            try:
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                logger.info("🗑️  Gemini response cache cleared")
            except Exception as e:
                logger.error(f"Cache clear failed: {e}")
    
    def test_connection(self) -> bool:
        """Quick API connectivity test"""
        try:
            test_prompt = "Hello, this is a test message. Please respond with 'Connection successful.'"
            result = self.infer_single(test_prompt)
            
            if result['success']:
                logger.info("✅ Gemini API connectivity test passed")
                return True
            else:
                logger.error(f"❌ Gemini API connectivity test failed: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Gemini API connectivity test error: {e}")
            return False

    def select_schema_for_batch(self, batch_size: int) -> type:
        """
        Select schema based on batch size
        
        Args:
            batch_size: Number of crops
            
        Returns:
            type: Selected schema class
        """
        if batch_size == 1:
            schema = create_single_crop_schema()
            logger.info(f"✅ Selected single-crop schema: {schema.__name__}")
        else:
            schema = create_batch_crop_schema()
            logger.info(f"✅ Selected batch schema: {schema.__name__} (processing {batch_size} crops)")
        
        return schema
    
    def update_schema_for_inference(self, schema: type):
        """
        Update schema used for inference
        
        Args:
            schema: New schema class
        """
        self.response_schema = schema
        # Reconfigure generation parameters
        self._setup_model()
        logger.info(f"✅ Schema updated: {schema.__name__}")


def create_gemini_inferencer(api_key: Optional[str] = None, **kwargs) -> GeminiInferencer:
    """
    Factory to create GeminiInferencer
    
    Args:
        api_key: API key
        **kwargs: Additional parameters
        
    Returns:
        GeminiInferencer: Instance
    """
    return GeminiInferencer(api_key=api_key, **kwargs)


# Simple test and example
if __name__ == "__main__":
    # Connectivity test
    try:
        inferencer = create_gemini_inferencer()
        
        # Test connection
        if inferencer.test_connection():
            print("✅ Gemini API connection OK")
            
            # Simple text inference test
            test_prompt = (
                "Please analyze this text and tell me what type of waste this might be: "
                "'a clear plastic bottle with a blue cap'"
            )
            
            result = inferencer.infer_single(test_prompt)
            if result['success']:
                print("=== Text Inference Result ===")
                print(result['response'])
                
                # Show stats
                print("\n=== Stats ===")
                stats = inferencer.get_stats()
                for key, value in stats.items():
                    print(f"{key}: {value}")
            else:
                print(f"Inference failed: {result['error']}")
        else:
            print("❌ Gemini API connection failed")
            
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        print("Please ensure:")
        print("1. google-generativeai installed: pip install google-generativeai")
        print("2. GEMINI_API_KEY environment variable is set")