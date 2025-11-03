#!/usr/bin/env python3
"""
CLIP Image-Text Matcher
- CLIP classification module for waste categorization
- Supports main/sub classes including Background and Unknown
- Provides embedding caching, batch optimizations, Unknown fallback
"""

import torch
import clip
from PIL import Image
import json
import os
import numpy as np
import pickle
import hashlib
from typing import List, Dict, Tuple, Optional

class CLIPMatcher:
    """
    CLIP matcher for waste classification.
    
    Capabilities:
    1) Load and cache text embeddings for all prompts
    2) Batch-encode images and compute similarities to prompts
    3) Aggregate scores across prompts for the same label
    4) Unknown fallback based on confidence threshold
    5) Support Background and Unknown classes
    """
    
    def __init__(self, label_config_path: str, device: str = "cuda", 
                 clip_threshold: float = 0.28, top_k: int = 3, 
                 cache_embeddings: bool = True, 
                 embedding_cache_path: str = "clip_embeddings_cache.pkl"):
        """
        Initialize CLIP matcher.
        
        Args:
            label_config_path: label config path (waste_labels.json)
            device: device (cuda/cpu)
            clip_threshold: below this, classify as Unknown
            top_k: return top-k results
            cache_embeddings: cache text embeddings or not
            embedding_cache_path: cache path for text embeddings
        """
        self.device = device
        self.clip_threshold = clip_threshold
        self.top_k = top_k
        self.cache_embeddings = cache_embeddings
        self.embedding_cache_path = embedding_cache_path
        
        # Load label config
        with open(label_config_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        
        # Load CLIP model
        print(f"🔄 Loading CLIP model (ViT-B/32) on device: {device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Prepare prompts and label mapping
        self._prepare_prompts_and_mapping()
        
        # Precompute and cache text embeddings
        self.text_embeddings = self._get_or_compute_text_embeddings()
        
        # Image embedding cache
        self.image_embedding_cache = {}
        
        print(f"✅ CLIP model loaded")
        print(f"   - Total prompts: {len(self.prompt_list)}")
        print(f"   - Main/Sub combinations: {len(self.label_map)}")
        print(f"   - Confidence threshold: {self.clip_threshold}")
        print(f"   - Device: {device}")

    def _prepare_prompts_and_mapping(self):
        """Prepare prompt list and label map"""
        self.prompt_list = []
        self.label_map = []  # [(main, sub)]
        
        for item in self.labels:
            main = item["main"]
            sub = item["sub"]
            for prompt in item["prompts"]:
                self.prompt_list.append(prompt)
                self.label_map.append((main, sub))
        
        print(f"📋 Label mapping ready:")
        print(f"   - Main classes: {sorted(set(item['main'] for item in self.labels))}")
        print(f"   - Has Background: {'Background' in set(item['main'] for item in self.labels)}")
        print(f"   - Has Unknown: {'Unknown' in set(item['main'] for item in self.labels)}")

    def _get_cache_key(self, prompts: List[str]) -> str:
        """Build cache key for prompts (MD5 over prompt contents)"""
        prompt_str = "|".join(sorted(prompts))
        return hashlib.md5(prompt_str.encode()).hexdigest()

    def _get_or_compute_text_embeddings(self) -> torch.Tensor:
        """
        Get or compute text embeddings with caching.
        
        Returns:
            torch.Tensor: normalized text embeddings
        """
        cache_key = self._get_cache_key(self.prompt_list)
        
        # Try load from cache file
        if self.cache_embeddings and os.path.exists(self.embedding_cache_path):
            try:
                with open(self.embedding_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                if cache_data.get('cache_key') == cache_key:
                    print("🔄 Loaded text embeddings from cache")
                    return cache_data['text_embeddings'].to(self.device)
                else:
                    print("⚠️  Cache key mismatch, recomputing text embeddings")
            except Exception as e:
                print(f"⚠️  Failed to load cache: {e}, recomputing text embeddings")
        
        # Recompute text embeddings
        print("🔄 Computing text embeddings...")
        text_tokens = clip.tokenize(self.prompt_list).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_tokens)
            # Normalize for cosine similarity
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        # Save to cache
        if self.cache_embeddings:
            try:
                cache_data = {
                    'cache_key': cache_key,
                    'text_embeddings': text_embeddings.cpu(),
                    'prompt_list': self.prompt_list,
                    'label_map': self.label_map
                }
                with open(self.embedding_cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"💾 Text embeddings cached to {self.embedding_cache_path}")
            except Exception as e:
                print(f"⚠️  Failed to save cache: {e}")
        
        return text_embeddings

    def _get_image_cache_key(self, image_path: str) -> str:
        """Build image cache key (based on path and mtime)"""
        try:
            stat = os.stat(image_path)
            return f"{image_path}_{stat.st_mtime}_{stat.st_size}"
        except:
            return image_path

    def _encode_images_batch(self, image_paths: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Batch-encode images with caching support.
        
        Args:
            image_paths: list of image paths
            batch_size: batch size
            
        Returns:
            torch.Tensor: normalized image embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_embeddings = []
            batch_images = []
            uncached_indices = []
            
            # Check cache
            for idx, path in enumerate(batch_paths):
                cache_key = self._get_image_cache_key(path)
                if cache_key in self.image_embedding_cache:
                    batch_embeddings.append(self.image_embedding_cache[cache_key])
                else:
                    try:
                        image = self.preprocess(Image.open(path)).unsqueeze(0)
                        batch_images.append(image)
                        uncached_indices.append(idx)
                        batch_embeddings.append(None)  # placeholder
                    except Exception as e:
                        print(f"⚠️  Failed to load image {path}: {e}")
                        # Use zero vector as fallback
                        zero_embedding = torch.zeros(512).to(self.device)
                        batch_embeddings.append(zero_embedding)
            
            # Encode uncached images in batch
            if batch_images:
                try:
                    batch_tensor = torch.cat(batch_images).to(self.device)
                    with torch.no_grad():
                        new_embeddings = self.model.encode_image(batch_tensor)
                        new_embeddings = new_embeddings / new_embeddings.norm(dim=-1, keepdim=True)
                    
                    # Update cache and results
                    for i, batch_idx in enumerate(uncached_indices):
                        embedding = new_embeddings[i]
                        batch_embeddings[batch_idx] = embedding
                        
                        # Cache embedding
                        if self.cache_embeddings:
                            cache_key = self._get_image_cache_key(batch_paths[batch_idx])
                            self.image_embedding_cache[cache_key] = embedding
                            
                except Exception as e:
                    print(f"⚠️  Batch image encoding failed: {e}")
                    # Use zero vector for failed embeddings
                    for batch_idx in uncached_indices:
                        if batch_embeddings[batch_idx] is None:
                            batch_embeddings[batch_idx] = torch.zeros(512).to(self.device)
            
            all_embeddings.extend(batch_embeddings)
        
        return torch.stack(all_embeddings)

    def classify_single(self, crop_info: Dict) -> Dict:
        """
        Classify a single crop (kept for backward compatibility)
        
        Args:
            crop_info: dict containing crop_path, etc.
            
        Returns:
            Dict: classification result
        """
        return self.classify_batch([crop_info])[0]

    def classify_batch(self, crop_infos: List[Dict], batch_size: int = 32) -> List[Dict]:
        """
        Batch classification of crops with optimized embedding computation
        
        Args:
            crop_infos: list of crop dicts
            batch_size: batch size
            
        Returns:
            List[Dict]: rich classification results
        """
        if not crop_infos:
            return []
        
        # Extract image paths
        image_paths = [crop["crop_path"] for crop in crop_infos]
        
        # Batch-encode images
        print(f"🔄 Encoding {len(image_paths)} images in batch...")
        image_embeddings = self._encode_images_batch(image_paths, batch_size)
        
        # Compute similarity matrix
        print("🔄 Computing image-text similarities...")
        with torch.no_grad():
            # Cosine similarities
            similarity_matrix = torch.matmul(image_embeddings, self.text_embeddings.T)
            # Convert to probabilities (softmax with temperature scaling)
            probs_matrix = torch.softmax(similarity_matrix * 100, dim=-1)
        
        # Build per-crop results
        results = []
        for i, crop in enumerate(crop_infos):
            probs = probs_matrix[i].cpu().numpy()
            
            # Aggregate probabilities per label (weighted by number of prompts)
            label_scores = {}
            label_counts = {}
            for j, (main, sub) in enumerate(self.label_map):
                key = (main, sub)
                if key not in label_scores:
                    label_scores[key] = []
                    label_counts[key] = 0
                label_scores[key].append(probs[j])
                label_counts[key] += 1
            
            # Weighted average by prompt count
            aggregated_scores = {}
            for key, scores in label_scores.items():
                # Weighted by prompt count (slight bias)
                weight = label_counts[key]
                avg_score = sum(scores) / len(scores)
                # Slightly favor labels with more prompts
                aggregated_scores[key] = float(avg_score * (1 + 0.1 * (weight - 1)))
            
            # Take top-k labels
            sorted_labels = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)
            top_k = sorted_labels[:self.top_k]
            best_label, best_score = top_k[0]
            
            # Unknown fallback strategy
            main_label, sub_label, confidence, unknown_info = self._smart_unknown_classification(
                aggregated_scores, best_label, best_score, top_k
            )
            
            # Rich result payload
            result = {
                "region_id": crop["region_id"],
                "crop_path": crop["crop_path"],
                "main_label": main_label,
                "sub_label": sub_label,
                "confidence": float(confidence),
                "top_k": [
                    {"main_label": k[0], "sub_label": k[1], "score": float(v)} 
                    for k, v in top_k
                ],
                "probabilities": {f"{k[0]}|{k[1]}": float(v) for k, v in aggregated_scores.items()},
                # Extra: context for downstream modules
                "classification_context": self._generate_classification_context(
                    aggregated_scores, top_k, main_label, sub_label
                ),
                "prompt_details": {
                    f"{k[0]}|{k[1]}": {
                        "prompt_count": label_counts[k],
                        "avg_score": float(sum(label_scores[k]) / len(label_scores[k])),
                        "max_score": float(max(label_scores[k]))
                    } for k, _ in top_k[:3]
                }
            }
            
            # Attach unknown classification info for debugging
            if main_label == "unknown":
                result["unknown_info"] = unknown_info
            
            results.append(result)
        
        return results

    def _smart_unknown_classification(self, aggregated_scores: Dict, best_label: Tuple, 
                                    best_score: float, top_k: List) -> Tuple[str, str, float, Dict]:
        """
        Unknown fallback strategy.
        
        Rules:
        1) Absolute threshold → Unknown
        2) Small gap to 2nd best → lower confidence
        3) Low distribution → Unknown
        4) Special handling for Background/Unknown labels
        
        Returns:
            Tuple: (main_label, sub_label, confidence, unknown_info)
        """
        main_label, sub_label = best_label
        confidence = best_score
        
        # Second-best score (if any)
        second_score = top_k[1][1] if len(top_k) > 1 else 0.0
        
        # Distribution stats
        all_scores = list(aggregated_scores.values())
        score_mean = sum(all_scores) / len(all_scores)
        score_std = (sum((s - score_mean) ** 2 for s in all_scores) / len(all_scores)) ** 0.5
        
        unknown_info = {
            "best_attempted_class": best_label,
            "best_attempted_score": best_score,
            "threshold": self.clip_threshold,
            "second_score": second_score,
            "score_mean": score_mean,
            "score_std": score_std,
            "reason": "unknown"
        }
        
        # Rule 1: absolute threshold (relaxed - only if significantly below threshold)
        if best_score < self.clip_threshold * 0.5:  # More strict: only classify as unknown if below half threshold
            unknown_info["reason"] = "below_threshold"
            return "unknown", "unknown", best_score, unknown_info
        
        # Rule 2: relative threshold (gap to second best) - more lenient
        if len(top_k) > 1 and (best_score - second_score) < 0.02:  # Reduced from 0.05 to 0.02
            unknown_info["reason"] = "low_confidence_gap"
            # Lower confidence but keep label
            confidence = best_score * 0.9  # Increased from 0.8 to 0.9
        
        # Rule 3: low distribution (relaxed - requires stricter conditions)
        if score_mean < 0.05 and score_std < 0.02 and best_score < self.clip_threshold:  # Stricter conditions
            unknown_info["reason"] = "low_distribution"
            return "unknown", "unknown", best_score, unknown_info
        
        # Rule 4: background special case (relaxed)
        if main_label == "background" and best_score < 0.2:  # Reduced from 0.3 to 0.2
            unknown_info["reason"] = "background_low_confidence"
            return "unknown", "unknown", best_score, unknown_info
        
        # Rule 5: explicit Unknown as top (unchanged)
        if main_label == "unknown" and best_score > self.clip_threshold:
            unknown_info["reason"] = "unknown_highest"
            return "unknown", "unknown", best_score, unknown_info
        
        return main_label, sub_label, confidence, unknown_info

    def _generate_classification_context(self, aggregated_scores: Dict, top_k: List, 
                                      main_label: str, sub_label: str) -> Dict:
        """
        Build classification context for downstream modules.
        
        Returns:
            Dict: context dict
        """
        # Aggregate scores by main class
        main_class_scores = {}
        for (main, sub), score in aggregated_scores.items():
            if main not in main_class_scores:
                main_class_scores[main] = []
            main_class_scores[main].append(score)
        
        # Max per main class
        main_class_max = {main: max(scores) for main, scores in main_class_scores.items()}
        
        # Top-3 main classes
        top_main_classes = sorted(main_class_max.items(), key=lambda x: x[1], reverse=True)[:3]
        
        context = {
            "primary_class": {
                "main": main_label,
                "sub": sub_label,
                "confidence": aggregated_scores.get((main_label, sub_label), 0.0)
            },
            "top_main_classes": [
                {"main": main, "max_score": score} for main, score in top_main_classes
            ],
            "classification_confidence": "high" if aggregated_scores.get((main_label, sub_label), 0.0) > 0.5 else "medium" if aggregated_scores.get((main_label, sub_label), 0.0) > 0.3 else "low",
            "alternative_classes": [
                {"main": k[0], "sub": k[1], "score": v} 
                for k, v in top_k[1:4]
            ]
        }
        
        return context

    def get_embedding_stats(self) -> Dict:
        """
        Get embedding cache statistics.
        
        Returns:
            Dict: stats
        """
        return {
            "text_embeddings_shape": self.text_embeddings.shape,
            "image_cache_size": len(self.image_embedding_cache),
            "total_prompts      ": len(self.prompt_list),
            "total_labels": len(self.label_map),
            "main_classes": sorted(set(item["main"] for item in self.labels)),
            "cache_enabled": self.cache_embeddings,
            "device": str(self.device),
            "clip_threshold": self.clip_threshold
        }

    def clear_image_cache(self):
        """Clear image embedding cache"""
        self.image_embedding_cache.clear()
        print("🗑️  Image embedding cache cleared")

    def save_image_cache(self, cache_path: Optional[str] = None):
        """
        Save image embedding cache to file.
        
        Args:
            cache_path: output path
        """
        if not self.cache_embeddings:
            return
        
        cache_path = cache_path or "image_embeddings_cache.pkl"
        try:
            # Move tensors to CPU for serialization
            cpu_cache = {k: v.cpu() for k, v in self.image_embedding_cache.items()}
            with open(cache_path, 'wb') as f:
                pickle.dump(cpu_cache, f)
            print(f"💾 Image embedding cache saved to {cache_path}")
        except Exception as e:
            print(f"⚠️  Failed to save image cache: {e}")

    def load_image_cache(self, cache_path: Optional[str] = None):
        """
        Load image embedding cache from file.
        
        Args:
            cache_path: input path
        """
        if not self.cache_embeddings:
            return
        
        cache_path = cache_path or "image_embeddings_cache.pkl"
        try:
            with open(cache_path, 'rb') as f:
                cpu_cache = pickle.load(f)
            # Move tensors back to target device
            self.image_embedding_cache = {k: v.to(self.device) for k, v in cpu_cache.items()}
            print(f"🔄 Image embedding cache loaded from {cache_path}, {len(self.image_embedding_cache)} items")
        except Exception as e:
            print(f"⚠️  Failed to load image cache: {e}")

# Backward-compatible factory
def create_clip_matcher(label_config_path: str, device: str = "cuda", **kwargs) -> CLIPMatcher:
    """
    Factory for CLIPMatcher.
    
    Args:
        label_config_path: label config path
        device: device
        **kwargs: extra params
        
    Returns:
        CLIPMatcher instance
    """
    return CLIPMatcher(label_config_path, device, **kwargs)

