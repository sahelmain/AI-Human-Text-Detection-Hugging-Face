"""
Optimization utilities for production performance on Hugging Face.
"""
import os
import gc
import time
import psutil
import streamlit as st
from typing import Any, Dict, List, Optional, Callable
from functools import wraps, lru_cache
import pickle
import hashlib
import threading
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """
    Memory optimization utilities for efficient resource usage.
    """
    
    def __init__(self):
        self.memory_threshold = 0.85  # 85% memory usage threshold
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024
            }
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return {"error": str(e)}
    
    def should_cleanup(self) -> bool:
        """Determine if memory cleanup is needed."""
        try:
            memory_usage = self.get_memory_usage()
            
            # Cleanup if memory usage is high or interval has passed
            high_memory = memory_usage.get("percent", 0) > self.memory_threshold * 100
            time_passed = time.time() - self.last_cleanup > self.cleanup_interval
            
            return high_memory or time_passed
        except Exception:
            return False
    
    def cleanup_memory(self):
        """Perform memory cleanup."""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Clear Streamlit cache if needed
            if hasattr(st, 'cache_data'):
                st.cache_data.clear()
            if hasattr(st, 'cache_resource'):
                st.cache_resource.clear()
            
            self.last_cleanup = time.time()
            
            logger.info(f"Memory cleanup performed, collected {collected} objects")
            
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    def monitor_memory(self, func: Callable) -> Callable:
        """Decorator to monitor memory usage of functions."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_memory = self.get_memory_usage()
            
            result = func(*args, **kwargs)
            
            end_memory = self.get_memory_usage()
            
            if start_memory and end_memory:
                memory_diff = end_memory.get("rss_mb", 0) - start_memory.get("rss_mb", 0)
                if memory_diff > 10:  # Log if function used more than 10MB
                    logger.info(f"{func.__name__} used {memory_diff:.1f}MB memory")
            
            # Auto-cleanup if needed
            if self.should_cleanup():
                self.cleanup_memory()
            
            return result
        
        return wrapper

class CacheOptimizer:
    """
    Intelligent caching system for models and predictions.
    """
    
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self.prediction_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_text_hash(self, text: str, model_name: str) -> str:
        """Generate hash for text and model combination."""
        combined = f"{text}:{model_name}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_cached_prediction(self, text: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction if available."""
        cache_key = self.get_text_hash(text, model_name)
        
        if cache_key in self.prediction_cache:
            self.cache_hits += 1
            cached_result = self.prediction_cache[cache_key]
            
            # Check if cache entry is still valid (not too old)
            if time.time() - cached_result["timestamp"] < 3600:  # 1 hour
                return cached_result["prediction"]
        
        self.cache_misses += 1
        return None
    
    def cache_prediction(self, text: str, model_name: str, prediction: Dict[str, Any]):
        """Cache a prediction result."""
        cache_key = self.get_text_hash(text, model_name)
        
        # Clean old cache entries if cache is full
        if len(self.prediction_cache) >= self.max_cache_size:
            self._cleanup_old_cache_entries()
        
        self.prediction_cache[cache_key] = {
            "prediction": prediction,
            "timestamp": time.time()
        }
    
    def _cleanup_old_cache_entries(self):
        """Remove old cache entries to make space."""
        current_time = time.time()
        
        # Remove entries older than 1 hour
        old_keys = [
            key for key, value in self.prediction_cache.items()
            if current_time - value["timestamp"] > 3600
        ]
        
        for key in old_keys:
            del self.prediction_cache[key]
        
        # If still too many entries, remove oldest ones
        if len(self.prediction_cache) >= self.max_cache_size:
            sorted_items = sorted(
                self.prediction_cache.items(),
                key=lambda x: x[1]["timestamp"]
            )
            
            # Keep only the newest 80% of entries
            keep_count = int(self.max_cache_size * 0.8)
            self.prediction_cache = dict(sorted_items[-keep_count:])
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.prediction_cache),
            "max_cache_size": self.max_cache_size
        }

class ModelOptimizer:
    """
    Optimization utilities for model loading and inference.
    """
    
    def __init__(self):
        self.model_load_times = {}
        self.preloaded_models = set()
    
    def preload_essential_models(self, model_loader, essential_models: List[str]):
        """Preload essential models in background."""
        def preload_worker():
            for model_name in essential_models:
                try:
                    start_time = time.time()
                    model_loader.load_model(model_name, required=False)
                    load_time = time.time() - start_time
                    
                    self.model_load_times[model_name] = load_time
                    self.preloaded_models.add(model_name)
                    
                    logger.info(f"Preloaded {model_name} in {load_time:.2f}s")
                    
                except Exception as e:
                    logger.warning(f"Failed to preload {model_name}: {e}")
        
        # Start preloading in background thread
        thread = threading.Thread(target=preload_worker, daemon=True)
        thread.start()
    
    def optimize_model_inference(self, func: Callable) -> Callable:
        """Decorator to optimize model inference."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Add any model-specific optimizations here
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            inference_time = time.time() - start_time
            
            # Log slow inferences
            if inference_time > 5.0:  # Log if inference takes more than 5 seconds
                logger.warning(f"Slow inference: {func.__name__} took {inference_time:.2f}s")
            
            return result
        
        return wrapper

class PerformanceProfiler:
    """
    Performance profiling utilities for optimization insights.
    """
    
    def __init__(self):
        self.function_times = {}
        self.function_calls = {}
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function performance."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            func_name = func.__name__
            
            # Track statistics
            if func_name not in self.function_times:
                self.function_times[func_name] = []
                self.function_calls[func_name] = 0
            
            self.function_times[func_name].append(execution_time)
            self.function_calls[func_name] += 1
            
            return result
        
        return wrapper
    
    def get_performance_report(self) -> Dict[str, Dict[str, float]]:
        """Get performance profiling report."""
        report = {}
        
        for func_name, times in self.function_times.items():
            if times:
                report[func_name] = {
                    "calls": self.function_calls[func_name],
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times)
                }
        
        return report
    
    def display_performance_dashboard(self):
        """Display performance dashboard in Streamlit."""
        st.subheader("⚡ Performance Dashboard")
        
        report = self.get_performance_report()
        
        if report:
            # Convert to DataFrame for display
            import pandas as pd
            
            df = pd.DataFrame(report).T
            df = df.round(3)
            
            st.dataframe(df)
            
            # Show top slowest functions
            slowest = df.nlargest(5, "avg_time")
            if not slowest.empty:
                st.subheader("🐌 Slowest Functions")
                st.dataframe(slowest[["calls", "avg_time", "max_time"]])
        else:
            st.info("No performance data available yet.")

# Global optimization instances
memory_optimizer = MemoryOptimizer()
cache_optimizer = CacheOptimizer()
model_optimizer = ModelOptimizer()
performance_profiler = PerformanceProfiler()

# Decorator shortcuts
monitor_memory = memory_optimizer.monitor_memory
optimize_inference = model_optimizer.optimize_model_inference
profile_performance = performance_profiler.profile_function
