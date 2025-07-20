"""
Performance utilities for Hugging Face deployment optimization.
"""
import streamlit as st
import functools
import time
import psutil
import os
from typing import Any, Callable

def cache_resource_with_ttl(ttl: int = 3600):
    """
    Cache resource with time-to-live for model loading.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return st.cache_resource(ttl=ttl)(func)(*args, **kwargs)
        return wrapper
    return decorator

def monitor_memory():
    """
    Monitor memory usage for debugging.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }

def optimize_model_loading():
    """
    Optimize model loading for faster startup.
    """
    # Pre-compile important modules
    import pickle
    import numpy as np
    import pandas as pd
    
    # Warm up cache
    return True

def performance_timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Log performance if in debug mode
        if os.getenv('DEBUG_PERFORMANCE', 'false').lower() == 'true':
            st.sidebar.text(f"{func.__name__}: {execution_time:.2f}s")
        
        return result
    return wrapper

def cleanup_memory():
    """
    Force garbage collection to free memory.
    """
    import gc
    gc.collect()

class PerformanceMonitor:
    """
    Context manager for monitoring performance.
    """
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = monitor_memory()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_memory = monitor_memory()
        
        duration = end_time - self.start_time
        memory_diff = end_memory['rss'] - self.start_memory['rss']
        
        if os.getenv('DEBUG_PERFORMANCE', 'false').lower() == 'true':
            st.sidebar.write(f"**{self.operation_name}**")
            st.sidebar.write(f"Duration: {duration:.2f}s")
            st.sidebar.write(f"Memory: {memory_diff:+.1f}MB")
