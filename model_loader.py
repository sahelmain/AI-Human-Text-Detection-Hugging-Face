"""
Optimized model loader for faster Hugging Face deployment startup.
"""
import pickle
import os
import streamlit as st
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import hashlib
import time

from config_manager import config
from error_handler import handle_model_errors, ModelLoadError
from performance_utils import cache_resource_with_ttl, PerformanceMonitor

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Optimized model loader with caching and error handling.
    """
    
    def __init__(self):
        self.models_dir = Path(config.get("models.cache_dir", "./models"))
        self.loaded_models = {}
        self.model_metadata = {}
        self._ensure_models_dir()
    
    def _ensure_models_dir(self):
        """Ensure models directory exists."""
        self.models_dir.mkdir(exist_ok=True)
    
    @st.cache_resource(ttl=3600)
    def _load_model_file(self, model_path: str) -> Any:
        """
        Load a single model file with caching.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model object
        """
        try:
            with PerformanceMonitor(f"Loading {os.path.basename(model_path)}"):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Successfully loaded model: {model_path}")
                return model
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            raise ModelLoadError(f"Could not load model {model_path}: {str(e)}")
    
    def _get_model_path(self, model_name: str) -> Path:
        """Get the full path for a model file."""
        model_mapping = {
            "CNN": "CNN.pkl",
            "LSTM": "LSTM.pkl", 
            "RNN": "RNN.pkl",
            "SVM": "svm_model.pkl",
            "decision_tree": "decision_tree_model.pkl",
            "adaboost": "adaboost_model.pkl",
            "logistic_regression": "logistic_regression_model.pkl",
            "multinomial_nb": "multinomial_nb_model.pkl",
            "tfidf_vectorizer": "tfidf_vectorizer.pkl",
            "vocab_to_idx": "vocab_to_idx.pkl",
            "complete_pipeline": "complete_pipeline.pkl"
        }
        
        filename = model_mapping.get(model_name, f"{model_name}.pkl")
        return self.models_dir / filename
    
    def _validate_model_file(self, model_path: Path) -> bool:
        """Validate that model file exists and is readable."""
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return False
        
        if model_path.stat().st_size == 0:
            logger.warning(f"Model file is empty: {model_path}")
            return False
        
        return True
    
    @handle_model_errors
    def load_model(self, model_name: str, required: bool = True) -> Optional[Any]:
        """
        Load a specific model with error handling.
        
        Args:
            model_name: Name of the model to load
            required: Whether the model is required for app functionality
            
        Returns:
            Loaded model or None if loading fails
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        model_path = self._get_model_path(model_name)
        
        if not self._validate_model_file(model_path):
            if required:
                raise ModelLoadError(f"Required model not found: {model_name}")
            return None
        
        try:
            model = self._load_model_file(str(model_path))
            self.loaded_models[model_name] = model
            
            # Store metadata
            self.model_metadata[model_name] = {
                "path": str(model_path),
                "size_mb": model_path.stat().st_size / (1024 * 1024),
                "loaded_at": time.time()
            }
            
            return model
        except Exception as e:
            if required:
                raise ModelLoadError(f"Failed to load required model {model_name}: {str(e)}")
            logger.warning(f"Optional model {model_name} could not be loaded: {str(e)}")
            return None
    
    def load_all_models(self) -> Dict[str, Any]:
        """
        Load all available models.
        
        Returns:
            Dictionary of loaded models
        """
        models_to_load = config.get("models.models_to_load", [])
        
        with st.spinner("Loading AI models... This may take a moment."):
            progress_bar = st.progress(0)
            
            for i, model_name in enumerate(models_to_load):
                try:
                    self.load_model(model_name, required=False)
                    progress_bar.progress((i + 1) / len(models_to_load))
                except Exception as e:
                    logger.warning(f"Could not load {model_name}: {str(e)}")
            
            progress_bar.empty()
        
        return self.loaded_models
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded models."""
        return self.model_metadata.copy()
    
    def preload_essential_models(self) -> bool:
        """
        Preload essential models for faster response.
        
        Returns:
            True if all essential models loaded successfully
        """
        essential_models = ["tfidf_vectorizer", "SVM", "CNN"]
        success = True
        
        for model_name in essential_models:
            try:
                self.load_model(model_name, required=True)
            except Exception as e:
                logger.error(f"Failed to preload essential model {model_name}: {str(e)}")
                success = False
        
        return success
    
    def cleanup_models(self):
        """Clear loaded models from memory."""
        self.loaded_models.clear()
        self.model_metadata.clear()
        
        # Force garbage collection
        import gc
        gc.collect()

# Global model loader instance
model_loader = ModelLoader()
