"""
Configuration manager for flexible Hugging Face deployment settings.
"""
import os
import json
from typing import Dict, Any, Optional
import streamlit as st

class ConfigManager:
    """
    Manages application configuration for different deployment environments.
    """
    
    def __init__(self):
        self.config = self._load_default_config()
        self._load_environment_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration settings."""
        return {
            "app": {
                "title": "AI vs Human Text Detection",
                "emoji": "🤖",
                "layout": "wide",
                "max_upload_size": 200,  # MB
                "supported_formats": [".txt", ".pdf", ".docx"]
            },
            "models": {
                "cache_dir": "./models",
                "load_timeout": 30,  # seconds
                "models_to_load": ["CNN", "LSTM", "RNN", "SVM", "decision_tree", "adaboost"]
            },
            "performance": {
                "cache_ttl": 3600,  # 1 hour
                "max_workers": 4,
                "memory_limit_mb": 4096,
                "enable_monitoring": False
            },
            "ui": {
                "theme": "light",
                "primary_color": "#FF6B6B",
                "show_debug": False,
                "enable_explanations": True,
                "default_model": "CNN"
            },
            "nltk": {
                "data_path": "./nltk_data",
                "required_packages": ["punkt", "stopwords", "vader_lexicon", "punkt_tab"]
            },
            "api": {
                "openai_enabled": False,
                "rate_limit": 100,  # requests per hour
                "timeout": 30  # seconds
            }
        }
    
    def _load_environment_config(self):
        """Load configuration from environment variables."""
        # App settings
        if os.getenv("APP_TITLE"):
            self.config["app"]["title"] = os.getenv("APP_TITLE")
        
        if os.getenv("MAX_UPLOAD_SIZE"):
            self.config["app"]["max_upload_size"] = int(os.getenv("MAX_UPLOAD_SIZE"))
        
        # Performance settings
        if os.getenv("CACHE_TTL"):
            self.config["performance"]["cache_ttl"] = int(os.getenv("CACHE_TTL"))
        
        if os.getenv("MEMORY_LIMIT_MB"):
            self.config["performance"]["memory_limit_mb"] = int(os.getenv("MEMORY_LIMIT_MB"))
        
        # Debug mode
        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            self.config["ui"]["show_debug"] = True
            self.config["performance"]["enable_monitoring"] = True
        
        # API settings
        if os.getenv("OPENAI_API_KEY"):
            self.config["api"]["openai_enabled"] = True
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., "app.title")
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split(".")
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get("ui.show_debug", False)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get("models", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self.get("performance", {})
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration."""
        return self.get("ui", {})
    
    def save_to_file(self, filepath: str):
        """Save current configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load configuration from file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                file_config = json.load(f)
                self._merge_config(file_config)
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing."""
        def merge_dict(base: dict, update: dict):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(self.config, new_config)

# Global configuration instance
config = ConfigManager()
