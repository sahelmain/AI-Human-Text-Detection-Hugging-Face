"""
Testing utilities for deployment validation.
"""
import unittest
import os
import sys
import tempfile
import pickle
from pathlib import Path
from typing import Dict, Any, List
import streamlit as st

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class DeploymentTests(unittest.TestCase):
    """
    Test suite for validating deployment readiness.
    """
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(__file__).parent
        self.models_dir = self.test_dir / "models"
    
    def test_required_files_exist(self):
        """Test that all required files exist."""
        required_files = [
            "app.py",
            "requirements.txt",
            "README.md",
            "utils.py",
            "setup_nltk.py"
        ]
        
        for filename in required_files:
            file_path = self.test_dir / filename
            self.assertTrue(
                file_path.exists(),
                f"Required file {filename} is missing"
            )
    
    def test_models_directory_exists(self):
        """Test that models directory exists and contains models."""
        self.assertTrue(
            self.models_dir.exists(),
            "Models directory does not exist"
        )
        
        # Check for key model files
        key_models = [
            "CNN.pkl",
            "LSTM.pkl",
            "RNN.pkl",
            "svm_model.pkl",
            "tfidf_vectorizer.pkl"
        ]
        
        for model_file in key_models:
            model_path = self.models_dir / model_file
            self.assertTrue(
                model_path.exists(),
                f"Key model file {model_file} is missing"
            )
    
    def test_model_files_valid(self):
        """Test that model files can be loaded."""
        model_files = [
            "svm_model.pkl",
            "tfidf_vectorizer.pkl",
            "decision_tree_model.pkl"
        ]
        
        for model_file in model_files:
            model_path = self.models_dir / model_file
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    self.assertIsNotNone(model, f"Model {model_file} loaded as None")
                except Exception as e:
                    self.fail(f"Failed to load model {model_file}: {str(e)}")
    
    def test_requirements_file_valid(self):
        """Test that requirements.txt is valid."""
        requirements_file = self.test_dir / "requirements.txt"
        
        with open(requirements_file, 'r') as f:
            requirements = f.read()
        
        # Check for essential packages
        essential_packages = [
            "streamlit",
            "pandas",
            "numpy",
            "scikit-learn",
            "plotly",
            "nltk"
        ]
        
        for package in essential_packages:
            self.assertIn(
                package,
                requirements,
                f"Essential package {package} not found in requirements.txt"
            )
    
    def test_streamlit_config_valid(self):
        """Test that Streamlit configuration is valid."""
        config_file = self.test_dir / ".streamlit" / "config.toml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_content = f.read()
            
            # Check for required sections
            self.assertIn("[server]", config_content)
            self.assertIn("headless = true", config_content)
    
    def test_readme_has_huggingface_metadata(self):
        """Test that README.md has proper Hugging Face metadata."""
        readme_file = self.test_dir / "README.md"
        
        with open(readme_file, 'r') as f:
            readme_content = f.read()
        
        # Check for HF metadata
        required_metadata = [
            "title:",
            "emoji:",
            "sdk: streamlit",
            "app_file: app.py"
        ]
        
        for metadata in required_metadata:
            self.assertIn(
                metadata,
                readme_content,
                f"Required metadata {metadata} not found in README.md"
            )
    
    def test_imports_work(self):
        """Test that critical imports work."""
        try:
            # Test basic imports
            import pandas as pd
            import numpy as np
            import streamlit as st
            import plotly.graph_objects as go
            import sklearn
            
            # Test custom module imports
            from utils import AITextDetector
            
        except ImportError as e:
            self.fail(f"Critical import failed: {str(e)}")
    
    def test_nltk_data_setup(self):
        """Test NLTK data setup script."""
        setup_script = self.test_dir / "setup_nltk.py"
        
        # Test that the script exists and is executable
        self.assertTrue(setup_script.exists())
        
        # Test that it contains required downloads
        with open(setup_script, 'r') as f:
            script_content = f.read()
        
        nltk_datasets = ["punkt", "stopwords", "vader_lexicon"]
        for dataset in nltk_datasets:
            self.assertIn(dataset, script_content)
    
    def test_security_module_exists(self):
        """Test that security enhancements are in place."""
        security_file = self.test_dir / "security.py"
        
        if security_file.exists():
            with open(security_file, 'r') as f:
                security_content = f.read()
            
            # Check for security features
            security_features = [
                "RateLimiter",
                "InputValidator",
                "sanitize_text"
            ]
            
            for feature in security_features:
                self.assertIn(feature, security_content)

def run_deployment_tests():
    """Run all deployment tests and return results."""
    test_suite = unittest.TestLoader().loadTestsFromTestCase(DeploymentTests)
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success": result.wasSuccessful()
    }

def validate_deployment_readiness() -> Dict[str, Any]:
    """
    Comprehensive deployment readiness validation.
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "ready_for_deployment": True,
        "issues": [],
        "warnings": [],
        "test_results": None
    }
    
    try:
        # Run unit tests
        test_results = run_deployment_tests()
        validation_results["test_results"] = test_results
        
        if not test_results["success"]:
            validation_results["ready_for_deployment"] = False
            validation_results["issues"].append("Unit tests failed")
        
        # Check file sizes (Hugging Face has limits)
        models_dir = Path("models")
        if models_dir.exists():
            total_size = sum(f.stat().st_size for f in models_dir.rglob('*') if f.is_file())
            total_size_mb = total_size / (1024 * 1024)
            
            if total_size_mb > 500:  # Warning at 500MB
                validation_results["warnings"].append(
                    f"Models directory is large ({total_size_mb:.1f}MB). "
                    "Consider model optimization for faster deployment."
                )
        
        # Check for .env files (shouldn't be in repo)
        if Path(".env").exists():
            validation_results["warnings"].append(
                ".env file found. Ensure sensitive data is not committed."
            )
        
    except Exception as e:
        validation_results["ready_for_deployment"] = False
        validation_results["issues"].append(f"Validation error: {str(e)}")
    
    return validation_results

if __name__ == "__main__":
    print("Running deployment validation...")
    results = validate_deployment_readiness()
    
    print(f"\nDeployment Ready: {results['ready_for_deployment']}")
    
    if results["issues"]:
        print("\n❌ Issues found:")
        for issue in results["issues"]:
            print(f"  - {issue}")
    
    if results["warnings"]:
        print("\n⚠️ Warnings:")
        for warning in results["warnings"]:
            print(f"  - {warning}")
    
    if results["ready_for_deployment"]:
        print("\n✅ Ready for Hugging Face deployment!")
    else:
        print("\n❌ Please fix issues before deploying.")
