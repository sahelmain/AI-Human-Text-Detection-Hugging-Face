"""
Robust error handling for production Hugging Face deployment.
"""
import streamlit as st
import logging
import traceback
import sys
from typing import Any, Callable, Optional
import functools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AITextDetectionError(Exception):
    """Custom exception for AI Text Detection app."""
    pass

class ModelLoadError(AITextDetectionError):
    """Exception raised when model loading fails."""
    pass

class PredictionError(AITextDetectionError):
    """Exception raised during prediction."""
    pass

def safe_execute(func: Callable, fallback_value: Any = None, error_message: str = None) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        fallback_value: Value to return if function fails
        error_message: Custom error message to display
    
    Returns:
        Function result or fallback value
    """
    try:
        return func()
    except Exception as e:
        error_msg = error_message or f"An error occurred: {str(e)}"
        logger.error(f"Safe execution failed: {error_msg}", exc_info=True)
        st.error(error_msg)
        return fallback_value

def handle_model_errors(func: Callable) -> Callable:
    """
    Decorator for handling model-related errors.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            error_msg = "Model file not found. Please check if all models are properly loaded."
            logger.error(f"Model file error: {str(e)}")
            st.error(error_msg)
            st.info("💡 Try refreshing the page or contact support if the issue persists.")
            return None
        except Exception as e:
            error_msg = f"Model error: {str(e)}"
            logger.error(f"Model execution error: {error_msg}", exc_info=True)
            st.error("An error occurred while processing your request.")
            st.info("💡 Please try again or contact support if the issue persists.")
            return None
    return wrapper

def handle_prediction_errors(func: Callable) -> Callable:
    """
    Decorator for handling prediction errors.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            error_msg = "Invalid input data. Please check your text input."
            logger.error(f"Prediction input error: {str(e)}")
            st.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(f"Prediction error: {error_msg}", exc_info=True)
            st.error("Unable to analyze the text. Please try again.")
            return None
    return wrapper

def display_error_info():
    """
    Display error information for debugging.
    """
    if st.checkbox("Show Error Details (Debug)", value=False):
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_type is not None:
            st.text("Error Details:")
            st.code(traceback.format_exception(exc_type, exc_value, exc_traceback))

def validate_input(text: str, min_length: int = 10, max_length: int = 10000) -> bool:
    """
    Validate input text.
    
    Args:
        text: Input text to validate
        min_length: Minimum text length
        max_length: Maximum text length
    
    Returns:
        True if valid, False otherwise
    """
    if not text or not text.strip():
        st.error("Please enter some text to analyze.")
        return False
    
    text_length = len(text.strip())
    
    if text_length < min_length:
        st.error(f"Text is too short. Please enter at least {min_length} characters.")
        return False
    
    if text_length > max_length:
        st.error(f"Text is too long. Please limit to {max_length} characters.")
        return False
    
    return True

def log_user_action(action: str, details: Optional[str] = None):
    """
    Log user actions for monitoring.
    """
    log_message = f"User action: {action}"
    if details:
        log_message += f" - {details}"
    logger.info(log_message)
