"""
API endpoints for programmatic access to AI text detection.
"""
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import streamlit as st
from pathlib import Path

@dataclass
class PredictionRequest:
    """Data class for prediction requests."""
    text: str
    model_name: Optional[str] = "CNN"
    include_confidence: bool = True
    include_explanation: bool = False

@dataclass
class PredictionResponse:
    """Data class for prediction responses."""
    prediction: str
    confidence: float
    model_used: str
    processing_time: float
    text_length: int
    timestamp: float
    explanation: Optional[str] = None

class APIEndpoints:
    """
    API-like endpoints for the Streamlit app.
    """
    
    def __init__(self, model_loader, security_manager, analytics_manager):
        self.model_loader = model_loader
        self.security_manager = security_manager
        self.analytics_manager = analytics_manager
    
    def predict_text(self, request: PredictionRequest, user_id: str) -> Dict[str, Any]:
        """
        Main prediction endpoint.
        
        Args:
            request: Prediction request data
            user_id: User identifier
            
        Returns:
            Prediction response
        """
        start_time = time.time()
        
        try:
            # Security validation
            if not self.security_manager.check_rate_limit(user_id, "prediction"):
                return {
                    "error": "Rate limit exceeded",
                    "code": 429,
                    "message": "Too many requests. Please try again later."
                }
            
            # Input validation
            is_valid, sanitized_text = self.security_manager.validate_and_sanitize_input(request.text)
            if not is_valid:
                return {
                    "error": "Invalid input",
                    "code": 400,
                    "message": sanitized_text
                }
            
            # Load model
            model = self.model_loader.load_model(request.model_name)
            if model is None:
                return {
                    "error": "Model not available",
                    "code": 503,
                    "message": f"Model {request.model_name} is not available"
                }
            
            # Make prediction (simplified - would use actual model prediction)
            prediction_result = self._make_prediction(sanitized_text, model, request.model_name)
            
            processing_time = time.time() - start_time
            
            # Track analytics
            self.analytics_manager.track_prediction(
                user_id=user_id,
                model_name=request.model_name,
                prediction=prediction_result["prediction"],
                confidence=prediction_result["confidence"],
                text_length=len(sanitized_text),
                processing_time=processing_time
            )
            
            # Build response
            response = {
                "prediction": prediction_result["prediction"],
                "confidence": prediction_result["confidence"],
                "model_used": request.model_name,
                "processing_time": processing_time,
                "text_length": len(sanitized_text),
                "timestamp": time.time()
            }
            
            if request.include_explanation:
                response["explanation"] = self._generate_explanation(
                    sanitized_text, 
                    prediction_result
                )
            
            return {"success": True, "data": response}
            
        except Exception as e:
            self.analytics_manager.track_error(
                error_type="prediction_error",
                error_message=str(e),
                user_id=user_id,
                model_name=request.model_name
            )
            
            return {
                "error": "Internal server error",
                "code": 500,
                "message": "An error occurred while processing your request"
            }
    
    def batch_predict(self, texts: List[str], model_name: str, user_id: str) -> Dict[str, Any]:
        """
        Batch prediction endpoint.
        
        Args:
            texts: List of texts to analyze
            model_name: Name of the model to use
            user_id: User identifier
            
        Returns:
            Batch prediction response
        """
        # Check rate limits for batch processing
        if not self.security_manager.check_rate_limit(user_id, "batch_prediction"):
            return {
                "error": "Rate limit exceeded",
                "code": 429,
                "message": "Batch processing rate limit exceeded"
            }
        
        if len(texts) > 100:  # Limit batch size
            return {
                "error": "Batch too large",
                "code": 400,
                "message": "Maximum 100 texts per batch"
            }
        
        results = []
        errors = []
        
        for i, text in enumerate(texts):
            try:
                request = PredictionRequest(text=text, model_name=model_name)
                result = self.predict_text(request, user_id)
                
                if result.get("success"):
                    results.append({"index": i, "result": result["data"]})
                else:
                    errors.append({"index": i, "error": result})
                    
            except Exception as e:
                errors.append({"index": i, "error": str(e)})
        
        return {
            "success": True,
            "data": {
                "results": results,
                "errors": errors,
                "total_processed": len(results),
                "total_errors": len(errors)
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about available models.
        
        Returns:
            Model information
        """
        model_info = self.model_loader.get_model_info()
        
        # Add performance statistics
        enhanced_info = {}
        for model_name, info in model_info.items():
            performance_stats = self.analytics_manager.get_model_performance_stats(model_name)
            enhanced_info[model_name] = {**info, "performance": performance_stats}
        
        return {
            "success": True,
            "data": {
                "available_models": list(enhanced_info.keys()),
                "model_details": enhanced_info,
                "default_model": "CNN"
            }
        }
    
    def get_analytics(self, user_id: str) -> Dict[str, Any]:
        """
        Get analytics data (admin only).
        
        Args:
            user_id: User identifier
            
        Returns:
            Analytics data
        """
        # Simple admin check (in real implementation, would be more sophisticated)
        if not self._is_admin_user(user_id):
            return {
                "error": "Unauthorized",
                "code": 403,
                "message": "Admin access required"
            }
        
        analytics_data = self.analytics_manager.get_dashboard_metrics()
        
        return {
            "success": True,
            "data": analytics_data
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint.
        
        Returns:
            Health status
        """
        try:
            # Check if essential models are loaded
            essential_models = ["SVM", "tfidf_vectorizer"]
            model_status = {}
            
            for model_name in essential_models:
                try:
                    model = self.model_loader.load_model(model_name, required=False)
                    model_status[model_name] = "healthy" if model is not None else "unavailable"
                except Exception:
                    model_status[model_name] = "error"
            
            overall_health = "healthy" if all(
                status == "healthy" for status in model_status.values()
            ) else "degraded"
            
            return {
                "success": True,
                "data": {
                    "status": overall_health,
                    "timestamp": time.time(),
                    "models": model_status,
                    "version": "1.0.0"
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "data": {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": time.time()
                }
            }
    
    def _make_prediction(self, text: str, model: Any, model_name: str) -> Dict[str, Any]:
        """
        Make prediction using the loaded model.
        
        Args:
            text: Text to analyze
            model: Loaded model
            model_name: Name of the model
            
        Returns:
            Prediction result
        """
        # This is a simplified version - in real implementation,
        # would use the actual model prediction logic from utils.py
        
        # For demonstration, return dummy prediction
        import random
        prediction = "AI" if random.random() > 0.5 else "Human"
        confidence = random.uniform(0.7, 0.99)
        
        return {
            "prediction": prediction,
            "confidence": confidence
        }
    
    def _generate_explanation(self, text: str, prediction_result: Dict[str, Any]) -> str:
        """
        Generate explanation for the prediction.
        
        Args:
            text: Input text
            prediction_result: Prediction result
            
        Returns:
            Explanation string
        """
        prediction = prediction_result["prediction"]
        confidence = prediction_result["confidence"]
        
        if prediction == "AI":
            return f"This text appears to be AI-generated with {confidence:.1%} confidence. " \
                   f"Key indicators include consistent style and potentially repetitive patterns."
        else:
            return f"This text appears to be human-written with {confidence:.1%} confidence. " \
                   f"Key indicators include natural variations and personal voice."
    
    def _is_admin_user(self, user_id: str) -> bool:
        """Check if user has admin privileges."""
        # Simple implementation - in production, would check against user database
        return user_id == "admin" or user_id.startswith("admin_")

# Helper functions for Streamlit integration
def create_api_interface():
    """Create API interface in Streamlit sidebar."""
    with st.sidebar.expander("🔌 API Access", expanded=False):
        st.markdown("### Programmatic Access")
        st.markdown("This app provides API-like endpoints for programmatic access:")
        
        endpoints = [
            "**predict_text()** - Single text prediction",
            "**batch_predict()** - Batch text processing", 
            "**get_model_info()** - Model information",
            "**health_check()** - System health status"
        ]
        
        for endpoint in endpoints:
            st.markdown(f"• {endpoint}")
        
        st.markdown("---")
        st.markdown("💡 **Usage Example:**")
        st.code("""
# Example usage in Python
request = PredictionRequest(
    text="Your text here",
    model_name="CNN",
    include_explanation=True
)
result = api.predict_text(request, user_id)
        """, language="python")
