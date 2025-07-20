"""
Analytics and monitoring module for deployment insights.
"""
import time
import json
import streamlit as st
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import logging
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

class AnalyticsManager:
    """
    Manages analytics and monitoring for the application.
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.user_analytics = defaultdict(dict)
        self.model_performance = defaultdict(list)
        self.system_metrics = defaultdict(list)
        
    def track_prediction(self, user_id: str, model_name: str, 
                        prediction: str, confidence: float, 
                        text_length: int, processing_time: float):
        """
        Track prediction analytics.
        
        Args:
            user_id: User identifier
            model_name: Name of the model used
            prediction: Prediction result (AI/Human)
            confidence: Prediction confidence
            text_length: Length of input text
            processing_time: Time taken for prediction
        """
        timestamp = time.time()
        
        prediction_data = {
            "timestamp": timestamp,
            "user_id": user_id,
            "model_name": model_name,
            "prediction": prediction,
            "confidence": confidence,
            "text_length": text_length,
            "processing_time": processing_time
        }
        
        self.metrics["predictions"].append(prediction_data)
        self.model_performance[model_name].append({
            "confidence": confidence,
            "processing_time": processing_time,
            "timestamp": timestamp
        })
        
        # Update user analytics
        if user_id not in self.user_analytics:
            self.user_analytics[user_id] = {
                "first_visit": timestamp,
                "total_predictions": 0,
                "models_used": set(),
                "avg_text_length": 0,
                "total_text_length": 0
            }
        
        user_data = self.user_analytics[user_id]
        user_data["total_predictions"] += 1
        user_data["models_used"].add(model_name)
        user_data["total_text_length"] += text_length
        user_data["avg_text_length"] = user_data["total_text_length"] / user_data["total_predictions"]
        user_data["last_activity"] = timestamp
    
    def track_model_loading(self, model_name: str, loading_time: float, success: bool):
        """
        Track model loading performance.
        
        Args:
            model_name: Name of the model
            loading_time: Time taken to load the model
            success: Whether loading was successful
        """
        self.metrics["model_loading"].append({
            "timestamp": time.time(),
            "model_name": model_name,
            "loading_time": loading_time,
            "success": success
        })
    
    def track_error(self, error_type: str, error_message: str, 
                   user_id: Optional[str] = None, model_name: Optional[str] = None):
        """
        Track application errors.
        
        Args:
            error_type: Type of error
            error_message: Error message
            user_id: User who encountered the error
            model_name: Model involved in the error
        """
        self.metrics["errors"].append({
            "timestamp": time.time(),
            "error_type": error_type,
            "error_message": error_message,
            "user_id": user_id,
            "model_name": model_name
        })
    
    def track_file_upload(self, user_id: str, file_type: str, 
                         file_size: int, processing_time: float):
        """
        Track file upload analytics.
        
        Args:
            user_id: User identifier
            file_type: Type of uploaded file
            file_size: Size of the file in bytes
            processing_time: Time taken to process the file
        """
        self.metrics["file_uploads"].append({
            "timestamp": time.time(),
            "user_id": user_id,
            "file_type": file_type,
            "file_size": file_size,
            "processing_time": processing_time
        })
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for analytics dashboard.
        
        Returns:
            Dictionary containing analytics data
        """
        current_time = time.time()
        hour_ago = current_time - 3600
        day_ago = current_time - 86400
        
        predictions = self.metrics["predictions"]
        
        # Filter recent predictions
        recent_predictions = [p for p in predictions if p["timestamp"] >= hour_ago]
        daily_predictions = [p for p in predictions if p["timestamp"] >= day_ago]
        
        # Calculate metrics
        total_predictions = len(predictions)
        hourly_predictions = len(recent_predictions)
        daily_predictions_count = len(daily_predictions)
        
        # Model usage statistics
        model_usage = Counter(p["model_name"] for p in predictions)
        
        # Prediction distribution
        prediction_distribution = Counter(p["prediction"] for p in predictions)
        
        # Average confidence by model
        avg_confidence = {}
        for model in model_usage.keys():
            model_predictions = [p for p in predictions if p["model_name"] == model]
            if model_predictions:
                avg_confidence[model] = sum(p["confidence"] for p in model_predictions) / len(model_predictions)
        
        # Processing time statistics
        processing_times = [p["processing_time"] for p in predictions if "processing_time" in p]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Error statistics
        errors = self.metrics["errors"]
        error_count = len(errors)
        recent_errors = [e for e in errors if e["timestamp"] >= hour_ago]
        error_rate = len(recent_errors) / max(hourly_predictions, 1)
        
        return {
            "total_predictions": total_predictions,
            "hourly_predictions": hourly_predictions,
            "daily_predictions": daily_predictions_count,
            "unique_users": len(self.user_analytics),
            "model_usage": dict(model_usage),
            "prediction_distribution": dict(prediction_distribution),
            "avg_confidence": avg_confidence,
            "avg_processing_time": avg_processing_time,
            "error_count": error_count,
            "error_rate": error_rate,
            "file_uploads": len(self.metrics["file_uploads"])
        }
    
    def get_model_performance_stats(self, model_name: str) -> Dict[str, Any]:
        """
        Get performance statistics for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Performance statistics
        """
        if model_name not in self.model_performance:
            return {}
        
        performance_data = self.model_performance[model_name]
        
        confidences = [p["confidence"] for p in performance_data]
        processing_times = [p["processing_time"] for p in performance_data]
        
        return {
            "total_predictions": len(performance_data),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "min_confidence": min(confidences) if confidences else 0,
            "max_confidence": max(confidences) if confidences else 0,
            "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
            "min_processing_time": min(processing_times) if processing_times else 0,
            "max_processing_time": max(processing_times) if processing_times else 0
        }
    
    def export_analytics_data(self) -> Dict[str, Any]:
        """
        Export all analytics data for analysis.
        
        Returns:
            Complete analytics dataset
        """
        return {
            "predictions": self.metrics["predictions"],
            "model_loading": self.metrics["model_loading"],
            "errors": self.metrics["errors"],
            "file_uploads": self.metrics["file_uploads"],
            "user_analytics": {
                user_id: {**data, "models_used": list(data["models_used"])}
                for user_id, data in self.user_analytics.items()
            },
            "dashboard_metrics": self.get_dashboard_metrics()
        }
    
    def display_analytics_dashboard(self):
        """Display analytics dashboard in Streamlit."""
        st.subheader("📊 Analytics Dashboard")
        
        metrics = self.get_dashboard_metrics()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", metrics["total_predictions"])
        
        with col2:
            st.metric("Hourly Predictions", metrics["hourly_predictions"])
        
        with col3:
            st.metric("Unique Users", metrics["unique_users"])
        
        with col4:
            st.metric("Avg Processing Time", f"{metrics['avg_processing_time']:.2f}s")
        
        # Model usage chart
        if metrics["model_usage"]:
            st.subheader("Model Usage Distribution")
            model_df = pd.DataFrame(
                list(metrics["model_usage"].items()),
                columns=["Model", "Usage Count"]
            )
            st.bar_chart(model_df.set_index("Model"))
        
        # Prediction distribution
        if metrics["prediction_distribution"]:
            st.subheader("Prediction Distribution")
            pred_df = pd.DataFrame(
                list(metrics["prediction_distribution"].items()),
                columns=["Prediction", "Count"]
            )
            st.bar_chart(pred_df.set_index("Prediction"))
        
        # Error monitoring
        if metrics["error_count"] > 0:
            st.subheader("Error Monitoring")
            st.warning(f"Total Errors: {metrics['error_count']}")
            st.metric("Error Rate", f"{metrics['error_rate']:.1%}")

# Global analytics manager instance
analytics_manager = AnalyticsManager()
