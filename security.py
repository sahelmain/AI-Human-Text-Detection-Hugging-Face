"""
Security enhancements for production Hugging Face deployment.
"""
import re
import hashlib
import time
import streamlit as st
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class SecurityManager:
    """
    Manages security features for the application.
    """
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.input_validator = InputValidator()
        self.session_manager = SessionManager()
    
    def validate_and_sanitize_input(self, text: str) -> Tuple[bool, str]:
        """
        Validate and sanitize user input.
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (is_valid, sanitized_text)
        """
        return self.input_validator.validate_and_sanitize(text)
    
    def check_rate_limit(self, user_id: str, action: str = "prediction") -> bool:
        """
        Check if user is within rate limits.
        
        Args:
            user_id: User identifier
            action: Action being performed
            
        Returns:
            True if within limits, False otherwise
        """
        return self.rate_limiter.check_limit(user_id, action)
    
    def log_security_event(self, event_type: str, details: str, severity: str = "INFO"):
        """
        Log security-related events.
        
        Args:
            event_type: Type of security event
            details: Event details
            severity: Event severity level
        """
        log_message = f"SECURITY [{severity}] {event_type}: {details}"
        
        if severity == "CRITICAL":
            logger.critical(log_message)
        elif severity == "WARNING":
            logger.warning(log_message)
        else:
            logger.info(log_message)

class RateLimiter:
    """
    Simple rate limiter to prevent abuse.
    """
    
    def __init__(self):
        self.requests = defaultdict(lambda: deque())
        self.limits = {
            "prediction": {"count": 100, "window": 3600},  # 100 requests per hour
            "upload": {"count": 20, "window": 3600},       # 20 uploads per hour
            "export": {"count": 10, "window": 3600}        # 10 exports per hour
        }
    
    def check_limit(self, user_id: str, action: str) -> bool:
        """
        Check if user is within rate limits for an action.
        
        Args:
            user_id: User identifier
            action: Action being performed
            
        Returns:
            True if within limits, False otherwise
        """
        if action not in self.limits:
            return True
        
        current_time = time.time()
        limit_config = self.limits[action]
        window_start = current_time - limit_config["window"]
        
        # Clean old requests
        user_requests = self.requests[f"{user_id}_{action}"]
        while user_requests and user_requests[0] < window_start:
            user_requests.popleft()
        
        # Check if within limits
        if len(user_requests) >= limit_config["count"]:
            return False
        
        # Add current request
        user_requests.append(current_time)
        return True
    
    def get_remaining_requests(self, user_id: str, action: str) -> int:
        """Get remaining requests for user action."""
        if action not in self.limits:
            return float('inf')
        
        current_time = time.time()
        limit_config = self.limits[action]
        window_start = current_time - limit_config["window"]
        
        user_requests = self.requests[f"{user_id}_{action}"]
        # Count recent requests
        recent_requests = sum(1 for req_time in user_requests if req_time >= window_start)
        
        return max(0, limit_config["count"] - recent_requests)

class InputValidator:
    """
    Validates and sanitizes user inputs.
    """
    
    def __init__(self):
        self.max_length = 50000  # Maximum text length
        self.min_length = 1      # Minimum text length
        
        # Patterns for potentially malicious content
        self.suspicious_patterns = [
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
            r'javascript:',                                          # JavaScript URLs
            r'on\w+\s*=',                                           # Event handlers
            r'<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>',   # Iframe tags
            r'<object\b[^<]*(?:(?!<\/object>)<[^<]*)*<\/object>',   # Object tags
        ]
        
        # Compile patterns for better performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.suspicious_patterns]
    
    def validate_length(self, text: str) -> Tuple[bool, str]:
        """Validate text length."""
        if len(text) < self.min_length:
            return False, "Text is too short"
        
        if len(text) > self.max_length:
            return False, f"Text exceeds maximum length of {self.max_length} characters"
        
        return True, ""
    
    def detect_suspicious_content(self, text: str) -> List[str]:
        """Detect potentially suspicious content in text."""
        detected = []
        
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                detected.append("Potentially malicious content detected")
                break
        
        return detected
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize input text by removing potentially harmful content.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove HTML/XML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove potential script content
        for pattern in self.compiled_patterns:
            text = pattern.sub('', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def validate_and_sanitize(self, text: str) -> Tuple[bool, str]:
        """
        Validate and sanitize input text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (is_valid, sanitized_text)
        """
        # Basic length validation
        is_valid, error_msg = self.validate_length(text)
        if not is_valid:
            return False, error_msg
        
        # Check for suspicious content
        suspicious_items = self.detect_suspicious_content(text)
        if suspicious_items:
            logger.warning(f"Suspicious content detected: {suspicious_items}")
            # For now, we'll sanitize rather than reject
        
        # Sanitize the text
        sanitized_text = self.sanitize_text(text)
        
        return True, sanitized_text

class SessionManager:
    """
    Manages user sessions and tracks usage.
    """
    
    def __init__(self):
        self.session_data = {}
    
    def get_user_id(self) -> str:
        """Get or create user ID for the session."""
        if 'user_id' not in st.session_state:
            # Create a simple session ID based on session info
            session_info = str(st.session_state.get('session_id', ''))
            user_id = hashlib.md5(session_info.encode()).hexdigest()[:8]
            st.session_state.user_id = user_id
        
        return st.session_state.user_id
    
    def track_usage(self, action: str, details: Optional[str] = None):
        """Track user actions for analytics and security."""
        user_id = self.get_user_id()
        
        if user_id not in self.session_data:
            self.session_data[user_id] = {
                "actions": [],
                "start_time": time.time(),
                "last_activity": time.time()
            }
        
        self.session_data[user_id]["actions"].append({
            "action": action,
            "timestamp": time.time(),
            "details": details
        })
        self.session_data[user_id]["last_activity"] = time.time()

# Global security manager instance
security_manager = SecurityManager()
