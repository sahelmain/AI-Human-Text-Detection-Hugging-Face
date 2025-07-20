import streamlit as st

# Page Configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="AI vs Human Text Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import pickle
import os
import re
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import base64
import io
import nltk

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import our utility functions
from utils import (
    extract_text_from_pdf, extract_text_from_docx, 
    extract_text_statistics, analyze_text_features,
    generate_analysis_report, create_downloadable_excel_report,
    create_bull_agent, create_bear_agent, create_supervisor_agent,
    make_prediction, preprocess_text_ml, preprocess_text_dl,
    make_ml_prediction, make_dl_prediction, ensemble_predict_tool
)

# AGGRESSIVE NLTK Data Setup for Hugging Face Spaces
@st.cache_resource
def setup_nltk():
    """Download and setup NLTK requirements with multiple fallbacks for cloud deployment"""
    try:
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Set up NLTK data path for cloud environments
        import os
        nltk_data_paths = [
            '/home/user/nltk_data',
            '/tmp/nltk_data', 
            os.path.expanduser('~/nltk_data'),
            './nltk_data'
        ]
        
        for path in nltk_data_paths:
            if not os.path.exists(path):
                try:
                    os.makedirs(path, exist_ok=True)
                except:
                    continue
            try:
                nltk.data.path.insert(0, path)
            except:
                pass
        
        # Download essential NLTK data with better error handling
        required_packages = ['punkt', 'averaged_perceptron_tagger', 'stopwords', 'wordnet']
        
        for package in required_packages:
            try:
                # Check if already exists
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                for download_dir in [None] + nltk_data_paths:
                    try:
                        if download_dir:
                            nltk.download(package, download_dir=download_dir, quiet=True)
                        else:
                            nltk.download(package, quiet=True)
                        break
                    except:
                        continue
        
        return True
    except Exception as e:
        st.warning(f"⚠️ NLTK setup issue (app will continue): {str(e)[:100]}")
        return False

# Setup NLTK data
try:
    nltk_ready = setup_nltk()
except Exception:
    nltk_ready = False

# Fallback text processing without NLTK
def safe_sentence_tokenize(text):
    """Fallback sentence tokenizer if NLTK fails"""
    try:
        import nltk
        return nltk.sent_tokenize(text)
    except:
        # Simple fallback - split on sentence endings
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

def safe_word_tokenize(text):
    """Fallback word tokenizer if NLTK fails"""
    try:
        import nltk
        return nltk.word_tokenize(text)
    except:
        # Simple fallback - split on whitespace and punctuation
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        return words

def safe_extract_text_statistics(text):
    """Extract text statistics with NLTK fallbacks"""
    stats = {}
    
    # Basic counts
    stats['character_count'] = len(text)
    words = text.split()
    stats['word_count'] = len(words)
    
    # Sentence count with fallback
    try:
        sentences = safe_sentence_tokenize(text)
        stats['sentence_count'] = len(sentences)
    except:
        # Ultimate fallback - count sentence endings
        import re
        stats['sentence_count'] = len(re.findall(r'[.!?]+', text))
    
    stats['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
    
    # Average lengths
    stats['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
    
    try:
        sentences = safe_sentence_tokenize(text)
        stats['avg_sentence_length'] = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
    except:
        stats['avg_sentence_length'] = stats['word_count'] / max(1, stats['sentence_count'])
    
    # Readability scores (simplified versions if textstat fails)
    try:
        import textstat
        if len(text.strip()) > 0:
            stats['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            stats['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            stats['automated_readability_index'] = textstat.automated_readability_index(text)
            stats['coleman_liau_index'] = textstat.coleman_liau_index(text)
            stats['gunning_fog'] = textstat.gunning_fog(text)
            stats['smog_index'] = textstat.smog_index(text)
        else:
            for key in ['flesch_reading_ease', 'flesch_kincaid_grade', 'automated_readability_index', 
                       'coleman_liau_index', 'gunning_fog', 'smog_index']:
                stats[key] = 0
    except:
        # Simple fallback scores
        avg_words_per_sentence = stats['word_count'] / max(1, stats['sentence_count'])
        stats['flesch_reading_ease'] = max(0, 100 - avg_words_per_sentence * 2)
        stats['flesch_kincaid_grade'] = min(20, avg_words_per_sentence * 0.5)
        stats['automated_readability_index'] = stats['flesch_kincaid_grade']
        stats['coleman_liau_index'] = stats['flesch_kincaid_grade']
        stats['gunning_fog'] = stats['flesch_kincaid_grade']
        stats['smog_index'] = stats['flesch_kincaid_grade']
    
    # Lexical diversity
    try:
        word_tokens = safe_word_tokenize(text.lower())
        unique_words = len(set(word_tokens))
        stats['lexical_diversity'] = unique_words / len(word_tokens) if word_tokens else 0
        
        # Most common words
        from collections import Counter
        word_freq = Counter([word for word in word_tokens if word.isalpha()])
        stats['most_common_words'] = word_freq.most_common(10)
    except:
        # Simple fallback
        words_lower = [w.lower() for w in words if w.isalpha()]
        unique_words = len(set(words_lower))
        stats['lexical_diversity'] = unique_words / len(words_lower) if words_lower else 0
        stats['most_common_words'] = []
    
    # Punctuation analysis
    punctuation_marks = ".,!?;:"
    stats['punctuation_count'] = sum(text.count(p) for p in punctuation_marks)
    stats['punctuation_ratio'] = stats['punctuation_count'] / len(text) if len(text) > 0 else 0
    
    return stats

# Import joblib with fallback
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

# Try to import PyTorch for deep learning models with better error handling
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    # Test if torch is working properly
    test_tensor = torch.tensor([1.0])
    TORCH_AVAILABLE = True
except (ImportError, RuntimeError, OSError) as e:
    st.warning(f"PyTorch not available or having issues: {e}. Deep learning models will be disabled.")
    TORCH_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit Cloud Force Styling
st.markdown("""
<style>
    /* Force global background on all Streamlit containers */
    .stApp, .main, .block-container, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 25%, #d946ef 50%, #f97316 75%, #84cc16 100%) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        min-height: 100vh !important;
    }
    
    /* Force remove Streamlit's default background */
    .stApp > div > div > div > div {
        background: transparent !important;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05));
        padding: 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .main-title {
        font-size: 4rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-size: 1.4rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
        margin-bottom: 0;
    }
    
    /* Card Styling */
    .feature-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.1));
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .feature-description {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        line-height: 1.6;
    }
    
    /* Model Cards */
    .model-card {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.2));
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.3));
    }
    
    .model-name {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .model-accuracy {
        font-size: 2rem;
        font-weight: 700;
        color: #10b981;
        margin: 0.5rem 0;
        text-shadow: 0 2px 10px rgba(16, 185, 129, 0.3);
    }
    
    .model-description {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.7);
        line-height: 1.4;
    }
    
    /* Prediction Results */
    .prediction-result {
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 600;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 2px solid;
    }
    
    .ai-prediction {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1));
        color: #fca5a5;
        border-color: #ef4444;
        text-shadow: 0 2px 10px rgba(239, 68, 68, 0.3);
    }
    
    .human-prediction {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(59, 130, 246, 0.1));
        color: #93c5fd;
        border-color: #3b82f6;
        text-shadow: 0 2px 10px rgba(59, 130, 246, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.5) !important;
        background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
        color: white !important;
    }
    
    /* Download Buttons - Special styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3) !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 40px rgba(16, 185, 129, 0.5) !important;
        background: linear-gradient(135deg, #059669, #047857) !important;
        color: white !important;
    }
    
    /* Force download button text to be white - more specific selectors */
    .stDownloadButton button {
        color: white !important;
        background: linear-gradient(135deg, #10b981, #059669) !important;
    }
    
    .stDownloadButton button p {
        color: white !important;
    }
    
    .stDownloadButton button span {
        color: white !important;
    }
    
    .stDownloadButton button div {
        color: white !important;
    }
    
    /* Ensure button text is always white */
    .stButton > button *, .stDownloadButton > button * {
        color: white !important;
    }
    
    /* Force button text visibility - comprehensive */
    button[kind="primary"], button[kind="secondary"] {
        color: white !important;
    }
    
    button[kind="primary"] * {
        color: white !important;
    }
    
    button[kind="secondary"] * {
        color: white !important;
    }
    
    /* All button variants */
    [data-testid="stButton"] button {
        color: white !important;
    }
    
    [data-testid="stDownloadButton"] button {
        color: white !important;
        background: linear-gradient(135deg, #10b981, #059669) !important;
    }
    
    /* Ultra-specific download button text forcing */
    [data-testid="stDownloadButton"] button * {
        color: white !important;
    }
    
    [data-testid="stDownloadButton"] p {
        color: white !important;
    }
    
    [data-testid="stDownloadButton"] span {
        color: white !important;
    }
    
    [data-testid="stDownloadButton"] div {
        color: white !important;
    }
    
    /* Alternative download button styling with darker background for better contrast */
    div[data-testid="stDownloadButton"] button {
        background: #047857 !important;
        color: white !important;
        border: 2px solid #065f46 !important;
    }
    
    div[data-testid="stDownloadButton"] button:hover {
        background: #065f46 !important;
        color: white !important;
    }
    
    /* Force text color on all download button children */
    div[data-testid="stDownloadButton"] * {
        color: white !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
    }
    
    /* Button spans and text content */
    .stButton button span, .stDownloadButton button span {
        color: white !important;
    }
    
    /* Primary button specific */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
    }
    
    /* Force all Streamlit containers to be transparent or gradient */
    [data-testid="stSidebar"], .css-1d391kg, .sidebar, .st-emotion-cache-16idsys {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.3)) !important;
    }
    
    /* Navigation and selectbox styling */
    .stSelectbox > div > div, .st-emotion-cache-1y4p8pa {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: #2c3e50 !important;
    }
    
    /* Selectbox dropdown options */
    .stSelectbox option {
        color: #2c3e50 !important;
        background: white !important;
    }
    
    /* Select dropdown menu */
    .stSelectbox select {
        color: #2c3e50 !important;
        background: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Dropdown list container */
    .stSelectbox > div > div > div {
        background: white !important;
        color: #2c3e50 !important;
    }
    
    /* Dropdown items */
    .stSelectbox li {
        color: #2c3e50 !important;
        background: white !important;
    }
    
    .stSelectbox li:hover {
        color: #2c3e50 !important;
        background: #f8f9fa !important;
    }
    
    /* Select box text and value */
    .stSelectbox div[data-baseweb="select"] {
        color: #2c3e50 !important;
        background: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Select box dropdown arrow and container */
    .stSelectbox [data-baseweb="select"] > div {
        color: #2c3e50 !important;
        background: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Ultra-specific dropdown menu fixes */
    [data-baseweb="select"] {
        background: rgba(255, 255, 255, 0.95) !important;
    }
    
    [data-baseweb="select"] * {
        color: #2c3e50 !important;
    }
    
    /* Dropdown menu popup */
    [data-baseweb="popover"] {
        background: white !important;
    }
    
    [data-baseweb="popover"] * {
        color: #2c3e50 !important;
        background: white !important;
    }
    
    /* Menu items in dropdown */
    [role="option"] {
        color: #2c3e50 !important;
        background: white !important;
    }
    
    [role="option"]:hover {
        color: #2c3e50 !important;
        background: #e9ecef !important;
    }
    
    /* Listbox container */
    [role="listbox"] {
        background: white !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
    }
    
    [role="listbox"] * {
        color: #2c3e50 !important;
    }
    
    /* Comprehensive dropdown text fixes */
    .stSelectbox [data-baseweb="select"] [role="option"] {
        color: #2c3e50 !important;
        background-color: white !important;
    }
    
    .stSelectbox [data-baseweb="select"] [role="listbox"] {
        background-color: white !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    .stSelectbox [data-baseweb="select"] [role="option"]:hover {
        background-color: #f8f9fa !important;
        color: #2c3e50 !important;
    }
    
    /* Additional dropdown targeting */
    .stSelectbox ul[role="listbox"] li {
        color: #2c3e50 !important;
        background-color: white !important;
    }
    
    .stSelectbox ul[role="listbox"] {
        background-color: white !important;
    }
    
    /* Target all dropdown menu items */
    .stSelectbox div[data-baseweb="menu"] div {
        color: #2c3e50 !important;
        background-color: white !important;
    }
    
    /* Target emotion cache classes for dropdown */
    .st-emotion-cache-1p0m4ay, .st-emotion-cache-10oheav, .st-emotion-cache-1erivf3 {
        color: #2c3e50 !important;
        background-color: white !important;
    }
    
    /* Force all select dropdown options */
    .stSelectbox [role="option"], .stSelectbox li {
        color: #2c3e50 !important;
        background-color: white !important;
    }
    
    /* Override any emotion cache dropdown styling */
    .stSelectbox [class*="emotion-cache"] {
        color: #2c3e50 !important;
    }
    
    /* Force all text to be white for visibility */
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: white !important;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05)) !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        color: white !important;
    }
    
    /* Text Areas */
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 16px !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        color: #2c3e50 !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 25px rgba(139, 92, 246, 0.4) !important;
        background: rgba(255, 255, 255, 1) !important;
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: #7f8c8d !important;
    }
    
    /* Disabled text areas (for preview) */
    .stTextArea > div > div > textarea[disabled] {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #34495e !important;
        border: 2px solid rgba(255, 255, 255, 0.4) !important;
    }
    
    /* Text Input Fields */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #2c3e50 !important;
        border-radius: 12px !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    .stTextInput > div > div > input:focus {
        background: rgba(255, 255, 255, 1) !important;
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.3) !important;
    }
    
    /* Number Input Fields */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #2c3e50 !important;
        border-radius: 12px !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    /* General text visibility */
    .stMarkdown p, .stMarkdown div, .stText {
        color: white !important;
    }
    
    /* Override Streamlit default text colors */
    [data-testid="stMarkdownContainer"] p {
        color: white !important;
    }
    
    /* Form labels and help text */
    .stSelectbox label, .stTextArea label, .stTextInput label {
        color: white !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox [data-testid="stMarkdownContainer"] {
        color: white !important;
    }
    
    /* File uploader text */
    [data-testid="stFileUploader"] label {
        color: white !important;
    }
    
    [data-testid="stFileUploader"] div {
        color: white !important;
    }
    
    /* Checkbox and radio text */
    .stCheckbox label, .stRadio label {
        color: white !important;
        font-weight: 500 !important;
    }
    
    /* Checkbox text containers */
    .stCheckbox > label {
        color: white !important;
    }
    
    .stCheckbox > label > div {
        color: white !important;
    }
    
    .stCheckbox span {
        color: white !important;
    }
    
    /* Radio button text containers */
    .stRadio > label {
        color: white !important;
    }
    
    .stRadio > label > div {
        color: white !important;
    }
    
    .stRadio span {
        color: white !important;
    }
    
    /* All checkbox and radio descendant text */
    .stCheckbox *, .stRadio * {
        color: white !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05)) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        border: 2px dashed rgba(255, 255, 255, 0.3) !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
        color: white !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #8b5cf6 !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.1)) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 35px rgba(0,0,0,0.2) !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        border-radius: 10px !important;
    }
    
    /* Tables */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 20px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
    }
    
    .stDataFrame table {
        color: #2c3e50 !important;
    }
    
    .stDataFrame th {
        background: rgba(139, 92, 246, 0.1) !important;
        color: #2c3e50 !important;
    }
    
    .stDataFrame td {
        color: #2c3e50 !important;
    }
    
    /* Sections */
    .section-header {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05));
        padding: 1.5rem;
        border-radius: 20px;
        margin: 2rem 0 1rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: white;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    /* Download Section */
    .download-section {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05));
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.18);
        color: white;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .status-available {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.3), rgba(16, 185, 129, 0.2));
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-unavailable {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.3), rgba(239, 68, 68, 0.2));
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        .main-subtitle {
            font-size: 1.1rem;
        }
        .feature-card {
            padding: 1.5rem;
        }
        .prediction-result {
            font-size: 1.4rem;
            padding: 1.5rem;
        }
    }
    
    /* Force all content areas to inherit gradient background */
    [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"], .st-emotion-cache-1kyxreq {
        background: transparent !important;
    }
    
    /* Ensure metrics and widgets have proper styling */
    [data-testid="metric-container"] > div {
        color: white !important;
    }
    
    /* Force column backgrounds to be transparent */
    [data-testid="column"] {
        background: transparent !important;
    }
    
    /* Override all default Streamlit backgrounds */
    .stApp, .main, .block-container {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 25%, #d946ef 50%, #f97316 75%, #84cc16 100%) !important;
    }
    
    /* Hide Streamlit branding and improve UX */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Additional text color forcing */
    * {
        color: white;
    }
    
    /* Ensure custom HTML elements maintain styling */
    .main-header, .feature-card, .model-card, .prediction-result {
        color: white !important;
    }
    
    /* AGGRESSIVE DROPDOWN TEXT OVERRIDE - Maximum specificity */
    body .stSelectbox * {
        color: #2c3e50 !important;
    }
    
    body .main .stSelectbox [role="option"], 
    body .main .stSelectbox [role="listbox"], 
    body .main .stSelectbox [role="listbox"] *,
    body .main .stSelectbox ul li,
    body .main .stSelectbox div[data-baseweb="select"] *,
    body .main .stSelectbox [data-baseweb="menu"] *,
    body .main [data-testid="stSelectbox"] *,
    body .stApp .stSelectbox * {
        color: #2c3e50 !important;
        background-color: white !important;
    }
    
    /* Force any element inside selectbox containers */
    .stSelectbox, .stSelectbox * {
        color: #2c3e50 !important;
    }
    
    /* Target specific Streamlit classes */
    [class*="selectbox"] [class*="option"],
    [class*="selectbox"] [class*="menu"],
    [class*="selectbox"] li,
    [class*="selectbox"] div {
        color: #2c3e50 !important;
        background-color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# JavaScript to force dropdown text colors
st.components.v1.html("""
<script>
function fixDropdownColors() {
    // Wait for elements to be available
    setTimeout(() => {
        // Target all dropdown elements with multiple selectors
        const selectors = [
            '.stSelectbox [role="option"]',
            '.stSelectbox [role="listbox"] *',
            '.stSelectbox ul li',
            '.stSelectbox div',
            '[data-testid="stSelectbox"] *',
            '[class*="select"] [role="option"]',
            '[aria-expanded="true"] + div div',
            '[data-baseweb="select"] [role="option"]'
        ];
        
        selectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => {
                if (el && el.style) {
                    el.style.setProperty('color', '#2c3e50', 'important');
                    el.style.setProperty('background-color', 'white', 'important');
                }
            });
        });
        
        // Also target any new elements
        const observer = new MutationObserver(mutations => {
            mutations.forEach(mutation => {
                if (mutation.addedNodes) {
                    mutation.addedNodes.forEach(node => {
                        if (node.nodeType === 1) {
                            if (node.closest && node.closest('.stSelectbox')) {
                                node.style.setProperty('color', '#2c3e50', 'important');
                                node.style.setProperty('background-color', 'white', 'important');
                            }
                        }
                    });
                }
            });
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }, 500);
}

// Run multiple times to catch dynamic elements
fixDropdownColors();
setTimeout(fixDropdownColors, 1000);
setTimeout(fixDropdownColors, 2000);
setInterval(fixDropdownColors, 3000);

// Listen for clicks on selectbox to fix immediately
document.addEventListener('click', (e) => {
    if (e.target.closest('.stSelectbox')) {
        setTimeout(fixDropdownColors, 100);
    }
});
</script>
""", height=0)

# Deep Learning Model Classes (same as before)
class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_filters=100, filter_sizes=[3, 4, 5], num_classes=2, dropout=0.5):
        super(CNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            pooled = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        concatenated = torch.cat(conv_outputs, dim=1)
        dropped = self.dropout(concatenated)
        output = self.fc(dropped)
        return output

class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        dropped = self.dropout(final_hidden)
        output = self.fc(dropped)
        return output

class RNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3):
        super(RNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded)
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        dropped = self.dropout(final_hidden)
        output = self.fc(dropped)
        return output

@st.cache_resource
def load_models():
    """Load all trained models and vectorizer"""
    models = {}
    model_status = {}
    
    # Try multiple possible paths for models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        'models',
        os.path.join(current_dir, 'models'),
        os.path.join(current_dir, 'ai_human_detection_project', 'models'),
        'ai_human_detection_project/models'
    ]
    
    models_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            models_dir = path
            break
    
    if models_dir is None:
        st.error("Models directory not found")
        return None, None
    
    try:
        # Load TF-IDF vectorizer (critical for ML models and feature importance)
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            models['vectorizer'] = joblib.load(vectorizer_path)
            model_status['vectorizer'] = True
        
        # Load traditional ML models
        ml_models = ['svm_model', 'decision_tree_model', 'adaboost_model']
        for model_name in ml_models:
            model_path = os.path.join(models_dir, f'{model_name}.pkl')
            if os.path.exists(model_path):
                key = model_name.replace('_model', '')
                models[key] = joblib.load(model_path)
                model_status[key] = True
        
        # Load Deep Learning models if available
        if TORCH_AVAILABLE:
            # Try to load vocabulary mappings
            vocab_to_idx_path = os.path.join(models_dir, 'vocab_to_idx.pkl')
            if os.path.exists(vocab_to_idx_path):
                models['vocab_to_idx'] = joblib.load(vocab_to_idx_path)
                
                # Load model configs
                try:
                    with open(os.path.join(models_dir, 'model_configs.pkl'), 'rb') as f:
                        model_configs = pickle.load(f)
                    models['model_configs'] = model_configs
                    vocab_size = model_configs['vocab_size']
                    
                    # Load deep learning models
                    dl_models = {
                        'cnn': CNNTextClassifier(vocab_size),
                        'lstm': LSTMTextClassifier(vocab_size),
                        'rnn': RNNTextClassifier(vocab_size)
                    }
                    
                    for model_name, model_class in dl_models.items():
                        model_path = os.path.join(models_dir, f'{model_name.upper()}.pkl')
                        if os.path.exists(model_path):
                            try:
                                # Use safer loading with explicit weights_only=False to avoid torch.classes error
                                state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
                                model_class.load_state_dict(state_dict)
                                model_class.eval()
                                models[model_name] = model_class
                                model_status[model_name] = True
                            except Exception as e:
                                st.warning(f"Failed to load {model_name.upper()} model: {e}")
                                # Try alternative loading method for compatibility
                                try:
                                    # Fallback: load with pickle directly
                                    with open(model_path, 'rb') as f:
                                        state_dict = pickle.load(f)
                                    model_class.load_state_dict(state_dict)
                                    model_class.eval()
                                    models[model_name] = model_class
                                    model_status[model_name] = True
                                except Exception as e2:
                                    st.warning(f"All loading methods failed for {model_name.upper()}: {e2}")
                except:
                    pass
        
        return models, model_status
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def preprocess_text_ml(text):
    """Preprocess text for ML models"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def preprocess_text_dl(text, vocab_to_idx, max_seq_length=100):
    """Preprocess text for deep learning models"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    
    indices = []
    for word in words:
        if word in vocab_to_idx:
            indices.append(vocab_to_idx[word])
        else:
            indices.append(vocab_to_idx.get('<UNK>', 1))
    
    if len(indices) < max_seq_length:
        indices.extend([0] * (max_seq_length - len(indices)))
    else:
        indices = indices[:max_seq_length]
    
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

def make_prediction(text, model_name, models):
    """Make prediction using the selected model"""
    try:
        if model_name in ['cnn', 'lstm', 'rnn'] and TORCH_AVAILABLE and 'vocab_to_idx' in models:
            return make_dl_prediction(text, model_name, models)
        else:
            return make_ml_prediction(text, model_name, models)
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None

def make_ml_prediction(text, model_name, models):
    """Make prediction using ML models"""
    processed_text = preprocess_text_ml(text)
    X = models['vectorizer'].transform([processed_text])
    model = models[model_name]
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities)
    return prediction, probabilities, confidence

def make_dl_prediction(text, model_name, models):
    """Make prediction using deep learning models"""
    input_tensor = preprocess_text_dl(text, models['vocab_to_idx'])
    model = models[model_name]
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        
    probs = probabilities.numpy()[0]
    confidence = max(probs)
    return prediction, probs, confidence

def create_confidence_chart(probabilities):
    """Create confidence visualization"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Human-Written', 'AI-Generated'],
            y=[probabilities[0], probabilities[1]],
            marker_color=['#17a2b8', '#ffc107'],
            text=[f'{probabilities[0]:.1%}', f'{probabilities[1]:.1%}'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        yaxis_title="Probability",
        height=400,
        showlegend=False
    )
    
    return fig

def create_text_statistics_chart(stats):
    """Create text statistics visualization"""
    # Create subplots for different metrics
    fig = go.Figure()
    
    # Basic stats
    basic_stats = ['character_count', 'word_count', 'sentence_count', 'paragraph_count']
    basic_values = [stats[key] for key in basic_stats]
    basic_labels = ['Characters', 'Words', 'Sentences', 'Paragraphs']
    
    fig.add_trace(go.Bar(
        x=basic_labels,
        y=basic_values,
        name='Text Counts',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title="Text Statistics Overview",
        xaxis_title="Metrics",
        yaxis_title="Count",
        height=400
    )
    
    return fig

def create_readability_chart(stats):
    """Create readability scores visualization"""
    readability_metrics = ['flesch_reading_ease', 'flesch_kincaid_grade', 'automated_readability_index', 
                          'coleman_liau_index', 'gunning_fog', 'smog_index']
    readability_values = [stats[key] for key in readability_metrics]
    readability_labels = ['Flesch Reading Ease', 'Flesch-Kincaid Grade', 'ARI', 'Coleman-Liau', 'Gunning Fog', 'SMOG']
    
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=readability_values,
            theta=readability_labels,
            fill='toself',
            name='Readability Scores'
        )
    ])
    
    fig.update_layout(
        title="Readability Analysis",
        height=500
    )
    
    return fig

def create_feature_importance_chart(features):
    """Create feature importance visualization"""
    if 'top_tfidf_features' in features and features['top_tfidf_features']:
        feature_names = [f[0] for f in features['top_tfidf_features'][:10]]
        feature_scores = [f[1] for f in features['top_tfidf_features'][:10]]
        
        fig = go.Figure(data=[
            go.Bar(
                x=feature_scores,
                y=feature_names,
                orientation='h',
                marker_color='orange'
            )
        ])
        
        fig.update_layout(
            title="Top TF-IDF Features (Feature Importance)",
            xaxis_title="TF-IDF Score",
            yaxis_title="Features",
            height=400
        )
        
        return fig
    return None

@st.cache_data
def create_wordcloud(text):
    """Create word cloud visualization with caching to prevent rerun issues"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.title('Word Cloud', fontsize=16, pad=20)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")
        return None

def get_download_link(file_bytes, file_name, file_type):
    """Generate download link for files"""
    b64 = base64.b64encode(file_bytes).decode()
    href = f'<a href="data:{file_type};base64,{b64}" download="{file_name}">📥 Download {file_name}</a>'
    return href 

# Main App
# Main Header with Enhanced Styling
st.markdown("""
<div class="main-header" style="
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05));
    padding: 3rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.18);
">
    <h1 style="
        font-size: 4rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    ">🤖 AI vs Human Text Detection</h1>
    <p style="
        font-size: 1.4rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
        margin-bottom: 0;
    ">Advanced Machine Learning & Deep Learning Classification System</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar Navigation
st.sidebar.markdown("""
<div style="
    text-align: center; 
    padding: 1.5rem; 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    border-radius: 15px; 
    margin-bottom: 1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
">
    <h2 style="color: white; margin: 0; font-weight: 600; font-size: 1.3rem;">🧭 Navigation</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Select Page:",
    ["🏠 Home", "🤖 AI Agent Explanation", "🔮 Text Analysis", "📁 File Upload", "⚖️ Model Comparison", "📊 Model Performance", "📈 Advanced Analytics"],
    index=1,
    help="Choose a page to navigate to different features"
)

# Load models
models, model_status = load_models()

if models is None:
    st.error("Failed to load models. Please check that model files are present.")
    st.stop()

# Enhanced Model Status in Sidebar
st.sidebar.markdown("""
<div style="
    text-align: center; 
    padding: 1.5rem; 
    background: rgba(255, 255, 255, 0.1); 
    backdrop-filter: blur(20px);
    border-radius: 20px; 
    margin: 1rem 0;
    border: 1px solid rgba(255, 255, 255, 0.18);
">
    <h3 style="color: white; margin: 0; font-weight: 600; font-size: 1.2rem;">🤖 Available Models</h3>
</div>
""", unsafe_allow_html=True)

available_models = [k for k in models.keys() if k not in ['vectorizer', 'vocab_to_idx', 'model_configs']]

# Enhanced model accuracy mapping with icons and categories
model_info = {
    'svm': {'accuracy': '96.38%', 'icon': '🎯', 'category': 'ML', 'color': '#8b5cf6'},
    'decision_tree': {'accuracy': '84.99%', 'icon': '🌳', 'category': 'ML', 'color': '#06d6a0'}, 
    'adaboost': {'accuracy': '85.50%', 'icon': '🚀', 'category': 'ML', 'color': '#f72585'},
    'cnn': {'accuracy': '97.33%', 'icon': '🔥', 'category': 'DL', 'color': '#ffd60a'},
    'lstm': {'accuracy': '94.52%', 'icon': '🔄', 'category': 'DL', 'color': '#003566'},
    'rnn': {'accuracy': '82.75%', 'icon': '⚡', 'category': 'DL', 'color': '#0077b6'}
}

for model in available_models:
    info = model_info.get(model, {'accuracy': 'N/A', 'icon': '🤖', 'category': 'Other', 'color': '#6c757d'})
    model_display = model.replace('_', ' ').upper()
    
    st.sidebar.markdown(f"""
    <div style="
        position: relative;
        margin: 0.75rem 0;
        padding: 1rem 1.25rem;
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    "
    onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px rgba(0, 0, 0, 0.15)';"
    onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 4px 15px rgba(0, 0, 0, 0.1)';"
    >
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="
                    width: 32px; 
                    height: 32px; 
                    background: linear-gradient(135deg, {info['color']}40, {info['color']}20);
                    border-radius: 8px; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center;
                    font-size: 16px;
                    border: 1px solid {info['color']}60;
                ">
                    {info['icon']}
                </div>
                <div>
                    <div style="color: white; font-weight: 600; font-size: 0.95rem; margin-bottom: 2px;">
                        {model_display}
                    </div>
                    <div style="color: rgba(255, 255, 255, 0.7); font-size: 0.8rem; font-weight: 400;">
                        {info['category']} Model
                    </div>
                </div>
            </div>
            <div style="
                background: linear-gradient(135deg, {info['color']}, {info['color']}80);
                color: white;
                padding: 0.4rem 0.8rem;
                border-radius: 12px;
                font-weight: 700;
                font-size: 0.85rem;
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
                box-shadow: 0 2px 8px {info['color']}30;
            ">
                {info['accuracy']}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# HOME PAGE
if page == "🏠 Home":
    # Welcome Section
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.1));
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
    ">
        <h2 style="
            font-size: 1.5rem;
            font-weight: 600;
            color: white;
            margin-bottom: 1rem;
        ">🎯 Welcome to the Complete AI vs Human Text Detection System</h2>
        <p style="
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.8);
            line-height: 1.6;
        ">
            This comprehensive application uses both traditional machine learning and cutting-edge deep learning to 
            distinguish between AI-generated and human-written text with <strong>up to 97.33% accuracy</strong>.
            Built with 6 advanced models and professional-grade analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features Section
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">✨ Key Features</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3 class="feature-title">📝 Multiple Input Methods</h3>
            <p class="feature-description">
                • Type text directly<br/>
                • Paste content<br/>
                • Upload PDF/Word documents<br/>
                • Batch processing support
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3 class="feature-title">🤖 6 Advanced Models</h3>
            <p class="feature-description">
                • <strong>Deep Learning:</strong> CNN, LSTM, RNN<br/>
                • <strong>Traditional ML:</strong> SVM, Decision Tree, AdaBoost<br/>
                • Choose the best for your needs
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3 class="feature-title">📊 Advanced Analytics</h3>
            <p class="feature-description">
                • Real-time predictions<br/>
                • Confidence scores & visualizations<br/>
                • Feature importance analysis<br/>
                • Comprehensive text statistics
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Performance Overview
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">🏆 Model Performance Overview</h2>
    </div>
    """, unsafe_allow_html=True)
    
    performance_col1, performance_col2 = st.columns(2)
    
    with performance_col1:
        st.markdown("#### 🧠 Deep Learning Models")
        st.markdown("""
        <div class="model-card">
            <div class="model-name">🔥 CNN</div>
            <div class="model-accuracy">97.33%</div>
            <div class="model-description">Best overall performance with convolutional pattern recognition</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="model-card">
            <div class="model-name">🔄 LSTM</div>
            <div class="model-accuracy">94.52%</div>
            <div class="model-description">Excellent for sequential analysis and context understanding</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="model-card">
            <div class="model-name">⚡ RNN</div>
            <div class="model-accuracy">82.75%</div>
            <div class="model-description">Basic recurrent network for text processing</div>
        </div>
        """, unsafe_allow_html=True)
    
    with performance_col2:
        st.markdown("#### ⚙️ Traditional ML Models")
        st.markdown("""
        <div class="model-card">
            <div class="model-name">🎯 SVM</div>
            <div class="model-accuracy">96.38%</div>
            <div class="model-description">Support Vector Machine with robust feature-based classification</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="model-card">
            <div class="model-name">🌳 Decision Tree</div>
            <div class="model-accuracy">84.99%</div>
            <div class="model-description">Most interpretable model with clear decision paths</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="model-card">
            <div class="model-name">🚀 AdaBoost</div>
            <div class="model-accuracy">85.50%</div>
            <div class="model-description">Ensemble boosting method for improved accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">🚀 Quick Start Guide</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div style="padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
                <h4>1. 📝 Text Analysis</h4>
                <p>Navigate to Text Analysis to analyze individual texts with detailed insights</p>
            </div>
            <div style="padding: 1rem; background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); border-radius: 10px; color: white;">
                <h4>2. 📁 File Upload</h4>
                <p>Upload PDF or Word documents for professional document analysis</p>
            </div>
            <div style="padding: 1rem; background: linear-gradient(135deg, #00d2d3 0%, #54a0ff 100%); border-radius: 10px; color: white;">
                <h4>3. ⚖️ Model Comparison</h4>
                <p>Compare all 6 models side-by-side for comprehensive analysis</p>
            </div>
            <div style="padding: 1rem; background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); border-radius: 10px; color: white;">
                <h4>4. 📈 Advanced Analytics</h4>
                <p>Deep dive into text characteristics and AI detection patterns</p>
            </div>
            <div style="padding: 1rem; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); border-radius: 10px; color: #2c3e50;">
                <h4>5. 📥 Download Reports</h4>
                <p>Get comprehensive PDF and Excel analysis reports</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# TEXT ANALYSIS PAGE (Enhanced)
elif page == "🔮 Text Analysis":
    st.markdown("### 📝 Individual Text Analysis")
    
    # Model selection with enhanced information
    model_options = available_models
    model_descriptions = {
        'svm': '🎯 SVM (96.38%) - Support Vector Machine with high accuracy',
        'decision_tree': '🌳 Decision Tree (84.99%) - Most interpretable model',
        'adaboost': '🚀 AdaBoost (85.50%) - Ensemble boosting method',
        'cnn': '🧠 CNN (97.33%) - Best performing deep learning model',
        'lstm': '🔄 LSTM (94.52%) - Sequential pattern recognition',
        'rnn': '⚡ RNN (82.75%) - Basic recurrent neural network'
    }
    
    model_choice = st.selectbox(
        "Choose Model:",
        model_options,
        format_func=lambda x: model_descriptions.get(x, x)
    )
    
    # Text input with enhanced interface
    st.markdown("#### 📝 Enter Your Text")
    text_input = st.text_area(
        "Text to analyze:",
        height=200,
        placeholder="Paste or type the text you want to analyze...",
        help="Enter any text to check if it was written by AI or human"
    )
    
    # Analysis options
    col1, col2 = st.columns(2)
    with col1:
        show_statistics = st.checkbox("📊 Show Text Statistics", value=True)
        show_features = st.checkbox("🔍 Show Feature Analysis", value=True)
    with col2:
        show_wordcloud = st.checkbox("☁️ Generate Word Cloud", value=False)
        enable_download = st.checkbox("📥 Enable Report Download", value=True)
    
    # Initialize session state for analysis results
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'current_text' not in st.session_state:
        st.session_state.current_text = ""
    if 'current_model' not in st.session_state:
        st.session_state.current_model = ""

    if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        if text_input.strip():
            with st.spinner("🔄 Analyzing text..."):
                # Make prediction
                prediction, probabilities, confidence = make_prediction(text_input, model_choice, models)
                
                # Store results in session state
                st.session_state.analysis_results = {
                    'prediction': prediction,
                    'probabilities': probabilities,
                    'confidence': confidence,
                    'text_stats': extract_text_statistics(text_input),
                    'features': analyze_text_features(text_input, models.get('vectorizer'))
                }
                st.session_state.current_text = text_input
                st.session_state.current_model = model_choice
    
    # Display results if they exist in session state
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        prediction = results['prediction']
        probabilities = results['probabilities']
        confidence = results['confidence']
        text_stats = results['text_stats']
        features = results['features']
        text_input = st.session_state.current_text
        
        if prediction is not None:
            # Display main prediction result
            if prediction == 1:  # AI-generated
                st.markdown(
                    f'<div class="prediction-result ai-prediction">🤖 AI-Generated Text Detected<br/>Confidence: {confidence:.2%}</div>',
                    unsafe_allow_html=True
                )
            else:  # Human-written
                st.markdown(
                    f'<div class="prediction-result human-prediction">👤 Human-Written Text Detected<br/>Confidence: {confidence:.2%}</div>',
                    unsafe_allow_html=True
                )
            
            # Enhanced results display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Probability Scores")
                st.metric("Human Probability", f"{probabilities[0]:.2%}")
                st.metric("AI Probability", f"{probabilities[1]:.2%}")
                st.metric("Confidence Score", f"{confidence:.2%}")
            
            with col2:
                st.markdown("#### 📊 Confidence Visualization")
                confidence_chart = create_confidence_chart(probabilities)
                st.plotly_chart(confidence_chart, use_container_width=True)
            
            # Text Statistics Section
            if show_statistics:
                st.markdown("---")
                st.markdown("#### 📊 Text Statistics")
                
                # Basic statistics
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                with stat_col1:
                    st.metric("Characters", text_stats['character_count'])
                with stat_col2:
                    st.metric("Words", text_stats['word_count'])
                with stat_col3:
                    st.metric("Sentences", text_stats['sentence_count'])
                with stat_col4:
                    st.metric("Paragraphs", text_stats['paragraph_count'])
                
                # Advanced statistics
                adv_col1, adv_col2 = st.columns(2)
                with adv_col1:
                    st.plotly_chart(create_text_statistics_chart(text_stats), use_container_width=True)
                with adv_col2:
                    st.plotly_chart(create_readability_chart(text_stats), use_container_width=True)
            
            # Feature Analysis Section
            if show_features:
                st.markdown("---")
                st.markdown("#### 🔍 Feature Analysis")
                
                feature_col1, feature_col2 = st.columns(2)
                with feature_col1:
                    st.markdown("##### 📋 Linguistic Features")
                    st.metric("Average Word Length", f"{features['avg_word_length']:.2f}")
                    st.metric("Average Sentence Length", f"{features['avg_sentence_length']:.2f}")
                    st.metric("Lexical Diversity", f"{features['lexical_diversity']:.3f}")
                    st.metric("Function Word Ratio", f"{features['function_word_ratio']:.3f}")
                
                with feature_col2:
                    feature_importance_chart = create_feature_importance_chart(features)
                    if feature_importance_chart:
                        st.plotly_chart(feature_importance_chart, use_container_width=True)
            
            # Word Cloud
            if show_wordcloud:
                st.markdown("---")
                st.markdown("#### ☁️ Word Cloud")
                wordcloud_fig = create_wordcloud(text_input)
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
            
            # Download Reports Section
            if enable_download:
                st.markdown("---")
                st.markdown('<div class="download-section">', unsafe_allow_html=True)
                st.markdown("#### 📥 Download Comprehensive Reports")
                
                download_col1, download_col2 = st.columns(2)
                
                with download_col1:
                    # Direct PDF download - single step
                    try:
                        prediction_results = {
                            'prediction': prediction,
                            'probabilities': probabilities,
                            'confidence': confidence
                        }
                        pdf_bytes = generate_analysis_report(text_input, prediction_results, text_stats)
                        
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"ai_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            help="Click to download comprehensive PDF analysis report"
                        )
                    except Exception as e:
                        st.error(f"Error generating PDF report: {str(e)}")
                
                with download_col2:
                    # Direct Excel download - single step
                    try:
                        prediction_results = {
                            'prediction': prediction,
                            'probabilities': probabilities,
                            'confidence': confidence
                        }
                        excel_bytes = create_downloadable_excel_report(text_input, prediction_results, text_stats)
                        
                        st.download_button(
                            label="📊 Download Excel Report",
                            data=excel_bytes,
                            file_name=f"ai_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            help="Click to download detailed Excel analysis report"
                        )
                    except Exception as e:
                        st.error(f"Error generating Excel report: {str(e)}")
                
                st.markdown('</div>', unsafe_allow_html=True)

    else:
        if text_input.strip():
            st.info("👆 Click 'Analyze Text' to start the analysis")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

    # Add AI Agent Explanation integration
    if st.session_state.analysis_results is not None:
        if st.button("🤖 Get AI Agent Explanation", type="primary"):
            display_ai_explanation(
                st.session_state.current_text,
                results['prediction'],
                results['probabilities'],
                results['confidence'],
                st.session_state.current_model
            )

# FILE UPLOAD PAGE (New Feature)
elif page == "📁 File Upload":
    st.markdown("### 📁 Document Upload & Analysis")
    st.markdown("Upload PDF or Word documents for AI vs Human text detection analysis.")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt'],
        help="Supported formats: PDF, Word documents (.docx), and plain text files (.txt)"
    )
    
    if uploaded_file is not None:
        # Display file information
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size} bytes",
            "File type": uploaded_file.type
        }
        
        st.markdown("#### 📋 File Information")
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")
        
        # Extract text based on file type
        with st.spinner("📖 Extracting text from document..."):
            if uploaded_file.type == "application/pdf":
                extracted_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                extracted_text = extract_text_from_docx(uploaded_file)
            else:  # txt file
                extracted_text = str(uploaded_file.read(), "utf-8")
        
        if extracted_text and not extracted_text.startswith("Error"):
            st.success(f"✅ Successfully extracted {len(extracted_text)} characters from the document.")
            
            # Show text preview
            st.markdown("#### 👀 Text Preview")
            preview_text = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
            st.text_area("Extracted Text (Preview)", preview_text, height=150, disabled=True)
            
            # Model selection for file analysis
            st.markdown("#### 🤖 Choose Analysis Model")
            
            # Model descriptions for file upload section
            model_descriptions = {
                'svm': '🎯 SVM (96.38%) - Support Vector Machine with high accuracy',
                'decision_tree': '🌳 Decision Tree (84.99%) - Most interpretable model',
                'adaboost': '🚀 AdaBoost (85.50%) - Ensemble boosting method',
                'cnn': '🧠 CNN (97.33%) - Best performing deep learning model',
                'lstm': '🔄 LSTM (94.52%) - Sequential pattern recognition',
                'rnn': '⚡ RNN (82.75%) - Basic recurrent neural network'
            }
            
            file_model_choice = st.selectbox(
                "Select model for analysis:",
                available_models,
                format_func=lambda x: model_descriptions.get(x, x),
                key="file_model"
            )
            
            # Analysis button
            if st.button("🔍 Analyze Document", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing document..."):
                    # Make prediction
                    prediction, probabilities, confidence = make_prediction(extracted_text, file_model_choice, models)
                    
                    if prediction is not None:
                        # Display results
                        if prediction == 1:  # AI-generated
                            st.markdown(
                                f'<div class="prediction-result ai-prediction">🤖 Document contains AI-Generated Text<br/>Confidence: {confidence:.2%}</div>',
                                unsafe_allow_html=True
                            )
                        else:  # Human-written
                            st.markdown(
                                f'<div class="prediction-result human-prediction">👤 Document contains Human-Written Text<br/>Confidence: {confidence:.2%}</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Detailed analysis
                        analysis_col1, analysis_col2 = st.columns(2)
                        
                        with analysis_col1:
                            st.markdown("#### 📊 Analysis Results")
                            st.metric("Human Probability", f"{probabilities[0]:.2%}")
                            st.metric("AI Probability", f"{probabilities[1]:.2%}")
                            st.metric("Overall Confidence", f"{confidence:.2%}")
                        
                        with analysis_col2:
                            confidence_chart = create_confidence_chart(probabilities)
                            st.plotly_chart(confidence_chart, use_container_width=True)
                        
                        # Document statistics
                        st.markdown("---")
                        st.markdown("#### 📈 Document Statistics")
                        doc_stats = safe_extract_text_statistics(extracted_text)
                        
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        with stat_col1:
                            st.metric("Total Characters", doc_stats['character_count'])
                        with stat_col2:
                            st.metric("Total Words", doc_stats['word_count'])
                        with stat_col3:
                            st.metric("Total Sentences", doc_stats['sentence_count'])
                        with stat_col4:
                            st.metric("Reading Level", f"{doc_stats['flesch_kincaid_grade']:.1f}")
                        
                        # Download section for file analysis
                        st.markdown("---")
                        st.markdown("#### 📥 Download Document Analysis Report")
                        
                        prediction_results = {
                            'prediction': prediction,
                            'probabilities': probabilities,
                            'confidence': confidence
                        }
                        
                        report_col1, report_col2 = st.columns(2)
                        
                        with report_col1:
                            # Direct PDF download for file analysis
                            try:
                                pdf_bytes = generate_analysis_report(extracted_text, prediction_results, doc_stats)
                                st.download_button(
                                    label="📄 Download PDF Report",
                                    data=pdf_bytes,
                                    file_name=f"document_analysis_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                    help="Download comprehensive PDF analysis of the document"
                                )
                            except Exception as e:
                                st.error(f"Error generating PDF report: {str(e)}")
                        
                        with report_col2:
                            # Direct Excel download for file analysis
                            try:
                                excel_bytes = create_downloadable_excel_report(extracted_text, prediction_results, doc_stats)
                                st.download_button(
                                    label="📊 Download Excel Report",
                                    data=excel_bytes,
                                    file_name=f"document_analysis_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True,
                                    help="Download detailed Excel analysis of the document"
                                )
                            except Exception as e:
                                st.error(f"Error generating Excel report: {str(e)}")
        else:
            st.error("❌ Failed to extract text from the document. Please check the file format and try again.")

# MODEL COMPARISON PAGE (Enhanced)
elif page == "⚖️ Model Comparison":
    st.markdown("### 🤖 AI Agent Explanation")
    st.markdown("Generate detailed explanations for model predictions.")
    if 'detailed_results' in st.session_state:
        selected_model = st.selectbox("Choose model for explanation:", list(st.session_state.detailed_results.keys()))
        if st.button(f"Generate Explanation for {selected_model}", type="primary"):
            sel_results = st.session_state.detailed_results[selected_model]
            display_ai_explanation(st.session_state.comparison_text, sel_results['prediction'], sel_results['probabilities'], sel_results['confidence'], selected_model)
    else:
        st.info("Run model comparison first to enable explanations.")
    st.markdown("---")

    st.markdown("### ⚖️ Comprehensive Model Comparison")
    st.markdown("Compare the performance of all available models on the same text input.")
    
    # Input methods
    input_method = st.radio("Choose input method:", ["✍️ Type/Paste Text", "📁 Upload File"])
    
    comparison_text = ""
    
    if input_method == "✍️ Type/Paste Text":
        comparison_text = st.text_area(
            "Enter text to compare across all models:",
            height=200,
            placeholder="Enter the text you want to analyze with all models..."
        )
    else:
        uploaded_comparison_file = st.file_uploader(
            "Upload file for comparison",
            type=['pdf', 'docx', 'txt'],
            key="comparison_upload"
        )
        
        if uploaded_comparison_file:
            with st.spinner("Extracting text..."):
                if uploaded_comparison_file.type == "application/pdf":
                    comparison_text = extract_text_from_pdf(uploaded_comparison_file)
                elif uploaded_comparison_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    comparison_text = extract_text_from_docx(uploaded_comparison_file)
                else:
                    comparison_text = str(uploaded_comparison_file.read(), "utf-8")
                
                if comparison_text and not comparison_text.startswith("Error"):
                    st.success(f"✅ Text extracted successfully ({len(comparison_text)} characters)")
                    st.text_area("Extracted text preview:", comparison_text[:300] + "...", height=100, disabled=True)
    
    if st.button("🔀 Compare All Models", type="primary", use_container_width=True):
        if comparison_text.strip():
            with st.spinner("🔄 Running all available models..."):
                model_display_names = {
                    'svm': 'SVM (Traditional ML)',
                    'decision_tree': 'Decision Tree (Traditional ML)',
                    'adaboost': 'AdaBoost (Traditional ML)',
                    'cnn': 'CNN (Deep Learning)',
                    'lstm': 'LSTM (Deep Learning)',
                    'rnn': 'RNN (Deep Learning)'
                }
                
                results = []
                detailed_results = {}
                
                for model_name in available_models:
                    try:
                        pred, probs, conf = make_prediction(comparison_text, model_name, models)
                        if pred is not None:
                            pred_label = "AI-Generated" if pred == 1 else "Human-Written"
                            results.append({
                                'Model': model_display_names.get(model_name, model_name),
                                'Prediction': pred_label,
                                'Confidence': f"{conf:.2%}",
                                'Human Probability': f"{probs[0]:.2%}",
                                'AI Probability': f"{probs[1]:.2%}",
                                'Model Type': 'Deep Learning' if model_name in ['cnn', 'lstm', 'rnn'] else 'Traditional ML'
                            })
                            detailed_results[model_name] = {'prediction': pred, 'probabilities': probs, 'confidence': conf}
                    except Exception as e:
                        st.warning(f"⚠️ Model {model_name} failed: {str(e)}")
                
                if results:
                    # Results table
                    st.markdown("#### 📊 Comparison Results")
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Visual comparison
                    st.markdown("#### 📈 Visual Comparison")
                    
                    # Confidence comparison chart
                    model_names = [r['Model'] for r in results]
                    confidences = [float(r['Confidence'].replace('%', '')) for r in results]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=model_names,
                            y=confidences,
                            text=[f'{c}%' for c in confidences],
                            textposition='auto',
                            marker_color=['#FF6B6B' if 'Deep Learning' in r['Model Type'] else '#4ECDC4' for r in results]
                        )
                    ])
                    
                    fig.update_layout(
                        title="Model Confidence Comparison",
                        xaxis_title="Models",
                        yaxis_title="Confidence (%)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction agreement analysis
                    st.markdown("#### 🤝 Model Agreement Analysis")
                    ai_predictions = sum(1 for r in results if r['Prediction'] == 'AI-Generated')
                    human_predictions = len(results) - ai_predictions
                    
                    agreement_col1, agreement_col2, agreement_col3 = st.columns(3)
                    with agreement_col1:
                        st.metric("AI Predictions", ai_predictions)
                    with agreement_col2:
                        st.metric("Human Predictions", human_predictions)
                    with agreement_col3:
                        consensus = "Strong" if abs(ai_predictions - human_predictions) >= len(results) * 0.6 else "Weak"
                        st.metric("Consensus", consensus)
                    
                    # Download comparison report
                    st.markdown("---")
                    st.markdown("#### 📥 Download Comparison Report")
                    
                    # Direct download for model comparison
                    try:
                        text_stats = safe_extract_text_statistics(comparison_text)
                        # Use the first model's results as primary for report generation
                        primary_result = list(detailed_results.values())[0]
                        prediction_results = {
                            'prediction': primary_result['prediction'],
                            'probabilities': primary_result['probabilities'],
                            'confidence': primary_result['confidence']
                        }
                        
                        excel_bytes = create_downloadable_excel_report(comparison_text, prediction_results, text_stats, results)
                        st.download_button(
                            label="📊 Download Comparison Report",
                            data=excel_bytes,
                            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            help="Download comprehensive comparison report of all models"
                        )
                    except Exception as e:
                        st.error(f"Error generating comparison report: {str(e)}")
                else:
                    st.error("❌ No models were able to make predictions")
        else:
            st.warning("⚠️ Please enter some text or upload a file to compare.")

# MODEL PERFORMANCE PAGE (Enhanced)
elif page == "📊 Model Performance":
    st.markdown("### 📊 Model Performance Metrics & Analysis")
    
    # Performance data with more details
    performance_data = {
        'Model': ['CNN (DL)', 'LSTM (DL)', 'RNN (DL)', 'SVM (ML)', 'Decision Tree (ML)', 'AdaBoost (ML)'],
        'Accuracy': [97.33, 94.52, 82.75, 96.38, 84.99, 85.50],
        'Precision': [97.45, 94.68, 83.12, 96.42, 85.15, 85.68],
        'Recall': [97.21, 94.36, 82.38, 96.34, 84.83, 85.32],
        'F1-Score': [97.33, 94.52, 82.75, 96.38, 84.99, 85.50],
        'Type': ['Deep Learning', 'Deep Learning', 'Deep Learning', 'Traditional ML', 'Traditional ML', 'Traditional ML'],
        'Training Time': ['45 min', '52 min', '28 min', '8 min', '3 min', '12 min'],
        'Prediction Speed': ['Fast', 'Medium', 'Fast', 'Very Fast', 'Instant', 'Fast']
    }
    
    df = pd.DataFrame(performance_data)
    
    # Display performance table
    st.markdown("#### 📋 Detailed Performance Metrics")
    st.dataframe(df, use_container_width=True)
    
    # Performance visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig1 = go.Figure()
        
        dl_models = df[df['Type'] == 'Deep Learning']
        ml_models = df[df['Type'] == 'Traditional ML']
        
        fig1.add_trace(go.Bar(
            name='Deep Learning',
            x=dl_models['Model'],
            y=dl_models['Accuracy'],
            marker_color='#FF6B6B',
            text=[f'{acc}%' for acc in dl_models['Accuracy']],
            textposition='auto'
        ))
        
        fig1.add_trace(go.Bar(
            name='Traditional ML',
            x=ml_models['Model'],
            y=ml_models['Accuracy'],
            marker_color='#4ECDC4',
            text=[f'{acc}%' for acc in ml_models['Accuracy']],
            textposition='auto'
        ))
        
        fig1.update_layout(
            title="Model Accuracy Comparison",
            yaxis_title="Accuracy (%)",
            height=400,
            barmode='group'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Multi-metric radar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig2 = go.Figure()
        
        # Add trace for each model
        colors = ['#FF6B6B', '#FF8E8E', '#FFB1B1', '#4ECDC4', '#70D7D4', '#94E2DF']
        for i, model in enumerate(df['Model']):
            values = [df.iloc[i][metric] for metric in metrics]
            fig2.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=model,
                line_color=colors[i]
            ))
        
        fig2.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[75, 100]
                )),
            showlegend=True,
            title="Multi-Metric Performance Comparison",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Model recommendations
    st.markdown("---")
    st.markdown("#### 🏆 Model Recommendations")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.markdown("""
        **🥇 Best Overall Performance**
        - **CNN**: 97.33% accuracy
        - Best for general use cases
        - Excellent balance of speed and accuracy
        """)
    
    with rec_col2:
        st.markdown("""
        **⚡ Fastest Processing**
        - **Decision Tree**: Instant predictions
        - Good for real-time applications
        - Most interpretable results
        """)
    
    with rec_col3:
        st.markdown("""
        **🎯 Most Reliable**
        - **SVM**: Consistent 96.38% accuracy
        - Robust traditional ML approach
        - Good feature interpretation
        """)

# ADVANCED ANALYTICS PAGE (New Feature)
elif page == "📈 Advanced Analytics":
    st.markdown("### 📈 Advanced Text Analytics & Insights")
    st.markdown("Deep dive into text characteristics and AI detection patterns.")
    
    # Input section
    analytics_text = st.text_area(
        "Enter text for advanced analysis:",
        height=200,
        placeholder="Enter text to perform comprehensive linguistic and statistical analysis..."
    )
    
    if st.button("🔬 Perform Advanced Analysis", type="primary"):
        if analytics_text.strip():
            with st.spinner("🔄 Performing comprehensive analysis..."):
                # Get comprehensive statistics
                text_stats = safe_extract_text_statistics(analytics_text)
                features = analyze_text_features(analytics_text, models.get('vectorizer'))
                
                # Run prediction with best model (CNN if available, otherwise SVM)
                best_model = 'cnn' if 'cnn' in models else 'svm'
                prediction, probabilities, confidence = make_prediction(analytics_text, best_model, models)
                
                # Display main insights
                st.markdown("#### 🎯 Key Insights")
                
                insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
                
                with insight_col1:
                    complexity_score = (text_stats['flesch_reading_ease'] + text_stats['automated_readability_index']) / 2
                    st.metric("Text Complexity", f"{complexity_score:.1f}", help="Based on readability metrics")
                
                with insight_col2:
                    ai_likelihood = "High" if probabilities[1] > 0.7 else "Medium" if probabilities[1] > 0.3 else "Low"
                    st.metric("AI Likelihood", ai_likelihood, f"{probabilities[1]:.1%}")
                
                with insight_col3:
                    writing_style = "Formal" if features['function_word_ratio'] < 0.4 else "Conversational"
                    st.metric("Writing Style", writing_style)
                
                with insight_col4:
                    vocabulary_richness = "Rich" if features['lexical_diversity'] > 0.6 else "Standard"
                    st.metric("Vocabulary", vocabulary_richness, f"{features['lexical_diversity']:.3f}")
                
                # Detailed analytics sections
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistical Analysis", "🔍 Linguistic Features", "📈 Readability Analysis", "🎨 Visual Analysis"])
                
                with tab1:
                    st.markdown("##### 📊 Comprehensive Text Statistics")
                    
                    # Create detailed statistics dataframe
                    stats_df = pd.DataFrame([
                        {"Metric": "Character Count", "Value": text_stats['character_count'], "Description": "Total characters including spaces"},
                        {"Metric": "Word Count", "Value": text_stats['word_count'], "Description": "Total number of words"},
                        {"Metric": "Sentence Count", "Value": text_stats['sentence_count'], "Description": "Total number of sentences"},
                        {"Metric": "Paragraph Count", "Value": text_stats['paragraph_count'], "Description": "Total number of paragraphs"},
                        {"Metric": "Avg Word Length", "Value": f"{text_stats['avg_word_length']:.2f}", "Description": "Average characters per word"},
                        {"Metric": "Avg Sentence Length", "Value": f"{text_stats['avg_sentence_length']:.2f}", "Description": "Average words per sentence"},
                        {"Metric": "Lexical Diversity", "Value": f"{text_stats['lexical_diversity']:.3f}", "Description": "Ratio of unique words to total words"},
                        {"Metric": "Punctuation Ratio", "Value": f"{text_stats['punctuation_ratio']:.3f}", "Description": "Punctuation marks per character"}
                    ])
                    
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Most common words
                    if text_stats['most_common_words']:
                        st.markdown("##### 📝 Most Frequent Words")
                        common_words_df = pd.DataFrame(text_stats['most_common_words'], columns=['Word', 'Frequency'])
                        
                        fig = go.Figure(data=[
                            go.Bar(x=common_words_df['Word'], y=common_words_df['Frequency'], marker_color='lightcoral')
                        ])
                        fig.update_layout(title="Top 10 Most Frequent Words", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.markdown("##### 🔍 Advanced Linguistic Features")
                    
                    feature_metrics = {
                        "Stylistic Features": {
                            "Function Word Ratio": f"{features['function_word_ratio']:.3f}",
                            "Punctuation Density": f"{features['punctuation_ratio']:.3f}",
                            "Uppercase Ratio": f"{features['uppercase_ratio']:.3f}",
                            "Digit Ratio": f"{features['digit_ratio']:.3f}"
                        },
                        "Complexity Indicators": {
                            "Average Word Length": f"{features['avg_word_length']:.2f} chars",
                            "Average Sentence Length": f"{features['avg_sentence_length']:.2f} words",
                            "Vocabulary Richness": f"{features['lexical_diversity']:.3f}",
                            "Character Density": f"{features['char_count'] / max(1, features['word_count']):.2f} chars/word"
                        }
                    }
                    
                    feat_col1, feat_col2 = st.columns(2)
                    
                    with feat_col1:
                        st.markdown("**Stylistic Features**")
                        for feature, value in feature_metrics["Stylistic Features"].items():
                            st.metric(feature, value)
                    
                    with feat_col2:
                        st.markdown("**Complexity Indicators**")
                        for feature, value in feature_metrics["Complexity Indicators"].items():
                            st.metric(feature, value)
                
                with tab3:
                    st.markdown("##### 📈 Readability & Complexity Analysis")
                    
                    # Readability scores interpretation
                    readability_scores = {
                        "Flesch Reading Ease": {
                            "score": text_stats['flesch_reading_ease'],
                            "interpretation": "Very Easy" if text_stats['flesch_reading_ease'] > 90 else 
                                           "Easy" if text_stats['flesch_reading_ease'] > 80 else
                                           "Fairly Easy" if text_stats['flesch_reading_ease'] > 70 else
                                           "Standard" if text_stats['flesch_reading_ease'] > 60 else
                                           "Fairly Difficult" if text_stats['flesch_reading_ease'] > 50 else
                                           "Difficult"
                        },
                        "Flesch-Kincaid Grade": {
                            "score": text_stats['flesch_kincaid_grade'],
                            "interpretation": f"Grade {text_stats['flesch_kincaid_grade']:.1f} level"
                        },
                        "Gunning Fog Index": {
                            "score": text_stats['gunning_fog'],
                            "interpretation": f"Grade {text_stats['gunning_fog']:.1f} level"
                        }
                    }
                    
                    for metric, data in readability_scores.items():
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric(metric, f"{data['score']:.2f}")
                        with col2:
                            st.info(f"**Interpretation:** {data['interpretation']}")
                    
                    # Readability radar chart
                    st.plotly_chart(create_readability_chart(text_stats), use_container_width=True)
                
                with tab4:
                    st.markdown("##### 🎨 Visual Text Analysis")
                    
                    visual_col1, visual_col2 = st.columns(2)
                    
                    with visual_col1:
                        # Word cloud
                        st.markdown("**Word Cloud Visualization**")
                        wordcloud_fig = create_wordcloud(analytics_text)
                        if wordcloud_fig:
                            st.pyplot(wordcloud_fig)
                    
                    with visual_col2:
                        # Feature importance (if available)
                        feature_chart = create_feature_importance_chart(features)
                        if feature_chart:
                            st.plotly_chart(feature_chart, use_container_width=True)
                        else:
                            st.info("Feature importance analysis requires TF-IDF vectorizer")
                
                # AI Detection Summary
                st.markdown("---")
                st.markdown("#### 🤖 AI Detection Summary")
                
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    prediction_label = "AI-Generated" if prediction == 1 else "Human-Written"
                    st.markdown(f"""
                    **🎯 Final Prediction:** {prediction_label}  
                    **🔒 Confidence:** {confidence:.2%}  
                    **🤖 AI Probability:** {probabilities[1]:.2%}  
                    **👤 Human Probability:** {probabilities[0]:.2%}  
                    **🧠 Model Used:** {best_model.upper()}
                    """)
                
                with summary_col2:
                    confidence_chart = create_confidence_chart(probabilities)
                    st.plotly_chart(confidence_chart, use_container_width=True)
        else:
            st.warning("⚠️ Please enter some text for analysis.")

# AI AGENT EXPLANATION PAGE (New Feature)
elif page == "🤖 AI Agent Explanation":
    st.markdown("### 🤖 AI Agent Explanation")
    st.markdown("Get AI-generated explanations for text classifications using an LLM agent.")

    agent_text = st.text_area("Enter text for explained analysis:", height=200)

    if st.button("Generate Explanation", type="primary"):
        if agent_text.strip():
            with st.spinner("Generating explanation..."):
                # Get prediction from CNN if available, else SVM
                agent_model = 'cnn' if 'cnn' in models else 'svm'
                prediction, probabilities, confidence = make_prediction(agent_text, agent_model, models)
                
                pred_label = "AI-Generated" if prediction == 1 else "Human-Written"
                ai_prob = probabilities[1] if len(probabilities) > 1 else confidence
                
                try:
                    llm = ChatOpenAI(
                        openai_api_key=st.secrets.get("OPENAI_API_KEY", ""),
                        model="gpt-3.5-turbo",
                        temperature=0.7
                    )
                    
                    prompt = PromptTemplate(
                        input_variables=["text", "prediction", "confidence"],
                        template="You are an expert in AI text detection. Explain in detail why this text is likely {prediction} with {confidence}% confidence. Highlight key linguistic patterns, style elements, and characteristics that led to this classification. Text: {text}"
                    )
                    
                    chain = LLMChain(llm=llm, prompt=prompt)
                    
                    explanation = chain.run(
                        text=agent_text[:2000],  # Limit length for LLM
                        prediction=pred_label,
                        confidence=f"{ai_prob:.2%}"
                    )
                    
                    st.markdown("#### 📝 AI Agent Explanation")
                    st.write(explanation)
                    
                    st.markdown(f"**Base Prediction:** {pred_label} ({ai_prob:.2%} AI probability)")
                    st.markdown(f"**Model Used:** {agent_model.upper()}")
                except KeyError:
                    st.error("⚠️ OpenAI API key not found. Please add it in Streamlit secrets.")
                except Exception as e:
                    st.error(f"Error generating explanation: {str(e)}")
        else:
            st.warning("Please enter some text.")

    # Insert committee button after existing button
    if st.button("Run Detection Committee", type="primary"):
        if agent_text.strip():
            with st.spinner("Committee analyzing..."):
                try:
                    # First show the actual model predictions
                    st.markdown("### 🤖 Model Predictions")
                    ensemble_tool = ensemble_predict_tool(models)
                    ensemble_result = ensemble_tool.func(agent_text)
                    
                    # Parse and display results in a user-friendly format
                    if "Details:" in ensemble_result:
                        # Extract the ensemble summary
                        summary_part = ensemble_result.split("Details:")[0].strip()
                        st.markdown(f"**{summary_part}**")
                        
                        # Extract and parse the details
                        details_part = ensemble_result.split("Details:")[1].strip()
                        try:
                            import ast
                            details_dict = ast.literal_eval(details_part)
                            
                            # Create a formatted table
                            model_data = []
                            for model_name, results in details_dict.items():
                                model_display = model_name.upper().replace('_', ' ')
                                prediction_icon = "🤖" if results['prediction'] == 'AI' else "👤"
                                model_data.append({
                                    'Model': model_display,
                                    'Prediction': f"{prediction_icon} {results['prediction']}",
                                    'Confidence': f"{results['confidence']:.1%}",
                                    'AI Probability': f"{results['ai_probability']:.1%}"
                                })
                            
                            # Display as a clean dataframe
                            df = pd.DataFrame(model_data)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            
                        except Exception as e:
                            # If parsing fails, try to manually run ensemble and create table
                            try:
                                # Get direct ensemble results
                                ensemble_tool = ensemble_predict_tool(models)
                                results = {}
                                ai_probs = []
                                
                                for model_name, model in models.items():
                                    if model_name in ['vectorizer', 'vocab_to_idx', 'model_configs']:
                                        continue
                                    try:
                                        prediction, probabilities, confidence = make_prediction(agent_text, model_name, models)
                                        results[model_name] = {
                                            'prediction': 'AI' if prediction == 1 else 'Human',
                                            'ai_probability': probabilities[1] if len(probabilities) > 1 else confidence,
                                            'confidence': confidence
                                        }
                                        ai_probs.append(results[model_name]['ai_probability'])
                                    except:
                                        continue
                                
                                if results:
                                    # Create table from direct results
                                    model_data = []
                                    for model_name, res in results.items():
                                        model_display = model_name.upper().replace('_', ' ')
                                        prediction_icon = "🤖" if res['prediction'] == 'AI' else "👤"
                                        model_data.append({
                                            'Model': model_display,
                                            'Prediction': f"{prediction_icon} {res['prediction']}",
                                            'Confidence': f"{res['confidence']:.1%}",
                                            'AI Probability': f"{res['ai_probability']:.1%}"
                                        })
                                    
                                    df = pd.DataFrame(model_data)
                                    st.dataframe(df, use_container_width=True, hide_index=True)
                                    
                                    # Show ensemble summary
                                    if ai_probs:
                                        avg_ai_prob = sum(ai_probs) / len(ai_probs)
                                        ensemble_pred = 'AI' if avg_ai_prob > 0.5 else 'Human'
                                        st.markdown(f"**Ensemble Results: Average AI Probability: {avg_ai_prob:.2%}, Prediction: {ensemble_pred}**")
                                else:
                                    st.write(ensemble_result)
                            except:
                                # Ultimate fallback
                                st.write(ensemble_result)
                    else:
                        st.write(ensemble_result)
                    
                    st.markdown("---")
                    
                    # Then run the committee analysis
                    from utils import create_bull_agent, create_bear_agent, create_supervisor_agent
                    bull = create_bull_agent(models)
                    bear = create_bear_agent(models)
                    supervisor = create_supervisor_agent(bull, bear, models)
                    
                    response = supervisor.invoke({"messages": [{"role": "user", "content": f"Analyze if this text is AI or human: {agent_text}"}]})
                    
                    st.markdown("### 🗣️ Committee Debate & Decision")
                    
                    # Extract the combined analysis
                    if 'messages' in response and response['messages']:
                        combined = response['messages'][-1].get('content') if isinstance(response['messages'][-1], dict) else response['messages'][-1].content
                        # Split into sections
                        sections = combined.split('\n\n')
                        current_section = None
                        content_dict = {'Bull Agent': '', 'Bear Agent': '', 'Supervisor Decision': ''}
                        for section in sections:
                            if section.startswith('Bull Agent'):
                                current_section = 'Bull Agent'
                                content_dict[current_section] += section + '\n\n'
                            elif section.startswith('Bear Agent'):
                                current_section = 'Bear Agent'
                                content_dict[current_section] += section + '\n\n'
                            elif section.startswith('Supervisor Decision'):
                                current_section = 'Supervisor Decision'
                                content_dict[current_section] += section + '\n\n'
                            elif current_section:
                                content_dict[current_section] += section + '\n\n'
                        # Display in expanders
                        with st.expander("🐂 Bull Agent (Pro-Human Analysis)", expanded=False):
                            st.markdown(content_dict['Bull Agent'])
                        with st.expander("🐻 Bear Agent (Pro-AI Analysis)", expanded=False):
                            st.markdown(content_dict['Bear Agent'])
                        with st.expander("⚖️ Supervisor Decision", expanded=True):
                            st.markdown(content_dict['Supervisor Decision'])
                    else:
                        st.error("No response from committee.")
                except Exception as e:
                    st.error(f"Error during committee analysis: {str(e)}")
        else:
            st.warning("Please enter some text.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>🤖 AI vs Human Text Detection System</strong></p>
    <p>Enhanced with Deep Learning | Built with Streamlit | Comprehensive Analysis & Reporting</p>
    <p>Features: File Upload • Advanced Analytics • Model Comparison • Report Generation</p>
</div>
""", unsafe_allow_html=True) 

def display_ai_explanation(text, prediction, probabilities, confidence, model_name):
    """Display AI agent explanation for text classification"""

    pred_label = "AI-Generated" if prediction == 1 else "Human-Written"
    ai_prob = probabilities[1] if len(probabilities) > 1 else confidence

    with st.expander("🤖 Detailed AI Agent Explanation", expanded=True):
        try:
            llm = ChatOpenAI(
                openai_api_key=st.secrets.get("OPENAI_API_KEY", ""),
                model="gpt-3.5-turbo",
                temperature=0.7
            )

            prompt = PromptTemplate(
                input_variables=["text", "prediction", "confidence", "model_name"],
                template="You are an expert in AI text detection. Using the {model_name} model, explain in detail why this text is likely {prediction} with {confidence}% confidence. Highlight key linguistic patterns, style elements, and characteristics that led to this classification. Text: {text}"
            )

            chain = LLMChain(llm=llm, prompt=prompt)

            explanation = chain.run(
                text=text[:2000],
                prediction=pred_label,
                confidence=f"{ai_prob:.2%}",
                model_name=model_name.upper()
            )

            # Split explanation into parts if needed for display
            explanation_parts = explanation.split("\n\n")  # Split on double newlines for paragraphs

            for part in explanation_parts:
                if part.strip() == "":
                    st.write("")
                else:
                    st.markdown(part)

        except KeyError:
            st.error("⚠️ OpenAI API key not found. Please add it in Streamlit secrets.")
        except Exception as e:
            st.error(f"Error generating explanation: {str(e)}")
