import pandas as pd
import numpy as np
import re
import nltk
import textstat
from collections import Counter
import PyPDF2
import docx
from fpdf import FPDF
import io
import base64
import pickle
from datetime import datetime
import torch
import torch.nn as nn

# NLTK Setup and Error Handling
def setup_nltk_safely():
    """Setup NLTK data with error handling"""
    try:
        # Try to download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        return True
    except Exception as e:
        print(f"NLTK setup failed: {e}")
        return False

# Safe tokenization functions with fallbacks
def safe_sentence_tokenize(text):
    """Safely tokenize sentences with fallback"""
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        # Fallback: simple sentence splitting
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception:
        # Ultimate fallback
        return text.split('. ')

def safe_word_tokenize(text):
    """Safely tokenize words with fallback"""
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        # Fallback: simple word splitting
        return text.split()
    except Exception:
        return text.split()

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

# Initialize NLTK
setup_nltk_safely()

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(uploaded_file):
    """Extract text from Word document"""
    try:
        doc = docx.Document(uploaded_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading Word document: {str(e)}"

def extract_text_statistics(text):
    """Extract comprehensive text statistics"""
    
    stats = {}
    
    # Basic counts
    stats['character_count'] = len(text)
    stats['word_count'] = len(text.split())
    stats['sentence_count'] = len(nltk.sent_tokenize(text))
    stats['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
    
    # Average lengths
    words = text.split()
    sentences = nltk.sent_tokenize(text)
    
    stats['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
    stats['avg_sentence_length'] = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
    
    # Readability scores
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
    
    # Lexical diversity
    unique_words = len(set(words))
    stats['lexical_diversity'] = unique_words / len(words) if words else 0
    
    # Most common words
    word_freq = Counter([word.lower() for word in words if word.isalpha()])
    stats['most_common_words'] = word_freq.most_common(10)
    
    # Punctuation analysis
    punctuation_marks = ".,!?;:"
    stats['punctuation_count'] = sum(text.count(p) for p in punctuation_marks)
    stats['punctuation_ratio'] = stats['punctuation_count'] / len(text) if len(text) > 0 else 0
    
    return stats

def analyze_text_features(text, vectorizer=None):
    """Analyze text features for machine learning interpretation"""
    
    features = {}
    
    # Basic linguistic features using safe tokenization
    words = safe_word_tokenize(text)
    sentences = safe_sentence_tokenize(text)
    
    # Length features
    features['char_count'] = len(text)
    features['word_count'] = len(words)
    features['sentence_count'] = len(sentences)
    features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
    features['avg_sentence_length'] = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
    
    # Readability features
    if len(text.strip()) > 0:
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        features['automated_readability_index'] = textstat.automated_readability_index(text)
    
    # Stylistic features
    features['punctuation_ratio'] = sum(text.count(p) for p in ".,!?;:") / len(text) if len(text) > 0 else 0
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0
    
    # Lexical diversity
    unique_words = len(set([word.lower() for word in words if word.isalpha()]))
    features['lexical_diversity'] = unique_words / len(words) if words else 0
    
    # Function words ratio (common AI indicators)
    function_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'}
    function_word_count = sum(1 for word in words if word.lower() in function_words)
    features['function_word_ratio'] = function_word_count / len(words) if words else 0
    
    # If vectorizer is available, get TF-IDF feature importance
    if vectorizer:
        try:
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            # Transform the text
            tfidf_matrix = vectorizer.transform([text])
            # Get non-zero features and their scores
            feature_scores = tfidf_matrix.toarray()[0]
            
            # Get top features
            top_indices = np.argsort(feature_scores)[-20:][::-1]  # Top 20 features
            top_features = [(feature_names[i], feature_scores[i]) for i in top_indices if feature_scores[i] > 0]
            
            features['top_tfidf_features'] = top_features
        except:
            features['top_tfidf_features'] = []
    
    return features

def generate_analysis_report(text, prediction_results, text_stats, model_comparison=None):
    """Generate a comprehensive PDF report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    
    # Title
    pdf.cell(0, 10, "AI vs Human Text Detection Report", 0, 1, 'C')
    pdf.ln(10)
    
    # Timestamp
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(5)
    
    # Text Preview
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Text Sample (First 200 characters):", 0, 1)
    pdf.set_font("Arial", size=10)
    preview_text = text[:200] + "..." if len(text) > 200 else text
    # Clean text for PDF - remove problematic characters
    clean_preview = preview_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, clean_preview)
    pdf.ln(5)
    
    # Prediction Results
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Prediction Results:", 0, 1)
    pdf.set_font("Arial", size=10)
    
    prediction_text = "AI-Generated" if prediction_results['prediction'] == 1 else "Human-Written"
    pdf.cell(0, 5, f"Prediction: {prediction_text}", 0, 1)
    pdf.cell(0, 5, f"Confidence: {prediction_results['confidence']:.2%}", 0, 1)
    pdf.cell(0, 5, f"Human Probability: {prediction_results['probabilities'][0]:.2%}", 0, 1)
    pdf.cell(0, 5, f"AI Probability: {prediction_results['probabilities'][1]:.2%}", 0, 1)
    pdf.ln(10)
    
    # Text Statistics
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Text Statistics:", 0, 1)
    pdf.set_font("Arial", size=10)
    
    pdf.cell(0, 5, f"Character Count: {text_stats['character_count']}", 0, 1)
    pdf.cell(0, 5, f"Word Count: {text_stats['word_count']}", 0, 1)
    pdf.cell(0, 5, f"Sentence Count: {text_stats['sentence_count']}", 0, 1)
    pdf.cell(0, 5, f"Average Word Length: {text_stats['avg_word_length']:.2f}", 0, 1)
    pdf.cell(0, 5, f"Average Sentence Length: {text_stats['avg_sentence_length']:.2f}", 0, 1)
    pdf.cell(0, 5, f"Lexical Diversity: {text_stats['lexical_diversity']:.3f}", 0, 1)
    pdf.ln(5)
    
    # Readability Scores
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Readability Scores:", 0, 1)
    pdf.set_font("Arial", size=10)
    
    pdf.cell(0, 5, f"Flesch Reading Ease: {text_stats['flesch_reading_ease']:.2f}", 0, 1)
    pdf.cell(0, 5, f"Flesch-Kincaid Grade: {text_stats['flesch_kincaid_grade']:.2f}", 0, 1)
    pdf.cell(0, 5, f"Automated Readability Index: {text_stats['automated_readability_index']:.2f}", 0, 1)
    pdf.cell(0, 5, f"Coleman-Liau Index: {text_stats['coleman_liau_index']:.2f}", 0, 1)
    pdf.cell(0, 5, f"Gunning Fog: {text_stats['gunning_fog']:.2f}", 0, 1)
    pdf.cell(0, 5, f"SMOG Index: {text_stats['smog_index']:.2f}", 0, 1)
    pdf.ln(10)
    
    # Model Comparison (if provided)
    if model_comparison:
        pdf.set_font("Arial", "B", size=12)
        pdf.cell(0, 10, "Model Comparison Results:", 0, 1)
        pdf.set_font("Arial", size=10)
        
        for result in model_comparison:
            pdf.cell(0, 5, f"{result['Model']}: {result['Prediction']} ({result['Confidence']})", 0, 1)
    
    # Convert to bytes - fpdf2 returns bytes directly
    try:
        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, str):
            # Older version returns string
            pdf_bytes = pdf_output.encode('latin-1')
        elif isinstance(pdf_output, bytearray):
            # Convert bytearray to bytes for Streamlit compatibility
            pdf_bytes = bytes(pdf_output)
        else:
            # Newer version returns bytes directly
            pdf_bytes = pdf_output
        return pdf_bytes
    except Exception as e:
        # Fallback - create a simple bytes response
        return b"PDF generation error occurred"

def create_downloadable_excel_report(text, prediction_results, text_stats, model_comparison=None):
    """Create downloadable Excel report with multiple sheets"""
    
    # Create Excel writer
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        
        # Summary sheet
        summary_data = {
            'Metric': ['Prediction', 'Confidence', 'Human Probability', 'AI Probability'],
            'Value': [
                'AI-Generated' if prediction_results['prediction'] == 1 else 'Human-Written',
                f"{prediction_results['confidence']:.2%}",
                f"{prediction_results['probabilities'][0]:.2%}",
                f"{prediction_results['probabilities'][1]:.2%}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Text statistics sheet
        stats_data = {
            'Statistic': list(text_stats.keys()),
            'Value': [str(v) if not isinstance(v, list) else str(v[:5]) for v in text_stats.values()]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Text_Statistics', index=False)
        
        # Model comparison sheet (if available)
        if model_comparison:
            comparison_df = pd.DataFrame(model_comparison)
            comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
    
    return output.getvalue() 

from langgraph.prebuilt import create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.chains import LLMChain
from langchain.tools import Tool
from langgraph.graph import StateGraph, MessagesState, START, END

# Prompts for agents
BULL_PROMPT = "You are an optimistic analyst arguing why the provided text is human-written. The text to analyze will be in the user message after 'Analyze why this text is human-written:'. First, extract that text. Then, use the ensemble_predict tool on that text to get predictions from all 6 ML/DL models. Then, use safe_extract_text_statistics on that text for stats. Base your argument on the model results favoring human-written, highlighting positive patterns like diversity and natural flow."

BEAR_PROMPT = "You are a pessimistic analyst arguing why the provided text is AI-generated. The text to analyze will be in the user message after 'Analyze why this text is AI-generated:'. First, extract that text. Then, use the ensemble_predict tool on that text to get predictions from all 6 ML/DL models. Then, use safe_extract_text_statistics on that text for stats. Base your argument on the model results favoring AI-generated, highlighting risks like repetition and uniform structure."

SUPERVISOR_PROMPT = PromptTemplate(
    input_variables=["bull_analysis", "bear_analysis", "text"],
    template="You are a neutral supervisor analyzing if a text is AI-generated or human-written. \n\nBull (pro-human) analysis: {bull_analysis}\n\nBear (pro-AI) analysis: {bear_analysis}\n\nText: {text}\n\nFirst, use the ensemble_predict tool to get predictions from all available models. Then, weigh both sides incorporating the model predictions, and provide a final decision with confidence level and detailed explanation."
)

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
        if model_name in ['cnn', 'lstm', 'rnn'] and 'vocab_to_idx' in models:
            return make_dl_prediction(text, model_name, models)
        else:
            return make_ml_prediction(text, model_name, models)
    except Exception as e:
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

def ensemble_predict_tool(models):
    def predict_func(text):
        results = {}
        ai_probs = []
        for model_name, model in models.items():
            if model_name in ['vectorizer', 'vocab_to_idx', 'model_configs']:
                continue
            try:
                prediction, probabilities, confidence = make_prediction(text, model_name, models)
                results[model_name] = {
                    'prediction': 'AI' if prediction == 1 else 'Human',
                    'ai_probability': probabilities[1] if len(probabilities) > 1 else confidence,
                    'confidence': confidence
                }
                ai_probs.append(results[model_name]['ai_probability'])
            except Exception as e:
                print(f"Error predicting with {model_name}: {str(e)}")
        if not ai_probs:
            return "No models could make predictions successfully."
        avg_ai_prob = sum(ai_probs) / len(ai_probs)
        ensemble_pred = 'AI' if avg_ai_prob > 0.5 else 'Human'
        return f"Ensemble Results: Average AI Probability: {avg_ai_prob:.2%}, Prediction: {ensemble_pred}. Details: {results}"
    return Tool(
        name="ensemble_predict",
        func=predict_func,
        description="Run ensemble predictions using all available ML/DL models on the input text and return averaged results with details."
    )

def create_bull_agent(models=None):
    llm = ChatOpenAI(openai_api_key=st.secrets.get("OPENAI_API_KEY", ""), model="gpt-3.5-turbo", temperature=0.7)
    tools = [safe_extract_text_statistics]
    if models:
        tools.append(ensemble_predict_tool(models))
    return create_react_agent(llm, tools=tools, prompt=BULL_PROMPT, name="bull")

def create_bear_agent(models=None):
    llm = ChatOpenAI(openai_api_key=st.secrets.get("OPENAI_API_KEY", ""), model="gpt-3.5-turbo", temperature=0.7)
    tools = [safe_extract_text_statistics]
    if models:
        tools.append(ensemble_predict_tool(models))
    return create_react_agent(llm, tools=tools, prompt=BEAR_PROMPT, name="bear")

def create_supervisor_agent(bull, bear, models=None):
    llm = ChatOpenAI(openai_api_key=st.secrets.get("OPENAI_API_KEY", ""), model="gpt-3.5-turbo", temperature=0.3)
    
    # Simple graph with supervisor calling bull and bear
    def supervisor_node(state):
        try:
            messages = state.get("messages", [])
            if not messages:
                return {"messages": [{"role": "assistant", "content": "No input text provided."}]}
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                input_text = last_message.content
            elif isinstance(last_message, dict):
                input_text = last_message.get("content", "")
            else:
                input_text = str(last_message)
            
            # Get analyses from both agents
            bull_result = bull.invoke({"messages": [{"role": "user", "content": f"Analyze why this text is human-written: {input_text}"}]})
            bear_result = bear.invoke({"messages": [{"role": "user", "content": f"Analyze why this text is AI-generated: {input_text}"}]})
            
            # Extract content from AIMessage objects
            bull_content = bull_result["messages"][-1].content if hasattr(bull_result["messages"][-1], 'content') else str(bull_result["messages"][-1])
            bear_content = bear_result["messages"][-1].content if hasattr(bear_result["messages"][-1], 'content') else str(bear_result["messages"][-1])
            
            # Get ensemble predictions if models available
            ensemble_result = ""
            if models:
                try:
                    ensemble_tool = ensemble_predict_tool(models)
                    ensemble_result = ensemble_tool.func(input_text)
                except Exception as e:
                    ensemble_result = f"Ensemble prediction failed: {str(e)}"
            
            # Create supervisor decision using LLM chain
            supervisor_prompt = PromptTemplate(
                input_variables=["bull_analysis", "bear_analysis", "ensemble_result", "text"],
                template="You are a neutral supervisor making a final decision based ONLY on the ensemble model predictions.\n\nEnsemble model predictions: {ensemble_result}\n\nRULE: Extract the 'Average AI Probability' from the ensemble results above. If it is greater than 50%, conclude 'AI-generated'. If it is less than 50%, conclude 'human-written'. Do not consider any other factors.\n\nBull argument: {bull_analysis}\nBear argument: {bear_analysis}\n\nYour response format:\n1. State the ensemble average AI probability\n2. Apply the rule (>50% = AI, <50% = Human)\n3. Give your final decision\n4. Briefly explain why the models reached this conclusion\n\nText analyzed: {text}"
            )
            
            chain = LLMChain(llm=llm, prompt=supervisor_prompt)
            supervisor_decision = chain.run(
                bull_analysis=bull_content,
                bear_analysis=bear_content,
                ensemble_result=ensemble_result,
                text=input_text[:2000]
            )
            
            combined_analysis = f"Bull Agent (Human-focused): {bull_content}\n\nBear Agent (AI-focused): {bear_content}\n\nSupervisor Decision: {supervisor_decision}"
            return {"messages": [{"role": "assistant", "content": combined_analysis}]}
        except Exception as e:
            return {"messages": [{"role": "assistant", "content": f"Error in supervisor analysis: {str(e)}"}]}
    
    graph = StateGraph(MessagesState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", END)
    return graph.compile() 