# Deployment Guide for Hugging Face Spaces

## Quick Deploy to Hugging Face Spaces

1. **Create a new Space on Hugging Face:**
   - Go to https://huggingface.co/new-space
   - Choose "Streamlit" as the SDK
   - Select the appropriate hardware (CPU should be sufficient)

2. **Clone and push this repository:**
   ```bash
   git clone https://github.com/sahelmain/AI-Human-Text-Detection-Hugging-Face.git
   cd AI-Human-Text-Detection-Hugging-Face
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push hf main
   ```

3. **Space Configuration:**
   The app is configured with the proper metadata in README.md:
   - SDK: streamlit
   - Python version: 3.8+
   - Entry point: app.py

## Features Ready for Deployment

✅ **Models**: All pre-trained models included
✅ **Dependencies**: Optimized requirements.txt
✅ **NLTK Data**: Automatic download on startup
✅ **UI Configuration**: Streamlit config optimized
✅ **Performance**: Cached model loading

## Hardware Requirements

- **CPU**: 2 cores minimum (recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for models and dependencies

## Environment Variables (Optional)

Create a `.env` file based on `.env.example` if you need to customize:
- OPENAI_API_KEY (for AI agent features)
- Model cache settings
- Performance optimizations

The app will work without these variables using fallback configurations.
