import nltk
import ssl
import os

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
def setup_nltk():
    """Download necessary NLTK data for the application."""
    nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    
    # Set NLTK data path
    nltk.data.path.append(nltk_data_path)
    
    # Download required datasets
    datasets = [
        'punkt',
        'stopwords', 
        'vader_lexicon',
        'punkt_tab'
    ]
    
    for dataset in datasets:
        try:
            nltk.download(dataset, download_dir=nltk_data_path, quiet=True)
            print(f"Downloaded {dataset}")
        except Exception as e:
            print(f"Error downloading {dataset}: {e}")

if __name__ == "__main__":
    setup_nltk()
