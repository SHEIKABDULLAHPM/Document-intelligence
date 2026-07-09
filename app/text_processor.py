import spacy
import nltk
from nltk.tokenize import sent_tokenize
import re

class TextProcessor:
    """
    NLP pipeline for text processing
    """
    def __init__(self):
        # Download NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
            
    def preprocess_text(self, text):
        """
        Clean and normalize text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep structure
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        
        return text.strip()
    
    def segment_sentences(self, text):
        """
        Split text into sentences
        """
        return sent_tokenize(text)
    
    def get_document_stats(self, text):
        """
        Get statistics about the document
        """
        doc = self.nlp(text)
        
        stats = {
            'word_count': len([token for token in doc if not token.is_punct]),
            'sentence_count': len(list(doc.sents)),
            'char_count': len(text),
            'avg_word_length': sum(len(token.text) for token in doc if not token.is_punct) / 
                               max(1, len([token for token in doc if not token.is_punct]))
        }
        
        return stats