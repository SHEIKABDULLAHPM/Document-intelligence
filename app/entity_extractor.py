import spacy
import re
from datetime import datetime
from typing import Dict, List, Tuple

class EntityExtractor:
    """
    Extract named entities and key information
    """
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
            
    def extract_entities(self, text: str) -> Dict:
        """
        Extract entities using spaCy NER
        """
        doc = self.nlp(text)
        
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geo-political entities (locations)
            'DATE': [],
            'MONEY': [],
            'PERCENT': [],
            'FAC': [],  # Facilities
            'PRODUCT': [],
            'EVENT': [],
            'LAW': [],
            'LANGUAGE': [],
            'WORK_OF_ART': []
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append({
                    'text': ent.text,  # Fixed: Added comma here
                    'start': ent.start_char,
                    'end': ent.end_char
                })
                
        # Extract additional dates using regex
        date_patterns = self._get_date_patterns()
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Check if match already exists
                exists = False
                for existing in entities['DATE']:
                    if existing['text'] == match:
                        exists = True
                        break
                if not exists:
                    entities['DATE'].append({'text': match, 'start': -1, 'end': -1})
                    
        return entities
    
    def _get_date_patterns(self) -> List[str]:
        """
        Common date patterns for extraction
        """
        return [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
            r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}',  # DD MMM YYYY
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2,4}',  # MMM DD, YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
        ]
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract important keywords using TF-IDF
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            vectorizer = TfidfVectorizer(
                max_features=top_n,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
            
            if len(sentences) < 2:
                # If only one sentence, use words
                words = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
                # Count word frequencies
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                return [(word, score/len(words)) for word, score in sorted_words[:top_n]]
                
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF score for each term
            avg_tfidf = tfidf_matrix.mean(axis=0).tolist()[0]
            term_scores = list(zip(feature_names, avg_tfidf))
            term_scores.sort(key=lambda x: x[1], reverse=True)
            
            return term_scores[:top_n]
        except Exception as e:
            print(f"Keyword extraction error: {e}")
            # Fallback: simple word frequency
            words = [token.text.lower() for token in self.nlp(text) if not token.is_stop and not token.is_punct]
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [(word, score/len(words)) for word, score in sorted_words[:top_n]]
    
    def extract_key_information(self, text: str) -> Dict:
        """
        Comprehensive key information extraction
        """
        entities = self.extract_entities(text)
        keywords = self.extract_keywords(text)
        
        # Extract emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        
        # Extract phone numbers
        phone_patterns = [
            r'\+\d{1,3}\s?\d{1,14}',  # International
            r'\d{3}[-.]?\d{3}[-.]?\d{4}',  # US format
            r'\(\d{3}\)\s?\d{3}[-.]?\d{4}',  # (XXX) XXX-XXXX
        ]
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, text))
            
        # Extract URLs
        urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text)
        
        return {
            'entities': entities,
            'keywords': keywords,
            'emails': list(set(emails)),
            'phone_numbers': list(set(phones)),
            'urls': list(set(urls)),
            'dates': entities['DATE']
        }