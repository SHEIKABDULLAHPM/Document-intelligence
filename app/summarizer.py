from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class DocumentSummarizer:
    """
    Generate concise summaries using transformer models
    """
    def __init__(self, model_name='facebook/bart-large-cnn'):
        """
        Initialize summarization pipeline
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Use BART for summarization
        try:
            self.summarizer = pipeline(
                'summarization',
                model=model_name,
                device=0 if self.device == 'cuda' else -1
            )
        except:
            # Fallback to smaller model if GPU memory is limited
            model_name = 'facebook/bart-base'
            self.summarizer = pipeline(
                'summarization',
                model=model_name,
                device=0 if self.device == 'cuda' else -1
            )
            
    def summarize(self, text, max_length=200, min_length=50):
        """
        Generate a concise summary
        """
        # Truncate if too long (models have token limits)
        if len(text) > 5000:
            text = text[:5000]  # Simple truncation
            
        try:
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Summarization failed: {e}")
            # Fallback to extractive summarization
            return self._extractive_summary(text, max_length)
            
    def _extractive_summary(self, text, max_length):
        """
        Simple extractive summarization as fallback
        """
        from nltk.corpus import stopwords
        from nltk.tokenize import sent_tokenize
        import nltk
        
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        
        sentences = sent_tokenize(text)
        if len(sentences) <= 3:
            return text
            
        # Simple heuristic: score sentences by important words
        stop_words = set(stopwords.words('english'))
        word_freq = {}
        for word in text.lower().split():
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
                
        sentence_scores = []
        for sent in sentences:
            score = sum(word_freq.get(word, 0) for word in sent.lower().split())
            sentence_scores.append((sent, score))
            
        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent for sent, _ in sentence_scores[:5]]
        
        # Reconstruct summary in original order
        summary = []
        for sent in sentences:
            if sent in top_sentences:
                summary.append(sent)
                
        return ' '.join(summary[:3])