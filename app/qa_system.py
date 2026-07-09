from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QASystem:
    """
    Answer questions about the document using semantic search
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # For semantic search and context retrieval
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded SentenceTransformer successfully")
        except Exception as e:
            logger.error(f"Error loading sentence transformer: {e}")
            self.encoder = None
        
        # For answer extraction
        try:
            self.qa_pipeline = pipeline(
                'question-answering',
                model='distilbert-base-cased-distilled-squad',
                device=0 if self.device == 'cuda' else -1
            )
            logger.info("Loaded QA pipeline successfully")
        except Exception as e:
            logger.error(f"Error loading QA model: {e}")
            try:
                self.qa_pipeline = pipeline(
                    'question-answering',
                    model='distilbert-base-uncased-distilled-squad',
                    device=0 if self.device == 'cuda' else -1
                )
                logger.info("Loaded fallback QA pipeline")
            except:
                self.qa_pipeline = None
                logger.error("Failed to load any QA model")
            
        self.context_chunks = []
        self.chunk_embeddings = None
        self.is_indexed = False
        
    def index_document(self, text):
        """
        Index document for semantic search with better chunking
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for indexing")
            self.is_indexed = False
            return
            
        logger.info(f"Indexing document of length: {len(text)}")
        
        # Intelligent chunking
        chunk_size = 500  # Smaller chunks for better precision
        overlap = 100
        
        # Split by sentences for better semantic chunks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Add overlapping context
        overlapping_chunks = []
        for i in range(len(chunks)):
            if i > 0:
                # Add previous chunk's last 100 chars for context
                prev_context = chunks[i-1][-100:] if len(chunks[i-1]) > 100 else chunks[i-1]
                overlapping_chunks.append(prev_context + " " + chunks[i])
            else:
                overlapping_chunks.append(chunks[i])
        
        self.context_chunks = overlapping_chunks
        logger.info(f"Created {len(self.context_chunks)} chunks")
        
        # Create embeddings
        if self.encoder and self.context_chunks:
            try:
                self.chunk_embeddings = self.encoder.encode(
                    self.context_chunks,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                self.is_indexed = True
                logger.info(f"Successfully indexed {len(self.context_chunks)} chunks")
            except Exception as e:
                logger.error(f"Error creating embeddings: {e}")
                self.chunk_embeddings = None
                self.is_indexed = False
        else:
            self.is_indexed = False
            
    def answer_question(self, question, top_k=3):
        """
        Answer a question using retrieved context
        """
        # Check if document is indexed
        if not self.is_indexed:
            logger.warning("Document not indexed")
            return {
                'answer': "Please process a document first.",
                'confidence': 0,
                'context': None
            }
            
        if not self.context_chunks:
            logger.warning("No context chunks available")
            return {
                'answer': "Please process a document first.",
                'confidence': 0,
                'context': None
            }
            
        try:
            logger.info(f"Processing question: {question[:50]}...")
            
            # Encode question
            if self.encoder:
                try:
                    question_embedding = self.encoder.encode(
                        question,
                        convert_to_tensor=True,
                        show_progress_bar=False
                    )
                    
                    # Find most relevant chunks
                    cos_scores = util.pytorch_cos_sim(question_embedding, self.chunk_embeddings)[0]
                    top_results = torch.topk(cos_scores, k=min(top_k, len(self.context_chunks)))
                    
                    # Get the best chunks
                    best_chunks = []
                    for idx in top_results[1]:
                        best_chunks.append(self.context_chunks[idx.item()])
                    
                    best_score = top_results[0][0].item()
                    best_chunk = best_chunks[0] if best_chunks else None
                    
                    logger.info(f"Best chunk score: {best_score}")
                    
                    # Check if we found relevant content
                    if best_score < 0.3 or not best_chunk:
                        return {
                            'answer': "I couldn't find relevant information to answer this question.",
                            'confidence': 0,
                            'context': None
                        }
                        
                except Exception as e:
                    logger.error(f"Semantic search error: {e}")
                    # Fallback to simple chunk selection
                    best_chunk = self.context_chunks[0] if self.context_chunks else None
            else:
                # Fallback: use first chunk
                best_chunk = self.context_chunks[0] if self.context_chunks else None
            
            if not best_chunk:
                return {
                    'answer': "No relevant information found.",
                    'confidence': 0,
                    'context': None
                }
                
            # Get answer using QA model
            if self.qa_pipeline:
                try:
                    answer = self.qa_pipeline(
                        question=question,
                        context=best_chunk
                    )
                    
                    logger.info(f"QA model score: {answer['score']}")
                    
                    if answer['score'] > 0.1 and answer['answer']:
                        # Find context around the answer
                        context_excerpt = self._get_context_excerpt(best_chunk, answer['answer'])
                        
                        return {
                            'answer': answer['answer'],
                            'confidence': answer['score'],
                            'context': context_excerpt,
                            'sources': [chunk[:200] + "..." for chunk in best_chunks[:2]]
                        }
                    else:
                        # QA model uncertain, return the relevant context
                        return {
                            'answer': f"Found relevant information: {best_chunk[:200]}...",
                            'confidence': 0.2,
                            'context': best_chunk
                        }
                        
                except Exception as e:
                    logger.error(f"QA pipeline error: {e}")
                    return {
                        'answer': f"Found relevant information: {best_chunk[:200]}...",
                        'confidence': 0.2,
                        'context': best_chunk
                    }
            else:
                # No QA pipeline, return context
                return {
                    'answer': f"Found relevant information: {best_chunk[:200]}...",
                    'confidence': 0.2,
                    'context': best_chunk
                }
                
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return {
                'answer': "Sorry, I encountered an error processing your question.",
                'confidence': 0,
                'context': None
            }
            
    def _get_context_excerpt(self, context: str, answer: str) -> str:
        """
        Get surrounding context for the answer
        """
        if not context or not answer:
            return ""
            
        # Find answer in context and get surrounding text
        answer_index = context.find(answer)
        if answer_index != -1:
            start = max(0, answer_index - 100)
            end = min(len(context), answer_index + len(answer) + 100)
            excerpt = context[start:end]
            
            # Clean up
            excerpt = excerpt.replace('\n', ' ')
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(context):
                excerpt = excerpt + "..."
                
            return excerpt
            
        return context[:200] + "..."
        
    def get_question_suggestions(self) -> list:
        """
        Generate suggested questions based on document content
        """
        if not self.is_indexed or not self.context_chunks:
            return [
                "What is the main topic of this document?",
                "What are the key dates mentioned?",
                "Who are the key people mentioned?",
                "What is the most important information?",
                "What actions are mentioned in the document?"
            ]
            
        suggestions = []
        
        # Extract entities from the first few chunks
        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')
            
            # Use first 3 chunks for entity extraction
            text_sample = " ".join(self.context_chunks[:3])
            doc = nlp(text_sample[:5000])
            
            entities = {}
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                if len(entities[ent.label_]) < 3:
                    entities[ent.label_].append(ent.text)
                    
            # Generate questions based on entities
            if entities.get('DATE'):
                suggestions.append(f"What happened on {entities['DATE'][0]}?")
            if entities.get('PERSON'):
                suggestions.append(f"Who is {entities['PERSON'][0]}?")
            if entities.get('ORG'):
                suggestions.append(f"What does {entities['ORG'][0]} do?")
            if entities.get('GPE'):
                suggestions.append(f"Where is {entities['GPE'][0]}?")
                
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            
        # Add generic suggestions
        generic = [
            "What is the main topic of this document?",
            "What are the key takeaways?",
            "What is the most important information?",
            "Can you summarize the key points?"
        ]
        
        # Add unique suggestions
        for g in generic:
            if g not in suggestions:
                suggestions.append(g)
                
        return suggestions[:5]