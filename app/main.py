import streamlit as st
from datetime import datetime
import tempfile
import os
from pathlib import Path

# Import our modules
from ocr_engine import OCREngine
from text_processor import TextProcessor
from entity_extractor import EntityExtractor
from summarizer import DocumentSummarizer
from qa_system import QASystem

# Page configuration
st.set_page_config(
    page_title="Document Intelligence Assistant",
    page_icon="📄",
    layout="wide"
)

class DocumentIntelligenceApp:
    def __init__(self):
        # Initialize components
        try:
            self.ocr = OCREngine()
        except:
            self.ocr = None
            
        try:
            self.text_processor = TextProcessor()
        except:
            self.text_processor = None
            
        try:
            self.entity_extractor = EntityExtractor()
        except:
            self.entity_extractor = None
            
        try:
            self.summarizer = DocumentSummarizer()
        except:
            self.summarizer = None
            
        # Initialize Q&A system
        try:
            self.qa_system = QASystem()
        except Exception as e:
            st.error(f"Failed to initialize Q&A system: {e}")
            self.qa_system = None
        
        # Session state initialization
        if 'document_text' not in st.session_state:
            st.session_state.document_text = ""
        if 'processed' not in st.session_state:
            st.session_state.processed = False
        if 'summary' not in st.session_state:
            st.session_state.summary = ""
        if 'extracted_info' not in st.session_state:
            st.session_state.extracted_info = None
        if 'qa_history' not in st.session_state:
            st.session_state.qa_history = []
        if 'suggested_questions' not in st.session_state:
            st.session_state.suggested_questions = []
        if 'document_loaded' not in st.session_state:
            st.session_state.document_loaded = False
        if 'current_question' not in st.session_state:
            st.session_state.current_question = ""
        if 'show_answer' not in st.session_state:
            st.session_state.show_answer = False
        if 'last_answer' not in st.session_state:
            st.session_state.last_answer = None
            
    def process_document(self, file):
        """
        Process uploaded document
        """
        if self.ocr is None:
            st.error("OCR engine not available")
            return False
            
        file_extension = Path(file.name).suffix.lower()
        file_path = self._save_temp_file(file)
        
        with st.spinner('Processing document...'):
            try:
                # Extract text
                if file_extension in ['.png', '.jpg', '.jpeg']:
                    text = self.ocr.extract_text_from_image(file_path)
                elif file_extension == '.pdf':
                    text = self.ocr.extract_text_from_pdf(file_path)
                else:
                    st.error(f"Unsupported file format: {file_extension}")
                    return False
                    
                if not text or not text.strip():
                    st.error("No text could be extracted from the document.")
                    return False
                    
                # Clean text
                if self.text_processor:
                    text = self.text_processor.preprocess_text(text)
                
                # Store in session
                st.session_state.document_text = text
                st.session_state.processed = True
                st.session_state.document_loaded = True
                st.session_state.qa_history = []
                st.session_state.current_question = ""
                st.session_state.show_answer = False
                st.session_state.last_answer = None
                
                # Index for Q&A - THIS IS THE CRITICAL STEP
                if self.qa_system:
                    st.info("Indexing document for Q&A...")
                    self.qa_system.index_document(text)
                    if self.qa_system.is_indexed:
                        st.success("✅ Document indexed successfully!")
                        # Generate suggestions
                        st.session_state.suggested_questions = self.qa_system.get_question_suggestions()
                    else:
                        st.warning("⚠️ Document indexing failed. Q&A may not work properly.")
                else:
                    st.error("Q&A system not available")
                
                return True
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                return False
            finally:
                try:
                    os.unlink(file_path)
                except:
                    pass
                
    def _save_temp_file(self, file):
        """
        Save uploaded file temporarily
        """
        with tempfile.NamedTemporaryFile(
            suffix=Path(file.name).suffix,
            delete=False
        ) as tmp_file:
            tmp_file.write(file.getvalue())
            return tmp_file.name
            
    def handle_question(self, question):
        """
        Handle question submission and return answer
        """
        if not st.session_state.processed or not st.session_state.document_loaded:
            return {"error": "Please process a document first!"}
            
        if not question or not question.strip():
            return {"error": "Please enter a question."}
            
        if self.qa_system:
            try:
                # Check if document is indexed
                if not self.qa_system.is_indexed:
                    # Try to re-index
                    st.info("Re-indexing document...")
                    self.qa_system.index_document(st.session_state.document_text)
                    
                result = self.qa_system.answer_question(question)
                
                # Add to history
                st.session_state.qa_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'confidence': result.get('confidence', 0),
                    'context': result.get('context', '')
                })
                
                return result
            except Exception as e:
                return {"error": f"Error: {e}"}
        else:
            return {"error": "Q&A system not available"}
            
    def render(self):
        st.title("📄 AI Document Intelligence Assistant")
        st.markdown("---")
        
        # Sidebar
        with st.sidebar:
            st.header("📤 Upload Document")
            
            uploaded_file = st.file_uploader(
                "Choose a document",
                type=['pdf', 'png', 'jpg', 'jpeg']
            )
            
            if uploaded_file:
                if st.button("Process Document", type="primary", use_container_width=True):
                    if self.process_document(uploaded_file):
                        st.success("✅ Document processed!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to process.")
                        
            # Show document status
            if st.session_state.document_loaded:
                st.markdown("---")
                st.success("📄 Document loaded!")
                st.caption(f"Characters: {len(st.session_state.document_text):,}")
                if st.session_state.processed:
                    # Check if Q&A is ready
                    if self.qa_system and self.qa_system.is_indexed:
                        st.caption("✅ Q&A ready")
                    else:
                        st.caption("⏳ Q&A indexing in progress...")
                        
            st.markdown("---")
            st.markdown("### ℹ️ About")
            st.markdown("""
            - 📝 Extract text via OCR
            - 📋 Generate summaries
            - 🔍 Extract key information
            - 💬 Ask questions about the document
            """)
        
        # Main content - Check if document is loaded
        if not st.session_state.document_loaded:
            st.info("👈 Please upload a document to get started.")
            return
            
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 Document Overview",
            "🔍 Smart Information Extraction",
            "💬 Intelligent Q&A",
            "📊 Document Analytics"
        ])
        
        # Tab 1: Document Overview
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📄 Extracted Text")
                with st.expander("View Full Text", expanded=False):
                    st.text_area(
                        "Document Content",
                        st.session_state.document_text,
                        height=400,
                        disabled=True,
                        key="doc_text_area"
                    )
                    
            with col2:
                st.subheader("📋 Smart Summary")
                if st.button("🤖 Generate AI Summary", use_container_width=True):
                    if self.summarizer:
                        with st.spinner("Generating summary..."):
                            try:
                                summary = self.summarizer.summarize(
                                    st.session_state.document_text,
                                    max_length=300,
                                    min_length=100
                                )
                                st.session_state.summary = summary
                                st.success("Summary generated!")
                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        st.error("Summarizer not available")
                        
                if st.session_state.summary:
                    st.markdown("---")
                    st.markdown("**AI Generated Summary:**")
                    st.markdown(st.session_state.summary)
                    
        # Tab 2: Smart Information Extraction
        with tab2:
            st.subheader("🔍 Intelligent Information Extraction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔎 Extract All Information", use_container_width=True):
                    if self.entity_extractor:
                        with st.spinner("Extracting information..."):
                            try:
                                info = self.entity_extractor.extract_key_information(
                                    st.session_state.document_text
                                )
                                st.session_state.extracted_info = info
                                st.success("✅ Information extracted!")
                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        st.error("Entity extractor not available")
                        
            with col2:
                st.markdown("**📊 Quick Stats**")
                if self.text_processor and st.session_state.document_text:
                    stats = self.text_processor.get_document_stats(
                        st.session_state.document_text
                    )
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Words", stats['word_count'])
                    col_b.metric("Sentences", stats['sentence_count'])
                    col_c.metric("Characters", stats['char_count'])
                    
            if st.session_state.extracted_info:
                info = st.session_state.extracted_info
                
                # Display in expandable sections
                with st.expander("📌 Named Entities", expanded=True):
                    cols = st.columns(3)
                    entity_types = ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY', 'PRODUCT']
                    for i, entity_type in enumerate(entity_types):
                        with cols[i % 3]:
                            if entity_type in info['entities'] and info['entities'][entity_type]:
                                st.markdown(f"**{entity_type}:**")
                                for entity in info['entities'][entity_type][:5]:
                                    st.write(f"• {entity['text']}")
                                    
                with st.expander("🔗 Relationships & Context", expanded=True):
                    if info.get('relationships'):
                        for rel in info['relationships'][:10]:
                            st.write(f"• {rel['subject']} → {rel['verb']} → {rel['object']}")
                    else:
                        st.info("No clear relationships detected")
                        
                with st.expander("📅 Important Dates with Context", expanded=True):
                    if info.get('dates_with_context'):
                        for date_item in info['dates_with_context'][:5]:
                            st.markdown(f"**{date_item['date']}**")
                            st.caption(date_item['context'][:150] + "...")
                            st.markdown("---")
                    else:
                        st.info("No dates found")
                        
                with st.expander("🔑 Key Topics & Keywords", expanded=True):
                    if info.get('keywords'):
                        cols = st.columns(4)
                        for i, (keyword, score) in enumerate(info['keywords'][:12]):
                            with cols[i % 4]:
                                st.metric(keyword, f"{score:.2f}")
                    else:
                        st.info("No keywords extracted")
                        
                with st.expander("📞 Contact Information", expanded=False):
                    if info.get('emails'):
                        st.markdown("**Emails:**")
                        for email in info['emails']:
                            st.write(f"• {email}")
                    if info.get('phone_numbers'):
                        st.markdown("**Phone Numbers:**")
                        for phone in info['phone_numbers']:
                            st.write(f"• {phone}")
                    if info.get('urls'):
                        st.markdown("**URLs:**")
                        for url in info['urls']:
                            st.write(f"• {url}")
                            
                with st.expander("💡 Important Facts", expanded=False):
                    if info.get('important_facts'):
                        for fact in info['important_facts']:
                            st.write(f"• {fact}")
                    else:
                        st.info("No key facts identified")
                        
        # Tab 3: Intelligent Q&A
        with tab3:
            st.subheader("💬 Ask Questions About Your Document")
            
            # Display current document status
            if st.session_state.processed and st.session_state.document_loaded:
                # Check if Q&A is ready
                if self.qa_system and self.qa_system.is_indexed:
                    st.success("✅ Document is indexed and ready for questions!")
                else:
                    st.warning("⚠️ Document is not yet indexed for Q&A. Please wait or re-process.")
            else:
                st.warning("⚠️ Please process a document first.")
                return
            
            # Question input section
            st.markdown("### Type your question below:")
            
            # Use a form to handle submission
            with st.form(key="qa_form", clear_on_submit=True):
                question_text = st.text_input(
                    "Enter your question:",
                    placeholder="e.g., What is the last date mentioned in the document?",
                    label_visibility="collapsed"
                )
                
                submitted = st.form_submit_button("🤖 Ask Question", use_container_width=True)
                
                if submitted and question_text:
                    # Process the question
                    result = self.handle_question(question_text)
                    
                    if result and "error" not in result:
                        # Store the answer in session state
                        st.session_state.last_answer = result
                        st.session_state.show_answer = True
                        st.session_state.current_question = question_text
                    elif result and "error" in result:
                        st.error(result["error"])
            
            # Display answer if available
            if st.session_state.show_answer and st.session_state.last_answer:
                result = st.session_state.last_answer
                
                st.markdown("---")
                st.markdown("### 🤖 Answer")
                
                # Display the question
                st.caption(f"**Question:** {st.session_state.current_question}")
                
                # Display the answer
                st.markdown(f"**{result['answer']}**")
                
                # Display confidence
                if result.get('confidence', 0) > 0.3:
                    st.caption(f"Confidence: {result['confidence']:.2%}")
                else:
                    st.caption("⚠️ Low confidence - this answer may not be accurate")
                    
                # Display context
                if result.get('context'):
                    with st.expander("📖 View Context"):
                        st.markdown(result['context'])
                        
                # Display sources
                if result.get('sources'):
                    with st.expander("📚 Source Excerpts"):
                        for source in result['sources']:
                            st.write(f"• {source}")
            
            # Suggested questions
            st.markdown("---")
            st.markdown("### 💡 Suggested Questions")
            
            if st.session_state.suggested_questions:
                cols = st.columns(2)
                for i, sq in enumerate(st.session_state.suggested_questions):
                    with cols[i % 2]:
                        with st.form(key=f"suggest_form_{i}"):
                            st.text_input("", value=sq, key=f"suggest_hidden_{i}", label_visibility="collapsed")
                            if st.form_submit_button(f"📌 {sq}", use_container_width=True):
                                result = self.handle_question(sq)
                                if result and "error" not in result:
                                    st.session_state.last_answer = result
                                    st.session_state.show_answer = True
                                    st.session_state.current_question = sq
                                    st.rerun()
                                elif result and "error" in result:
                                    st.error(result["error"])
            else:
                st.info("No suggestions available. Try extracting information first.")
            
            # Q&A History
            if st.session_state.qa_history:
                st.markdown("---")
                st.markdown("### 📜 Q&A History")
                
                for i, item in enumerate(reversed(st.session_state.qa_history[-5:])):
                    with st.expander(f"Q: {item['question'][:50]}...", expanded=False):
                        st.markdown(f"**Answer:** {item['answer']}")
                        if item.get('confidence', 0) > 0:
                            st.caption(f"Confidence: {item['confidence']:.2%}")
                        if item.get('context'):
                            st.caption(f"Context: {item['context'][:200]}...")
                            
        # Tab 4: Document Analytics
        with tab4:
            st.subheader("📊 Document Analytics")
            
            if self.text_processor:
                stats = self.text_processor.get_document_stats(st.session_state.document_text)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Words", stats['word_count'])
                col2.metric("Total Sentences", stats['sentence_count'])
                col3.metric("Average Word Length", f"{stats['avg_word_length']:.1f}")
                col4.metric("Characters", stats['char_count'])
                
                # Word frequency
                st.markdown("### 📊 Word Frequency")
                words = st.session_state.document_text.lower().split()
                from collections import Counter
                word_freq = Counter(words).most_common(20)
                
                # Display as a simple bar chart
                st.bar_chart(
                    {word: count for word, count in word_freq if len(word) > 3}
                )
                
                # Readability metrics
                st.markdown("### 📖 Readability Metrics")
                sentences = st.session_state.document_text.split('.')
                avg_sentence_length = stats['word_count'] / max(1, len(sentences))
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Sentences", len(sentences))
                col2.metric("Avg Words/Sentence", f"{avg_sentence_length:.1f}")
                col3.metric("Unique Words", len(set(words)))
                
                if st.button("🔄 Refresh Analytics", use_container_width=True):
                    st.rerun()
                    
        # Footer
        st.markdown("---")
        st.caption("AI Document Intelligence Assistant - Powered by NLP and OCR")

# Run the app
if __name__ == "__main__":
    app = DocumentIntelligenceApp()
    app.render()