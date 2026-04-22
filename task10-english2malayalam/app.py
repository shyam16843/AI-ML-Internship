import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="English-Malayalam Translator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .translation-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class EnglishMalayalamTranslator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-ml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
    def translate(self, text: str, max_length: int = 512) -> str:
        """Translate English text to Malayalam"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                              padding=True, max_length=max_length).to(self.device)
        
        with torch.no_grad():
            translated = self.model.generate(**inputs, max_length=max_length)
        
        translation = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        return translation
    
    def translate_batch(self, texts: List[str], max_length: int = 512) -> List[str]:
        """Translate multiple English texts to Malayalam"""
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, 
                              padding=True, max_length=max_length).to(self.device)
        
        with torch.no_grad():
            translated = self.model.generate(**inputs, max_length=max_length)
        
        translations = [self.tokenizer.decode(t, skip_special_tokens=True) 
                       for t in translated]
        return translations
    
    def translate_with_confidence(self, text: str, max_length: int = 512) -> Dict:
        """Translate with additional information and confidence metrics"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                              padding=True, max_length=max_length).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_length=max_length,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        translation = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Calculate basic confidence
        if hasattr(outputs, 'scores') and outputs.scores:
            scores = torch.stack(outputs.scores, dim=1)
            probabilities = torch.softmax(scores, dim=-1)
            max_probs = torch.max(probabilities, dim=-1).values
            confidence = torch.mean(max_probs).item()
        else:
            confidence = 0.7
        
        return {
            'english': text,
            'malayalam': translation,
            'confidence': round(confidence, 3),
            'token_count': len(inputs['input_ids'][0]),
            'method': 'direct_translation'
        }

class EnhancedEnglishMalayalamTranslator:
    def __init__(self):
        self.translator = EnglishMalayalamTranslator()
        
    def assess_translation_quality(self, english_text: str, malayalam_text: str) -> Dict:
        """Assess the quality of translation"""
        quality_score = 0.0
        issues = []
        
        if len(malayalam_text.strip()) == 0:
            quality_score = 0.0
            issues.append("Empty translation")
        elif malayalam_text.lower() == english_text.lower():
            quality_score = 0.1
            issues.append("Translation identical to input")
        else:
            quality_score = 0.7
            
            if '[' in malayalam_text and ']' in malayalam_text:
                quality_score -= 0.2
                issues.append("Contains untranslated tokens")
            
            if len(malayalam_text) < len(english_text) * 0.3:
                quality_score -= 0.1
                issues.append("Translation too short")
            
            if len(malayalam_text) > len(english_text) * 3:
                quality_score -= 0.1
                issues.append("Translation too long")
        
        return {
            'score': max(0.0, min(1.0, quality_score)),
            'issues': issues,
            'grade': self._score_to_grade(quality_score)
        }
    
    def _score_to_grade(self, score: float) -> str:
        if score >= 0.9: return 'A+'
        elif score >= 0.8: return 'A'
        elif score >= 0.7: return 'B'
        elif score >= 0.6: return 'C'
        elif score >= 0.5: return 'D'
        else: return 'F'
    
    def translate_with_analysis(self, text: str) -> Dict:
        """Translate with comprehensive analysis"""
        translation_result = self.translator.translate_with_confidence(text)
        quality_assessment = self.assess_translation_quality(
            translation_result['english'], 
            translation_result['malayalam']
        )
        
        comprehensive_result = {**translation_result, **quality_assessment}
        return comprehensive_result

# Initialize translator
@st.cache_resource
def load_translator():
    with st.spinner("Loading translation model..."):
        translator = EnhancedEnglishMalayalamTranslator()
        return translator

def main():
    # Header
    st.markdown('<div class="main-header">🌍 English-Malayalam Translator</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    st.sidebar.subheader("Translation Options")
    
    max_length = st.sidebar.slider("Maximum Translation Length", 64, 512, 256)
    show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
    show_quality = st.sidebar.checkbox("Show Quality Assessment", value=True)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Single Translation", "Batch Translation", "Sample Translations", "About"])
    
    with tab1:
        st.header("Single Text Translation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input English Text")
            english_text = st.text_area(
                "Enter English text to translate:",
                height=150,
                placeholder="Type your English text here...",
                key="single_input"
            )
            
            translate_button = st.button("Translate", type="primary", key="single_translate")
        
        with col2:
            st.subheader("Malayalam Translation")
            
            if translate_button and english_text:
                try:
                    translator = load_translator()
                    result = translator.translate_with_analysis(english_text)
                    
                    st.markdown('<div class="translation-box">', unsafe_allow_html=True)
                    st.write("**Translation:**")
                    st.success(result['malayalam'])
                    
                    if show_confidence:
                        confidence_class = "confidence-high" if result['confidence'] > 0.8 else "confidence-medium" if result['confidence'] > 0.6 else "confidence-low"
                        st.markdown(f'<p class="{confidence_class}">Confidence: {result["confidence"]}</p>', unsafe_allow_html=True)
                    
                    if show_quality:
                        st.write(f"Quality Score: {result['score']:.3f} ({result['grade']})")
                        if result['issues']:
                            st.warning(f"Issues: {', '.join(result['issues'])}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Translation error: {str(e)}")
            else:
                st.info("Enter text and click 'Translate' to see the Malayalam translation")
    
    with tab2:
        st.header("Batch Translation")
        
        st.subheader("Input Multiple Texts")
        batch_input = st.text_area(
            "Enter multiple English texts (one per line):",
            height=200,
            placeholder="Type each English sentence on a separate line...",
            key="batch_input"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            batch_translate = st.button("Translate Batch", type="primary", key="batch_translate")
        
        if batch_translate and batch_input:
            texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
            
            if texts:
                try:
                    translator = load_translator()
                    
                    with st.spinner(f"Translating {len(texts)} sentences..."):
                        results = []
                        for text in texts:
                            result = translator.translate_with_analysis(text)
                            results.append(result)
                    
                    # Display results in a table
                    st.subheader("Translation Results")
                    
                    results_df = pd.DataFrame(results)
                    display_df = results_df[['english', 'malayalam', 'confidence', 'score', 'grade']]
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Translations", len(results))
                    with col2:
                        st.metric("Average Confidence", f"{results_df['confidence'].mean():.3f}")
                    with col3:
                        st.metric("Average Quality", f"{results_df['score'].mean():.3f}")
                    with col4:
                        grade_counts = results_df['grade'].value_counts()
                        most_common_grade = grade_counts.index[0] if not grade_counts.empty else "N/A"
                        st.metric("Most Common Grade", most_common_grade)
                    
                    # Download option
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="english_malayalam_translations.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Batch translation error: {str(e)}")
            else:
                st.warning("Please enter at least one text to translate.")
    
    with tab3:
        st.header("Sample Translations")
        
        sample_sentences = [
            "Hello, how are you?",
            "Good morning, have a nice day!",
            "Thank you for your help",
            "Where is the nearest hospital?",
            "I love machine learning and artificial intelligence",
            "Kerala is known for its beautiful backwaters",
            "What is your name?",
            "Can you please help me with this?"
        ]
        
        st.subheader("Try these sample sentences:")
        
        cols = st.columns(2)
        for i, sentence in enumerate(sample_sentences):
            with cols[i % 2]:
                if st.button(sentence, key=f"sample_{i}", use_container_width=True):
                    st.session_state.single_input = sentence
                    st.rerun()
        
        st.subheader("Quick Translation Demo")
        if st.button("Run Sample Translation Demo"):
            translator = load_translator()
            
            progress_bar = st.progress(0)
            results = []
            
            for i, sentence in enumerate(sample_sentences):
                result = translator.translate_with_analysis(sentence)
                results.append(result)
                progress_bar.progress((i + 1) / len(sample_sentences))
            
            # Display results
            for i, result in enumerate(results):
                with st.expander(f"Sample {i+1}: {result['english']}"):
                    st.write(f"**Malayalam:** {result['malayalam']}")
                    st.write(f"**Confidence:** {result['confidence']}")
                    st.write(f"**Quality:** {result['score']:.3f} ({result['grade']})")
    
    with tab4:
        st.header("About This App")
        
        st.markdown("""
        ### English-Malayalam Translation System
        
        This web application provides high-quality translation from English to Malayalam 
        using state-of-the-art neural machine translation.
        
        **Features:**
        - 🚀 Fast and accurate translations
        - 📊 Confidence scoring for each translation
        - 🔍 Quality assessment with grading
        - 📁 Batch translation for multiple texts
        - 💾 Download results as CSV
        
        **Technical Details:**
        - **Model:** Helsinki-NLP/opus-mt-en-ml
        - **Framework:** PyTorch with Transformers
        - **Interface:** Streamlit
        
        **How it works:**
        1. Enter English text in the input area
        2. Click the translate button
        3. Get instant Malayalam translation with quality metrics
        
        The system uses a sequence-to-sequence transformer model specifically trained 
        for English to Malayalam translation tasks.
        """)
        
        # System info
        st.subheader("System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**PyTorch Version:** {torch.__version__}")
            st.write(f"**Device:** {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
        
        with col2:
            translator = load_translator()
            st.write(f"**Model:** Helsinki-NLP/opus-mt-en-ml")
            st.write(f"**Vocabulary Size:** {translator.translator.tokenizer.vocab_size}")

if __name__ == "__main__":
    main()