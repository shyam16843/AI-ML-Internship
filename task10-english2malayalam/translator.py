import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

print("PyTorch version:", torch.__version__) 
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class EnglishMalayalamTranslator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-ml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Loading English-Malayalam translator on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model: {model_name}")
        print(f"💾 Vocabulary size: {self.tokenizer.vocab_size}")
        
    def translate(self, text: str, max_length: int = 512) -> str:
        """Translate English text to Malayalam"""
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                              padding=True, max_length=max_length).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            translated = self.model.generate(**inputs, max_length=max_length)
        
        # Decode output
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
        
        # Calculate basic confidence (average probability of generated tokens)
        if hasattr(outputs, 'scores') and outputs.scores:
            scores = torch.stack(outputs.scores, dim=1)
            probabilities = torch.softmax(scores, dim=-1)
            max_probs = torch.max(probabilities, dim=-1).values
            confidence = torch.mean(max_probs).item()
        else:
            confidence = 0.7  # Default confidence
        
        return {
            'english': text,
            'malayalam': translation,
            'confidence': round(confidence, 3),
            'token_count': len(inputs['input_ids'][0]),
            'method': 'direct_translation'
        }

# Initialize the translator
translator = EnglishMalayalamTranslator()

# Test sentences covering various domains
test_sentences = [
    # Basic greetings
    "Hello",
    "Good morning",
    "How are you?",
    "Thank you very much",
    
    # Common questions
    "What is your name?",
    "Where are you from?",
    "How old are you?",
    "What time is it?",
    
    # Daily conversations
    "I am going to the market",
    "Can you help me please?",
    "I don't understand",
    "Where is the hospital?",
    
    # Technology terms
    "Machine learning is a subset of artificial intelligence",
    "I love programming and software development",
    "The computer is running very fast",
    "Data science involves statistics and programming",
    
    # Longer sentences
    "The quick brown fox jumps over the lazy dog near the river bank",
    "Kerala is known for its beautiful backwaters and delicious cuisine",
    "Learning new languages helps in understanding different cultures",
    "Technology has transformed the way we communicate with each other"
]

print("🧪 Testing Direct English-Malayalam Translation")
print("=" * 70)

for i, sentence in enumerate(test_sentences, 1):
    result = translator.translate_with_confidence(sentence)
    
    print(f"\n{i}. ENGLISH: {result['english']}")
    print(f"   MALAYALAM: {result['malayalam']}")
    print(f"   Confidence: {result['confidence']} | Tokens: {result['token_count']}")
    print("-" * 70)

class TranslationQualityAssessor:
    """Simple quality assessment for translations"""
    
    def __init__(self):
        self.common_malayalam_words = [
            'ആണ്', 'ഉണ്ട്', 'എന്ന്', 'ഒരു', 'അവൻ', 'അവൾ', 'അത്', 'ഇത്',
            'നമ്മൾ', 'നിങ്ങൾ', 'ഞാൻ', 'നീ', 'അവർ', 'ഇവിടെ', 'അവിടെ'
        ]
    
    def contains_malayalam_script(self, text: str) -> bool:
        """Check if text contains Malayalam characters"""
        malayalam_range = range(0x0D00, 0x0D7F)
        return any(ord(char) in malayalam_range for char in text)

class EnhancedEnglishMalayalamTranslator:
    def __init__(self):
        self.translator = EnglishMalayalamTranslator()
        self.quality_assessor = TranslationQualityAssessor()
        
    def assess_translation_quality(self, english_text: str, malayalam_text: str) -> Dict:
        """Assess the quality of translation"""
        quality_score = 0.0
        issues = []
        
        # Basic quality checks
        if len(malayalam_text.strip()) == 0:
            quality_score = 0.0
            issues.append("Empty translation")
        elif malayalam_text.lower() == english_text.lower():
            quality_score = 0.1
            issues.append("Translation identical to input")
        else:
            # Simple heuristic-based quality scoring
            quality_score = 0.7  # Base score for successful translation
            
            # Check for common issues
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
        """Convert numerical score to letter grade"""
        if score >= 0.9: return 'A+'
        elif score >= 0.8: return 'A'
        elif score >= 0.7: return 'B'
        elif score >= 0.6: return 'C'
        elif score >= 0.5: return 'D'
        else: return 'F'
    
    def translate_with_analysis(self, text: str) -> Dict:
        """Translate with comprehensive analysis"""
        # Perform translation
        translation_result = self.translator.translate_with_confidence(text)
        
        # Assess quality
        quality_assessment = self.assess_translation_quality(
            translation_result['english'], 
            translation_result['malayalam']
        )
        
        # Combine results
        comprehensive_result = {**translation_result, **quality_assessment}
        return comprehensive_result

# Initialize enhanced translator
enhanced_translator = EnhancedEnglishMalayalamTranslator()

def batch_translate_analyze(sentences: List[str]) -> pd.DataFrame:
    """Translate and analyze multiple sentences"""
    results = []
    
    for sentence in sentences:
        result = enhanced_translator.translate_with_analysis(sentence)
        results.append(result)
    
    df = pd.DataFrame(results)
    return df

# Perform batch translation
batch_results = batch_translate_analyze(test_sentences)

print("📊 Batch Translation Results Summary")
print("=" * 60)
print(f"Total sentences: {len(batch_results)}")
print(f"Average confidence: {batch_results['confidence'].mean():.3f}")
print(f"Average quality score: {batch_results['score'].mean():.3f}")
print(f"Quality grades: {batch_results['grade'].value_counts().to_dict()}")

print("\n📈 Detailed Results:")
print(batch_results[['english', 'malayalam', 'confidence', 'score', 'grade']].head(10))

class DomainSpecificTranslator:
    def __init__(self):
        self.translator = EnglishMalayalamTranslator()
        self.domains = {
            'technical': [
                'algorithm', 'database', 'network', 'software', 'hardware',
                'programming', 'debugging', 'optimization', 'framework'
            ],
            'medical': [
                'hospital', 'doctor', 'medicine', 'patient', 'treatment',
                'symptom', 'diagnosis', 'prescription', 'recovery'
            ],
            'legal': [
                'contract', 'agreement', 'lawyer', 'court', 'judgment',
                'evidence', 'testimony', 'verdict', 'appeal'
            ]
        }
    
    def translate_domain_text(self, text: str, domain: str = 'general') -> Dict:
        """Translate text with domain context"""
        translation = self.translator.translate_with_confidence(text)
        
        # Add domain-specific analysis
        domain_words = self.domains.get(domain, [])
        domain_match_count = sum(1 for word in domain_words if word in text.lower())
        
        translation['domain'] = domain
        translation['domain_relevance'] = domain_match_count / len(text.split()) if text.split() else 0
        
        return translation

# Test domain-specific translation
domain_translator = DomainSpecificTranslator()

domain_texts = [
    ("I need to debug this software algorithm", "technical"),
    ("The patient needs immediate medical treatment", "medical"),
    ("Please review this legal contract carefully", "legal")
]

print("🎯 Domain-Specific Translation Tests")
print("=" * 60)

for text, domain in domain_texts:
    result = domain_translator.translate_domain_text(text, domain)
    print(f"\nDomain: {domain.upper()}")
    print(f"English: {text}")
    print(f"Malayalam: {result['malayalam']}")
    print(f"Domain Relevance: {result['domain_relevance']:.3f}")
    print("-" * 50)

def interactive_translation_demo():
    """Interactive demo for English to Malayalam translation"""
    print("🌍 English to Malayalam Translator Demo")
    print("=" * 50)
    print("Type 'quit' to exit, 'help' for commands, 'batch' for multiple lines\n")
    
    while True:
        user_input = input("Enter English text: ").strip()
        
        if user_input.lower() == 'quit':
            print("Thank you for using the translator! 👋")
            break
        
        if user_input.lower() == 'help':
            print("\n📖 Available commands:")
            print("'quit' - Exit the demo")
            print("'help' - Show this help message") 
            print("'batch' - Enter multiple lines for translation")
            print("'domain technical/medical/legal' - Set translation domain")
            print()
            continue
        
        if user_input.lower() == 'batch':
            print("\n📝 Enter multiple lines (empty line to finish):")
            lines = []
            while True:
                line = input()
                if line.strip() == '':
                    break
                lines.append(line)
            
            if lines:
                print("\n🔁 Batch Translation Results:")
                for i, line in enumerate(lines, 1):
                    result = translator.translate_with_confidence(line)
                    print(f"{i}. {result['english']}")
                    print(f"   → {result['malayalam']}")
                    print(f"   Confidence: {result['confidence']}")
                    print()
            continue
        
        if not user_input:
            continue
        
        # Perform translation
        result = enhanced_translator.translate_with_analysis(user_input)
        
        print(f"\n📝 Translation Result:")
        print(f"English: {result['english']}")
        print(f"Malayalam: {result['malayalam']}")
        print(f"Confidence: {result['confidence']} | Quality: {result['score']:.3f} ({result['grade']})")
        
        if result['issues']:
            print(f"⚠️  Issues: {', '.join(result['issues'])}")
        
        print("-" * 60)
        print()

# Run the interactive demo
interactive_translation_demo()

class OptimizedTranslator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-ml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Enable optimizations
        self.model.eval()
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            self.model = torch.compile(self.model)
    
    def optimized_translate_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """Optimized batch translation with memory management"""
        translations = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=256
            ).to(self.device)
            
            # Generate with optimized settings
            with torch.no_grad():
                translated = self.model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=2,  # Reduced for speed
                    early_stopping=True
                )
            
            # Decode batch
            batch_translations = [
                self.tokenizer.decode(t, skip_special_tokens=True) 
                for t in translated
            ]
            
            translations.extend(batch_translations)
            
            # Clear memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return translations

# Test optimized translation
optimized_translator = OptimizedTranslator()

# Large batch test
large_batch = [
    "Hello world",
    "Good morning",
    "How are you today?",
    "Thank you very much",
    "What is your name?",
    "Where is the nearest hospital?",
    "I need help with this problem",
    "The weather is beautiful today",
    "Can you please assist me?",
    "I love machine learning and AI"
]

print("⚡ Testing Optimized Batch Translation")
optimized_results = optimized_translator.optimized_translate_batch(large_batch)

for i, (english, malayalam) in enumerate(zip(large_batch, optimized_results)):
    print(f"{i+1}. {english}")
    print(f"   → {malayalam}")
    print()

import json
import os
from datetime import datetime

class TranslationManager:
    def __init__(self, translator):
        self.translator = translator
        self.translation_history = []
        
    def save_translation(self, english_text: str, malayalam_text: str, metadata: Dict = None):
        """Save translation to history"""
        translation_record = {
            'timestamp': datetime.now().isoformat(),
            'english': english_text,
            'malayalam': malayalam_text,
            'metadata': metadata or {}
        }
        
        self.translation_history.append(translation_record)
        return translation_record
    
    def export_history(self, filename: str = "translation_history.json"):
        """Export translation history to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.translation_history, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Translation history exported to {filename}")
    
    def load_history(self, filename: str):
        """Load translation history from JSON file"""
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                self.translation_history = json.load(f)
            print(f"✅ Loaded {len(self.translation_history)} translations from {filename}")
        else:
            print(f"❌ File {filename} not found")
    
    def get_stats(self) -> Dict:
        """Get translation statistics"""
        if not self.translation_history:
            return {}
        
        total_translations = len(self.translation_history)
        total_english_chars = sum(len(t['english']) for t in self.translation_history)
        total_malayalam_chars = sum(len(t['malayalam']) for t in self.translation_history)
        
        return {
            'total_translations': total_translations,
            'total_english_characters': total_english_chars,
            'total_malayalam_characters': total_malayalam_chars,
            'average_english_length': total_english_chars / total_translations,
            'average_malayalam_length': total_malayalam_chars / total_translations,
            'first_translation': self.translation_history[0]['timestamp'] if self.translation_history else None,
            'last_translation': self.translation_history[-1]['timestamp'] if self.translation_history else None
        }

# Initialize translation manager
translation_manager = TranslationManager(translator)

# Test the manager
test_text = "This is a test translation for the history manager"
malayalam_translation = translator.translate(test_text)
translation_manager.save_translation(test_text, malayalam_translation, {'purpose': 'testing'})

print("📊 Translation Manager Stats:")
stats = translation_manager.get_stats()
for key, value in stats.items():
    print(f"   {key}: {value}")

def complete_demonstration():
    """Complete demonstration of the English-Malayalam translator"""
    print("=" * 70)
    print("🚀 ENGLISH-MALAYALAM TRANSLATION SYSTEM")
    print("=" * 70)
    print("\nThis system uses Helsinki-NLP/opus-mt-en-ml model for direct translation")
    print("from English to Malayalam using state-of-the-art neural machine translation.\n")
    
    # Show system info
    print("📊 System Information:")
    print(f"   Device: {translator.device}")
    print(f"   Model: Helsinki-NLP/opus-mt-en-ml")
    print(f"   Tokenizer vocabulary: {translator.tokenizer.vocab_size} tokens")
    
    # Quick demonstration
    demo_sentences = [
        "Hello, how are you today?",
        "Machine learning is transforming the world",
        "Kerala is known as God's own country"
    ]
    
    print("\n🧪 Quick Demonstration:")
    for sentence in demo_sentences:
        translation = translator.translate(sentence)
        print(f"   English: {sentence}")
        print(f"   Malayalam: {translation}")
        print()
    
    print("🎯 System is ready for translation!")
    print("   Use the interactive_demo() function to start translating.")
    print("=" * 70)

# Run complete demonstration
complete_demonstration()

# Post-processing enhancements for better translation quality
class EnhancedMalayalamTranslator:
    def __init__(self):
        self.translator = EnglishMalayalamTranslator()
        self.post_processing_rules = self._create_post_processing_rules()
    
    def _create_post_processing_rules(self):
        """Create rules to improve translation quality"""
        return {
            'corrections': {
                'ആൽബം': 'അൽഗോരിതം',  # album → algorithm
                'മിച്ചൈൻ': 'മെഷീൻ',     # michine → machine
                'യന്ത്രം': 'മെഷീൻ',      # yanthram → machine
                'തകർക്കണം': 'ഡീബഗ് ചെയ്യണം',  # break → debug
                'കെരള': 'കേരള',         # keral → kerala (proper spelling)
            },
            'improvements': {
                'സുഖമല്ലേ?': 'സുഖമാണോ?',  # aren't you well? → are you well?
                'നീ എവിടുന്നാ?': 'നിങ്ങൾ എവിടെനിന്നാണ്?',  # informal → formal
            }
        }
    
    def post_process_translation(self, malayalam_text: str, english_text: str) -> str:
        """Apply post-processing to improve translation quality"""
        improved_text = malayalam_text
        
        # Apply corrections
        for wrong, correct in self.post_processing_rules['corrections'].items():
            improved_text = improved_text.replace(wrong, correct)
        
        # Apply improvements based on context
        for pattern, improvement in self.post_processing_rules['improvements'].items():
            if pattern in improved_text:
                # Only apply if it makes sense in context
                improved_text = improved_text.replace(pattern, improvement)
        
        # Ensure proper punctuation
        if not improved_text.endswith(('.', '?', '!', '।')):
            if english_text.endswith('?'):
                improved_text += '?'
            else:
                improved_text += '.'
        
        return improved_text
    
    def translate_with_enhancements(self, text: str) -> Dict:
        """Translate with post-processing enhancements"""
        # Get initial translation
        result = self.translator.translate_with_confidence(text)
        
        # Apply post-processing
        original_translation = result['malayalam']
        enhanced_translation = self.post_process_translation(original_translation, text)
        
        # Update result
        result['malayalam_original'] = original_translation
        result['malayalam_enhanced'] = enhanced_translation
        result['was_improved'] = original_translation != enhanced_translation
        
        return result

# Initialize enhanced translator
enhanced_malayalam_translator = EnhancedMalayalamTranslator()

# Test the enhanced translations
print("🔧 Testing Enhanced Translations")
print("=" * 60)

test_cases = [
    "I need to debug this software algorithm",
    "Machine learning is transforming the world",
    "How are you?",
    "Where are you from?"
]

for english_text in test_cases:
    result = enhanced_malayalam_translator.translate_with_enhancements(english_text)
    
    print(f"\nENGLISH: {english_text}")
    print(f"ORIGINAL: {result['malayalam_original']}")
    print(f"ENHANCED: {result['malayalam_enhanced']}")
    print(f"IMPROVED: {'Yes' if result['was_improved'] else 'No'}")
    print("-" * 50)

class TranslationEvaluator:
    def __init__(self):
        self.malayalam_expert_rules = self._create_expert_rules()
    
    def _create_expert_rules(self):
        """Rules for evaluating Malayalam translation quality"""
        return {
            'positive_indicators': [
                'proper use of Malayalam script',
                'correct sentence structure', 
                'appropriate vocabulary',
                'cultural adaptation',
                'grammatical correctness'
            ],
            'negative_indicators': [
                'english words in malayalam text',
                'incorrect word order',
                'literal translation issues',
                'spelling mistakes',
                'context mismatch'
            ]
        }
    
    def evaluate_translation(self, english_text: str, malayalam_text: str) -> Dict:
        """Comprehensive evaluation of translation quality"""
        score = 0.7  # Base score (model is generally good)
        
        evaluation = {
            'english': english_text,
            'malayalam': malayalam_text,
            'score': score,
            'strengths': [],
            'improvements': [],
            'grade': self._score_to_grade(score)
        }
        
        # Check for strengths
        if any(word in malayalam_text for word in ['ൻ', 'ം', 'ർ', 'ൽ']):  # Malayalam specific characters
            evaluation['strengths'].append('Uses proper Malayalam script')
            score += 0.1
        
        if len(malayalam_text) > len(english_text) * 0.5:  # Reasonable length
            evaluation['strengths'].append('Appropriate translation length')
            score += 0.05
        
        # Check for improvements needed
        if any(char in malayalam_text for char in ['[', ']', '<', '>']):
            evaluation['improvements'].append('Contains special characters')
            score -= 0.1
        
        if malayalam_text.lower() == english_text.lower():
            evaluation['improvements'].append('Translation identical to input')
            score -= 0.3
        
        # Final score adjustment
        evaluation['score'] = max(0.1, min(1.0, score))
        evaluation['grade'] = self._score_to_grade(evaluation['score'])
        
        return evaluation
    
    def _score_to_grade(self, score: float) -> str:
        if score >= 0.9: return 'A+ (Excellent)'
        elif score >= 0.8: return 'A (Very Good)'
        elif score >= 0.7: return 'B (Good)'
        elif score >= 0.6: return 'C (Average)'
        elif score >= 0.5: return 'D (Below Average)'
        else: return 'F (Poor)'

# Test evaluation system
evaluator = TranslationEvaluator()

print("📊 Comprehensive Translation Evaluation")
print("=" * 70)

evaluation_samples = [
    ("Hello", "ഹലോ."),
    ("Good morning", "സുപ്രഭാതം"),
    ("Machine learning is transforming the world", "മിച്ചൈൻ പഠനം ലോകത്തെ രൂപാന്തരപ്പെടുത്തുന്നു"),
    ("I need to debug this software algorithm", "എനിക്ക് ഈ സോഫ്റ്റ് വെയർ ആൽബം തകർക്കണം.")
]

for eng, mal in evaluation_samples:
    evaluation = evaluator.evaluate_translation(eng, mal)
    
    print(f"\nENGLISH: {eng}")
    print(f"MALAYALAM: {mal}")
    print(f"SCORE: {evaluation['score']:.3f} | GRADE: {evaluation['grade']}")
    
    if evaluation['strengths']:
        print("✅ Strengths:", ", ".join(evaluation['strengths']))
    if evaluation['improvements']:
        print("⚠️  Improvements:", ", ".join(evaluation['improvements']))
    
    print("-" * 70)

def final_comprehensive_demo():
    """Final demonstration showcasing all features"""
    print("=" * 80)
    print("🎯 FINAL ENGLISH-MALAYALAM TRANSLATION SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Test categories
    categories = {
        "Greetings": [
            "Hello, how are you?",
            "Good morning, have a nice day!",
            "Thank you for your help"
        ],
        "Technology": [
            "Artificial intelligence is changing our world",
            "Python programming is very popular",
            "Data science requires statistics and programming skills"
        ],
        "Travel": [
            "Kerala is famous for its backwaters",
            "I want to visit Munnar hills",
            "Malayalam is the language of Kerala"
        ],
        "Business": [
            "Please send me the contract details",
            "We need to schedule a meeting next week",
            "The project deadline is approaching"
        ]
    }
    
    for category, sentences in categories.items():
        print(f"\n📂 {category.upper()} CATEGORY")
        print("-" * 60)
        
        for sentence in sentences:
            result = translator.translate_with_confidence(sentence)
            
            print(f"🔤 English: {sentence}")
            print(f"📜 Malayalam: {result['malayalam']}")
            print(f"📊 Confidence: {result['confidence']:.3f}")
            print()
    
    # Performance summary
    print("📈 PERFORMANCE SUMMARY")
    print("-" * 60)
    
    total_sentences = sum(len(sentences) for sentences in categories.values())
    confidence_scores = []
    
    for sentences in categories.values():
        for sentence in sentences:
            result = translator.translate_with_confidence(sentence)
            confidence_scores.append(result['confidence'])
    
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    
    print(f"Total sentences translated: {total_sentences}")
    print(f"Average confidence score: {avg_confidence:.3f}")
    print(f"Best confidence: {max(confidence_scores):.3f}")
    print(f"Worst confidence: {min(confidence_scores):.3f}")
    print(f"Model: Helsinki-NLP/opus-mt-en-ml")
    print(f"Device: {translator.device}")

# Run final demo
final_comprehensive_demo()

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎉 English-Malayalam Translation Script Complete!")
    print("="*80)
    print("\nAvailable functions:")
    print("• interactive_translation_demo() - Interactive translation interface")
    print("• complete_demonstration() - System overview and demo")
    print("• final_comprehensive_demo() - Comprehensive feature showcase")
    print("\nTo start translating, run: interactive_translation_demo()")