import spacy
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

class BiasDetector:
    def __init__(self):
        """Initialize the bias detector with spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✓ Bias detector initialized successfully!")
        except OSError:
            print("Error: English model not found. Please download it using:")
            print("python -m spacy download en_core_web_sm")
            raise
    
    def detect_gender_bias(self, text):
        """Detect potential gender bias in text"""
        doc = self.nlp(text)
        
        # Gender-related analysis
        male_pronouns = ['he', 'him', 'his', 'himself']
        female_pronouns = ['she', 'her', 'hers', 'herself']
        neutral_pronouns = ['they', 'them', 'their', 'themself']
        
        male_count = sum(1 for token in doc if token.text.lower() in male_pronouns)
        female_count = sum(1 for token in doc if token.text.lower() in female_pronouns)
        neutral_count = sum(1 for token in doc if token.text.lower() in neutral_pronouns)
        
        # Occupation and role analysis
        occupations = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG"]:
                for token in ent.root.subtree:
                    if token.dep_ in ["attr", "nsubj", "dobj"] and token.pos_ == "NOUN":
                        occupations.append(token.text)
        
        return {
            'male_pronouns': male_count,
            'female_pronouns': female_count,
            'neutral_pronouns': neutral_count,
            'gender_balance_score': female_count / (male_count + female_count + 1),
            'potential_occupations': list(set(occupations)),
            'total_entities': len(doc.ents),
            'sentence_length': len(text.split())
        }
    
    def analyze_corpus(self, texts):
        """Analyze multiple texts for bias patterns"""
        results = []
        for i, text in enumerate(texts):
            analysis = self.detect_gender_bias(text)
            analysis['text_id'] = i + 1
            analysis['text_preview'] = text[:80] + "..." if len(text) > 80 else text
            results.append(analysis)
        
        return pd.DataFrame(results)
    
    def generate_report(self, texts):
        """Generate comprehensive bias analysis report"""
        df = self.analyze_corpus(texts)
        
        print("=" * 70)
        print("AI ETHICS BIAS DETECTION REPORT")
        print("=" * 70)
        
        total_male = df['male_pronouns'].sum()
        total_female = df['female_pronouns'].sum()
        total_neutral = df['neutral_pronouns'].sum()
        total_pronouns = total_male + total_female + total_neutral
        
        print(f"\nPRONOUN DISTRIBUTION ANALYSIS:")
        print(f"Male pronouns (he/him/his): {total_male}")
        print(f"Female pronouns (she/her/hers): {total_female}")
        print(f"Neutral pronouns (they/them/their): {total_neutral}")
        print(f"Total pronouns: {total_pronouns}")
        
        if total_pronouns > 0:
            print(f"\nGENDER RATIOS:")
            print(f"Male: {total_male/total_pronouns:.1%}")
            print(f"Female: {total_female/total_pronouns:.1%}")
            print(f"Neutral: {total_neutral/total_pronouns:.1%}")
        
        # Bias detection logic
        if total_male > total_female * 2 and total_male > 5:
            print("\n⚠️  POTENTIAL MALE GENDER BIAS DETECTED")
            print("   Male pronouns significantly outnumber female pronouns")
        elif total_female > total_male * 2 and total_female > 5:
            print("\n⚠️  POTENTIAL FEMALE GENDER BIAS DETECTED")
            print("   Female pronouns significantly outnumber male pronouns")
        else:
            print("\n✓ Gender representation appears relatively balanced")
        
        # Show specific examples
        biased_texts = df[(df['male_pronouns'] > 2) | (df['female_pronouns'] > 2)]
        if len(biased_texts) > 0:
            print(f"\nTEXTS WITH PRONOUN IMBALANCES:")
            for _, row in biased_texts.iterrows():
                bias_type = "male" if row['male_pronouns'] > row['female_pronouns'] else "female"
                print(f"Text {row['text_id']}: {bias_type} bias ({row['male_pronouns']}M/{row['female_pronouns']}F)")
                print(f"   Preview: {row['text_preview']}")
        
        return df

def main():
    """Main function to demonstrate the bias detection system"""
    print("Initializing AI Ethics Bias Detection System...")
    detector = BiasDetector()
    
    # Sample dataset for analysis (replace with your own texts)
    sample_texts = [
        "The doctor performed the surgery successfully. He was very skilled and experienced in his field.",
        "Nurses should always check their patients carefully. She needs to be attentive to all details.",
        "Engineers must verify their calculations thoroughly. They should ensure everything is precise.",
        "The CEO announced her decision to expand the company. She will lead the organization to new markets.",
        "Programmers often work long hours to meet deadlines. He needs to manage his time effectively.",
        "Teachers educate our children and shape their future. She plays a crucial role in society.",
        "The construction worker finished his shift exhausted. He had been working hard all day.",
        "Scientists conduct important research that benefits humanity. They make valuable contributions.",
        "The manager told his team to complete the project. He provided clear instructions.",
        "The designer created her portfolio with care. She included her best work.",
    ]
    
    print(f"\nAnalyzing {len(sample_texts)} text samples...")
    
    # Generate comprehensive report
    report_df = detector.generate_report(sample_texts)
    
    # Save detailed results
    report_df.to_csv('bias_analysis_detailed.csv', index=False)
    print(f"\n✓ Detailed results saved to 'bias_analysis_detailed.csv'")
    
    # Create summary statistics
    summary = {
        'total_texts': len(sample_texts),
        'total_male_pronouns': report_df['male_pronouns'].sum(),
        'total_female_pronouns': report_df['female_pronouns'].sum(),
        'total_neutral_pronouns': report_df['neutral_pronouns'].sum(),
        'average_sentence_length': report_df['sentence_length'].mean()
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('bias_analysis_summary.csv', index=False)
    print(f"✓ Summary statistics saved to 'bias_analysis_summary.csv'")
    
    # Simple visualization
    try:
        plt.figure(figsize=(12, 8))
        
        # Pronoun distribution
        plt.subplot(2, 2, 1)
        pronouns = ['Male', 'Female', 'Neutral']
        counts = [summary['total_male_pronouns'], summary['total_female_pronouns'], summary['total_neutral_pronouns']]
        colors = ['lightblue', 'lightpink', 'lightgreen']
        plt.bar(pronouns, counts, color=colors)
        plt.title('Pronoun Distribution')
        plt.ylabel('Count')
        
        # Text length distribution
        plt.subplot(2, 2, 2)
        plt.hist(report_df['sentence_length'], bins=10, alpha=0.7, color='skyblue')
        plt.title('Sentence Length Distribution')
        plt.xlabel('Words per sentence')
        plt.ylabel('Frequency')
        
        # Gender balance by text
        plt.subplot(2, 2, 3)
        plt.scatter(report_df['male_pronouns'], report_df['female_pronouns'], alpha=0.6)
        plt.plot([0, max(report_df['male_pronouns'].max(), report_df['female_pronouns'].max())],
         [0, max(report_df['male_pronouns'].max(), report_df['female_pronouns'].max())],
         'r--', alpha=0.5) # Ideal balance line
        plt.title('Male vs Female Pronouns per Text')
        plt.xlabel('Male Pronouns')
        plt.ylabel('Female Pronouns')

        plt.subplot(2, 2, 4)
        labels = ['Male', 'Female', 'Neutral']
        sizes = [summary['total_male_pronouns'], summary['total_female_pronouns'], summary['total_neutral_pronouns']]
        colors = ['lightblue', 'lightpink', 'lightgreen']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.title('Pronoun Proportions')

        plt.suptitle('Bias Analysis Summary', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('bias_analysis_visualization.png', dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved as 'bias_analysis_visualization.png'")
        
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the generated CSV files for detailed analysis")
    print("2. Examine the visualization for patterns")
    print("3. Replace sample texts with your actual dataset")
    print("4. Extend the analysis with additional bias detection features")

if __name__ == "__main__":
    main()