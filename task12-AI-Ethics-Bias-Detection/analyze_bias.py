import pandas as pd
import spacy
from textblob import TextBlob
import sys

# Load small English model for linguistic features
nlp = spacy.load("en_core_web_sm")

def analyze_text(text):
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    sentiment = TextBlob(text).sentiment.polarity
    return {'person_entities': persons, 'sentiment': sentiment}

def analyze_bias(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    analysis_results = df['response'].apply(analyze_text)
    df_analysis = pd.DataFrame(analysis_results.tolist())
    df = pd.concat([df, df_analysis], axis=1)
    df.to_csv(output_csv, index=False)
    print(f"Bias report for {input_csv} saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python analyze_bias.py <input_csv> <output_csv>")
    else:
        analyze_bias(sys.argv[1], sys.argv[2])
