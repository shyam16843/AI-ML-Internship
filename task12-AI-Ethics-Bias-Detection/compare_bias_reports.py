import pandas as pd

def compare_reports(original_file='bias_report.csv', mitigated_file='bias_report_mitigated.csv'):
    df_orig = pd.read_csv(original_file)
    df_mitig = pd.read_csv(mitigated_file)
    
    # Compare sentiment averages
    orig_sentiment_mean = df_orig['sentiment'].mean()
    mitig_sentiment_mean = df_mitig['sentiment'].mean()
    sentiment_diff = mitig_sentiment_mean - orig_sentiment_mean
    
    # Compare entity counts (number of distinct person entities)
    orig_entities_count = df_orig['person_entities'].apply(lambda x: len(eval(x))).sum()
    mitig_entities_count = df_mitig['person_entities'].apply(lambda x: len(eval(x))).sum()
    entities_diff = mitig_entities_count - orig_entities_count
    
    print(f"Average sentiment (original): {orig_sentiment_mean:.3f}")
    print(f"Average sentiment (mitigated): {mitig_sentiment_mean:.3f}")
    print(f"Difference in sentiment: {sentiment_diff:.3f}\n")
    
    print(f"Total person entities (original): {orig_entities_count}")
    print(f"Total person entities (mitigated): {mitig_entities_count}")
    print(f"Difference in person entities: {entities_diff}")

if __name__ == "__main__":
    compare_reports()
