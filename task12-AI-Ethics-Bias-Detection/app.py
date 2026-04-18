import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Ethics Bias Detection",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #6f42c1; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { text-align: center; color: #6c757d; margin-bottom: 2rem; }
    .bias-alert { background-color: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107; }
    .bias-ok { background-color: #d4edda; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚖️ AI Ethics Bias Detection Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Detect gender, occupation & representation bias in LLM outputs using NLP analysis</div>', unsafe_allow_html=True)

def detect_bias(text):
    tokens = re.findall(r'\b\w+\b', text.lower())

    male_pronouns = ['he', 'him', 'his', 'himself']
    female_pronouns = ['she', 'her', 'hers', 'herself']
    neutral_pronouns = ['they', 'them', 'their', 'themself']

    male_count = sum(1 for t in tokens if t in male_pronouns)
    female_count = sum(1 for t in tokens if t in female_pronouns)
    neutral_count = sum(1 for t in tokens if t in neutral_pronouns)

    total = male_count + female_count + neutral_count
    balance_score = female_count / (male_count + female_count + 1)

    # Estimate named entities by counting capitalized words (excluding first word of sentence)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    entity_estimate = 0
    for sent in sentences:
        words = sent.split()
        entity_estimate += sum(1 for w in words[1:] if w and w[0].isupper())

    if male_count > female_count * 2 and male_count > 2:
        bias_type = "⚠️ Male Bias"
    elif female_count > male_count * 2 and female_count > 2:
        bias_type = "⚠️ Female Bias"
    else:
        bias_type = "✅ Balanced"

    return {
        'male_pronouns': male_count,
        'female_pronouns': female_count,
        'neutral_pronouns': neutral_count,
        'total_pronouns': total,
        'balance_score': round(balance_score, 3),
        'bias_verdict': bias_type,
        'entities': entity_estimate,
        'word_count': len(text.split())
    }

def analyze_corpus(texts):
    results = []
    for i, text in enumerate(texts):
        r = detect_bias(text)
        r['text_id'] = i + 1
        r['preview'] = text[:80] + "..." if len(text) > 80 else text
        results.append(r)
    return pd.DataFrame(results)

# Sidebar
st.sidebar.title("⚙️ Options")
mode = st.sidebar.radio("Mode", ["Single Text Analysis", "Batch Analysis", "Demo Dataset"])

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown("This pipeline detects gender and representation bias in AI-generated text using NLP analysis.")

DEMO_TEXTS = [
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

if mode == "Single Text Analysis":
    st.header("Analyze a Single Text")
    text_input = st.text_area(
        "Enter text to analyze for bias:",
        height=150,
        placeholder="Paste any AI-generated text here to check for gender bias...",
        value="The engineer solved the problem quickly. He was proud of his work and shared it with his team."
    )

    if st.button("🔍 Analyze for Bias", type="primary"):
        if text_input.strip():
            result = detect_bias(text_input)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Male Pronouns", result['male_pronouns'])
            with col2:
                st.metric("Female Pronouns", result['female_pronouns'])
            with col3:
                st.metric("Neutral Pronouns", result['neutral_pronouns'])
            with col4:
                st.metric("Balance Score", result['balance_score'])

            st.markdown("---")
            verdict = result['bias_verdict']
            if "⚠️" in verdict:
                st.markdown(f'<div class="bias-alert"><strong>Bias Verdict: {verdict}</strong><br>This text shows signs of gender imbalance in pronoun usage.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bias-ok"><strong>Bias Verdict: {verdict}</strong><br>Pronoun usage appears relatively balanced.</div>', unsafe_allow_html=True)

            if result['total_pronouns'] > 0:
                fig, ax = plt.subplots(figsize=(5, 3))
                categories = ['Male', 'Female', 'Neutral']
                values = [result['male_pronouns'], result['female_pronouns'], result['neutral_pronouns']]
                colors = ['#4e79a7', '#f28e2b', '#76b7b2']
                ax.bar(categories, values, color=colors)
                ax.set_title('Pronoun Distribution')
                ax.set_ylabel('Count')
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No pronouns detected in this text.")

elif mode == "Batch Analysis":
    st.header("Batch Text Analysis")
    st.markdown("Enter multiple texts, one per line.")
    batch_input = st.text_area(
        "Texts to analyze (one per line):",
        height=250,
        placeholder="The engineer fixed the bug. He was pleased with the result.\nThe nurse cared for her patients gently.\nThe scientist published their findings."
    )

    if st.button("🔍 Analyze All", type="primary"):
        texts = [t.strip() for t in batch_input.split('\n') if t.strip()]
        if texts:
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                df = analyze_corpus(texts)

            st.subheader("Results Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Texts", len(df))
            with col2:
                st.metric("Total Male Pronouns", df['male_pronouns'].sum())
            with col3:
                st.metric("Total Female Pronouns", df['female_pronouns'].sum())
            with col4:
                biased = df[df['bias_verdict'] != '✅ Balanced'].shape[0]
                st.metric("Biased Texts", biased)

            st.subheader("Per-Text Results")
            display_df = df[['text_id', 'preview', 'male_pronouns', 'female_pronouns', 'neutral_pronouns', 'bias_verdict']]
            display_df.columns = ['#', 'Text Preview', 'Male', 'Female', 'Neutral', 'Verdict']
            st.dataframe(display_df, use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Report (CSV)", csv, "bias_analysis.csv", "text/csv")

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 4))
                total_m = df['male_pronouns'].sum()
                total_f = df['female_pronouns'].sum()
                total_n = df['neutral_pronouns'].sum()
                ax.pie([total_m, total_f, total_n],
                       labels=['Male', 'Female', 'Neutral'],
                       colors=['#4e79a7', '#f28e2b', '#76b7b2'],
                       autopct='%1.1f%%')
                ax.set_title('Overall Pronoun Distribution')
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(df['male_pronouns'], df['female_pronouns'], alpha=0.7, color='#6f42c1', s=100)
                max_val = max(df['male_pronouns'].max(), df['female_pronouns'].max()) + 1
                ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect balance')
                ax.set_xlabel('Male Pronouns')
                ax.set_ylabel('Female Pronouns')
                ax.set_title('Male vs Female Pronouns per Text')
                ax.legend()
                st.pyplot(fig)
                plt.close()
        else:
            st.warning("Please enter at least one text.")

elif mode == "Demo Dataset":
    st.header("Demo: Occupational Gender Bias")
    st.markdown("This demo analyzes 10 sentences about professionals to detect gender stereotyping in role assignments.")

    st.subheader("Sample Texts")
    for i, t in enumerate(DEMO_TEXTS):
        st.markdown(f"**{i+1}.** {t}")

    if st.button("🔍 Run Demo Analysis", type="primary"):
        with st.spinner("Analyzing demo dataset..."):
            df = analyze_corpus(DEMO_TEXTS)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Male Pronouns", df['male_pronouns'].sum())
        with col2:
            st.metric("Total Female Pronouns", df['female_pronouns'].sum())
        with col3:
            st.metric("Neutral Pronouns", df['neutral_pronouns'].sum())
        with col4:
            biased = df[df['bias_verdict'] != '✅ Balanced'].shape[0]
            st.metric("Biased Texts", f"{biased}/10")

        total_m = df['male_pronouns'].sum()
        total_f = df['female_pronouns'].sum()
        if total_m > total_f * 1.5:
            st.markdown('<div class="bias-alert"><strong>⚠️ Overall Male Bias Detected</strong><br>Male pronouns dominate across this dataset — a common pattern in AI-generated professional text.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="bias-ok"><strong>✅ Relatively Balanced Dataset</strong></div>', unsafe_allow_html=True)

        st.subheader("Per-Text Breakdown")
        display_df = df[['text_id', 'preview', 'male_pronouns', 'female_pronouns', 'bias_verdict']]
        display_df.columns = ['#', 'Text Preview', 'Male', 'Female', 'Verdict']
        st.dataframe(display_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie([total_m, total_f, df['neutral_pronouns'].sum()],
                   labels=['Male', 'Female', 'Neutral'],
                   colors=['#4e79a7', '#f28e2b', '#76b7b2'],
                   autopct='%1.1f%%', startangle=140)
            ax.set_title('Pronoun Distribution Across Dataset')
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            df_plot = df[['text_id', 'male_pronouns', 'female_pronouns']].set_index('text_id')
            df_plot.plot(kind='bar', ax=ax, color=['#4e79a7', '#f28e2b'])
            ax.set_title('Male vs Female Pronouns per Text')
            ax.set_xlabel('Text ID')
            ax.set_ylabel('Count')
            ax.legend(['Male', 'Female'])
            plt.xticks(rotation=0)
            st.pyplot(fig)
            plt.close()

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Report", csv, "bias_demo_report.csv", "text/csv")

st.markdown("---")
st.markdown("**Built by Ghanashyam T V** | [GitHub](https://github.com/shyam16843) | [LinkedIn](https://linkedin.com/in/ghanashyam-tv)")
