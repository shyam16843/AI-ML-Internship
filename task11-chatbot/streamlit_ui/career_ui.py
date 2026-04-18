import streamlit as st
import requests
import nltk
import os

# Download nltk data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Career intent keywords and recommendations
CAREER_KEYWORDS = {
    "tech": ["software", "python", "engineer", "data", "machine learning", "ai", "coding", "programming", "developer", "computer"],
    "arts": ["design", "painting", "music", "writing", "art", "creative", "drawing", "photography", "film"],
    "commerce": ["business", "finance", "accounting", "sales", "marketing", "management", "economics", "banking"]
}

CAREER_RECOMMENDATIONS = {
    "tech": """🖥️ **Tech Career Recommendations**

Based on your interests, here are strong career paths:

- **Software Engineer** — Build applications and systems
- **Data Scientist / ML Engineer** — Work with AI and machine learning
- **Web Developer** — Frontend, backend, or full-stack development
- **DevOps / Cloud Engineer** — Infrastructure and deployment

**Next steps:** Build projects, contribute to open source, get certified (AWS, Google Cloud, TensorFlow).""",

    "arts": """🎨 **Arts & Creative Career Recommendations**

Based on your interests, here are strong career paths:

- **Graphic Designer / UI-UX Designer** — Visual design for apps and websites
- **Content Writer / Copywriter** — Writing for brands and media
- **Musician / Music Producer** — Performance or studio work
- **Photographer / Videographer** — Visual storytelling

**Next steps:** Build a portfolio, freelance on Fiverr/Upwork, connect with creative communities.""",

    "commerce": """💼 **Commerce & Business Career Recommendations**

Based on your interests, here are strong career paths:

- **Financial Analyst** — Investment and financial planning
- **Marketing Specialist** — Digital marketing and brand strategy
- **Business Analyst** — Bridging business and technology
- **Accountant / CA** — Financial management and auditing

**Next steps:** Get certified (CFA, Google Digital Marketing), build Excel/data skills, network on LinkedIn."""
}

def extract_career_intent(text):
    if not text:
        return None
    try:
        words = nltk.word_tokenize(text.lower())
    except Exception:
        words = text.lower().split()
    
    for intent, kw_list in CAREER_KEYWORDS.items():
        for kw in kw_list:
            if kw in words or kw in text.lower():
                return intent
    return None

def get_hf_response(text, hf_token):
    """Query Hugging Face Inference API"""
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    prompt = f"""<s>[INST] You are a helpful AI career counsellor. Give practical, encouraging career advice in 3-4 sentences.

User question: {text} [/INST]"""
        payload = {
        "inputs": f"You are a career counsellor. Give practical career advice in 3-4 sentences. Question: {text}",
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '').strip()
            return "I received a response but couldn't parse it. Please try again."
        elif response.status_code == 503:
            return "The AI model is loading (this takes ~20 seconds on first use). Please click Ask again in a moment."
        else:
            return f"API error {response.status_code}. Please check your Hugging Face token in settings."
    except requests.exceptions.Timeout:
        return "Request timed out. The model may be loading — please try again in 20 seconds."
    except Exception as e:
        return f"Connection error: {str(e)}"

# Page config
st.set_page_config(
    page_title="AI Virtual Career Counsellor",
    page_icon="🎯",
    layout="centered"
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; color: #1f77b4; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { text-align: center; color: #6c757d; margin-bottom: 2rem; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎯 AI Virtual Career Counsellor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Get personalized career guidance powered by NLP + AI</div>', unsafe_allow_html=True)

# Get HF token from Streamlit secrets
hf_token = None
try:
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    pass

if not hf_token:
    st.warning("⚠️ Hugging Face API token not configured. Rule-based recommendations will still work, but AI responses require a token.")

# Sidebar
st.sidebar.title("💡 How to use")
st.sidebar.markdown("""
1. Type your interests or career question
2. Click **Ask** to get recommendations
3. For tech/arts/commerce topics, you get instant recommendations
4. For other questions, the AI generates personalized advice

**Example questions:**
- "I love Python and machine learning"
- "I enjoy creative writing and design"
- "I'm interested in finance and business"
- "What skills do I need for data science?"
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by Ghanashyam T V**")
st.sidebar.markdown("[GitHub](https://github.com/shyam16843) | [LinkedIn](https://linkedin.com/in/ghanashyam-tv)")

# Sample questions
st.markdown("**Try these:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("💻 I love coding and AI", use_container_width=True):
        st.session_state["user_input"] = "I love coding and AI"
with col2:
    if st.button("🎨 I enjoy art and design", use_container_width=True):
        st.session_state["user_input"] = "I enjoy art and design"
with col3:
    if st.button("💼 I'm into business and finance", use_container_width=True):
        st.session_state["user_input"] = "I'm into business and finance"

st.markdown("---")

# Initialize session state
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
if "response" not in st.session_state:
    st.session_state["response"] = ""
if "response_type" not in st.session_state:
    st.session_state["response_type"] = ""

def ask():
    user_input = st.session_state["user_input"]
    if not user_input.strip():
        return
    
    intent = extract_career_intent(user_input)
    
    if intent:
        st.session_state["response"] = CAREER_RECOMMENDATIONS[intent]
        st.session_state["response_type"] = "rule"
    else:
        if hf_token:
            with st.spinner("AI is thinking..."):
                st.session_state["response"] = get_hf_response(user_input, hf_token)
                st.session_state["response_type"] = "llm"
        else:
            st.session_state["response"] = "I couldn't detect a specific career domain from your message. Try mentioning interests like 'coding', 'design', 'finance', or 'machine learning' for instant recommendations."
            st.session_state["response_type"] = "fallback"
    
    st.session_state["user_input"] = ""

# Input
user_input = st.text_area(
    "Tell me about your interests and background:",
    key="user_input",
    height=120,
    placeholder="e.g. I love working with data and Python, and I'm interested in machine learning..."
)

st.button("🎯 Ask", on_click=ask, type="primary", use_container_width=True)

# Display response
if st.session_state.get("response"):
    st.markdown("---")
    if st.session_state["response_type"] == "rule":
        st.success("**Career Recommendation:**")
    elif st.session_state["response_type"] == "llm":
        st.info("**AI Career Advice:**")
    else:
        st.warning("**Guidance:**")
    
    st.markdown(st.session_state["response"])
