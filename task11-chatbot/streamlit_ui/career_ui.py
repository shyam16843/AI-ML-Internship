import streamlit as st
import requests
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

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

# Only trigger rule-based response for very simple one-topic messages
SIMPLE_TRIGGERS = {
    "tech": ["i love coding", "i like coding", "i enjoy coding", "i love programming", 
             "i like programming", "i love software", "i like software", "i love computers",
             "i am a developer", "i want to be a developer"],
    "arts": ["i love art", "i like art", "i enjoy art", "i love design", "i like design",
             "i love drawing", "i love painting", "i love music", "i love writing",
             "i love photography", "i enjoy creative"],
    "commerce": ["i love business", "i like business", "i love finance", "i like finance",
                 "i love accounting", "i like marketing", "i love economics", "i like banking"]
}

def extract_career_intent(text):
    """Only match simple trigger phrases — everything else goes to Gemini"""
    text_lower = text.lower().strip()
    for intent, triggers in SIMPLE_TRIGGERS.items():
        for trigger in triggers:
            if trigger in text_lower:
                return intent
    return None  # All other messages go to Gemini

def get_gemini_response(conversation_history, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    contents = []
    for msg in conversation_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })

    payload = {
        "system_instruction": {
            "parts": [{"text": "You are a friendly and expert AI career counsellor. Give practical, encouraging, and personalized career advice. Keep responses concise (3-5 sentences). Ask follow-up questions to better understand the user's background and goals."}]
        },
        "contents": contents
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        elif response.status_code == 429:
            return "⏳ Rate limit reached. Please wait a moment and try again."
        elif response.status_code == 400:
            return "❌ Invalid API key. Please check your Gemini API key in Streamlit secrets."
        else:
            return f"❌ API error {response.status_code}. Please try again."
    except requests.exceptions.Timeout:
        return "⏳ Request timed out. Please try again."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Page config
st.set_page_config(page_title="AI Career Counsellor", page_icon="🎯", layout="centered")

st.markdown("""
<style>
    .main-header { font-size: 2rem; color: #1f77b4; text-align: center; margin-bottom: 0.3rem; }
    .sub-header { text-align: center; color: #6c757d; margin-bottom: 1rem; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎯 AI Career Counsellor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your personal career guidance chatbot powered by Gemini AI</div>', unsafe_allow_html=True)

# Get Gemini API key
gemini_key = None
try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not gemini_key:
    st.warning("⚠️ Gemini API key not configured. Add GEMINI_API_KEY to Streamlit secrets.")

# Sidebar
st.sidebar.title("💡 How to use")
st.sidebar.markdown("""
Chat naturally with the AI career counsellor!

**Try asking:**
- "I love coding, what career suits me?"
- "I enjoy art and design, what should I do?"
- "What skills do I need for data science?"
- "I love Python and ML, which should I specialize in?"
- "What is the scope of AI engineering?"
""")
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state["messages"] = []
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.markdown("**Built by Ghanashyam T V**")
st.sidebar.markdown("[GitHub](https://github.com/shyam16843) | [LinkedIn](https://linkedin.com/in/ghanashyam-tv)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "👋 Hi! I'm your AI Career Counsellor. Tell me about your interests, skills, or what you're passionate about — and I'll help guide you toward the right career path. What would you like to explore today?"
        }
    ]

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            intent = extract_career_intent(prompt)
            if intent:
                response = CAREER_RECOMMENDATIONS[intent]
            elif gemini_key:
                response = get_gemini_response(st.session_state["messages"], gemini_key)
            else:
                response = "Please configure your Gemini API key in Streamlit secrets to get AI-powered responses!"
            st.markdown(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})
