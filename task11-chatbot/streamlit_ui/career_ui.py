import streamlit as st
import requests
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

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
    
    # If the message is a follow-up/specific question, let Gemini handle it
    followup_keywords = ["specialize", "specialise", "what should", "which one", 
                         "best for me", "how to", "how do", "difference", "between",
                         "suggest", "recommend", "path", "roadmap", "salary", "scope"]
    if any(kw in text.lower() for kw in followup_keywords):
        return None  # Send to Gemini for personalized answer

    try:
        words = nltk.word_tokenize(text.lower())
    except Exception:
        words = text.lower().split()

    # Only match intent for simple/short introductory messages
    if len(words) > 12:
        return None  # Long questions go to Gemini

    for intent, kw_list in CAREER_KEYWORDS.items():
        for kw in kw_list:
            if kw in words or kw in text.lower():
                return intent
    return None

def get_gemini_response(conversation_history, api_key):
    """Query Gemini with full conversation history for true multi-turn chat"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    # Build contents from conversation history
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
    .user-msg { background-color: #1f77b4; color: white; padding: 10px 15px; border-radius: 18px 18px 4px 18px; margin: 5px 0; max-width: 80%; margin-left: auto; text-align: right; }
    .bot-msg { background-color: #f0f2f6; color: #333; padding: 10px 15px; border-radius: 18px 18px 18px 4px; margin: 5px 0; max-width: 80%; }
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
- "What career suits me if I love coding?"
- "I enjoy art and design, what should I do?"
- "What skills do I need for data science?"
- "I'm confused between tech and business"
- "What should I specialize in?"
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
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            intent = extract_career_intent(prompt)
            if intent:
                response = CAREER_RECOMMENDATIONS[intent]
            elif gemini_key:
                response = get_gemini_response(st.session_state["messages"], gemini_key)
            else:
                response = "Please configure your Gemini API key in Streamlit secrets to get AI-powered responses. For now, try mentioning 'coding', 'design', or 'finance' for instant recommendations!"
            st.markdown(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})
