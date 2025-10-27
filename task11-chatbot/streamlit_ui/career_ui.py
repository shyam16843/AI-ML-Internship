import streamlit as st
import ollama
import nltk

print(nltk.__version__)
nltk.data.path.append('C:/Users/User/nltk_data')
# Download punkt tokenizer only once at startup
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')
print(nltk.data.find('tokenizers/punkt/english.pickle'))
tokens = nltk.word_tokenize("Hello world how are you?")
print(tokens)

CAREER_KEYWORDS = {
    "tech": ["software", "python", "engineer", "data", "machine learning", "AI"],
    "arts": ["design", "painting", "music", "writing", "art"],
    "commerce": ["business", "finance", "accounting", "sales", "marketing"]
}

CAREER_RECOMMENDATIONS = {
    "tech": "Recommended careers: Software Engineer, Data Scientist, AI/ML Engineer, Web Developer.",
    "arts": "Recommended careers: Graphic Designer, Writer, Musician, Artist.",
    "commerce": "Recommended careers: Financial Analyst, Accountant, Marketing Specialist, Business Manager."
}

def extract_career_intent(text):
    if not text:
        return None
    words = nltk.word_tokenize(text.lower())
    for intent, kw_list in CAREER_KEYWORDS.items():
        for kw in kw_list:
            if kw in words or kw in text.lower():
                return intent
    return None

def get_llm_response(text):
    try:
        response = ollama.chat(
            model='llama3.2:3b',
            messages=[{"role": "user", "content": text}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error querying LLM: {e}"

st.title("AI Virtual Career Counsellor")

# Use session state for input field
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

response = ""   # holds answer to display

def ask():
    user_input = st.session_state["user_input"]
    intent = extract_career_intent(user_input)
    if intent:
        st.session_state["response"] = CAREER_RECOMMENDATIONS[intent]
    else:
        st.session_state["response"] = get_llm_response(user_input)
    st.session_state["user_input"] = ""  # Auto-clear input

st.text_area("Tell me about your interests and background:", key="user_input")
if st.button("Ask", on_click=ask):
    pass

# Display answer BELOW the input section
if "response" in st.session_state and st.session_state["response"]:
    st.info("LLM Answer (for general guidance):")
    st.write(st.session_state["response"])
