"""
Streamlit UI — EndoCare AI (Endometriosis Clinical Decision Support)
----------------------------------------------------------------------
Chat-first landing page UI on top of the existing retrieval (query.py)
and grounded generation (generate.py) pipeline. Run with:

    streamlit run app_endometriosis.py
"""
import streamlit as st
import streamlit.components.v1 as components

import config
from query import load_index, retrieve
from generate import generate_grounded_answer
from translate import detect_and_translate_to_english, translate_from_english, is_arabic


# ---------- Page setup ----------
st.set_page_config(
    page_title="EndoCare AI",
    page_icon="🎗️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------- Cohesive rose / plum palette (single warm hue family) ----------
DEEP = "#4A2545"     # deep plum
MID = "#8C5B7D"      # dusty mauve
LIGHT = "#E8B4A0"    # warm blush
CREAM = "#FBF3EC"    # warm cream background
INK = "#3A2E38"      # soft near-black for text

SUPPORTIVE_LINES = [
    "You're not alone in this.",
    "It's okay to ask the questions you've been afraid to ask.",
    "Your pain is real, and so is your right to understand it.",
    "Take your time — there's no rush here.",
    "Every question brings you closer to feeling in control.",
]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* Force warm cream background across the entire Streamlit page and containers */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .main {{
    background-color: {CREAM} !important;
    background: {CREAM} !important;
    color: {INK} !important;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 6rem; max-width: 760px; }}

/* Trust strip */
.trust-strip {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    justify-content: center;
    margin: 1.3rem 0 1.6rem 0;
}}
.trust-pill {{
    background: white;
    border: 1px solid #eee0e6;
    border-radius: 999px;
    padding: 0.45rem 0.95rem;
    font-size: 0.78rem;
    color: {INK};
    box-shadow: 0 3px 10px rgba(0,0,0,0.04);
}}

/* Chat message containers */
[data-testid="stChatMessage"] {{
    background: transparent !important;
}}

/* User question bubble */
.user-card {{
    background: {MID};
    color: white !important;
    border-radius: 16px;
    padding: 0.9rem 1.2rem;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 4px 12px rgba(140, 91, 125, 0.18);
}}

/* Assistant answer card */
.assistant-card {{
    background: white;
    border: 1px solid #f0e6ea;
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    color: {INK};
    font-size: 0.93rem;
    line-height: 1.6;
}}

/* Chat input pill (native Streamlit chat input, restyled) */
[data-testid="stChatInput"] {{
    background: transparent !important;
}}
[data-testid="stChatInput"] > div {{
    background: white !important;
    border-radius: 999px !important;
    border: 1px solid #eee0e6 !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.06) !important;
}}
[data-testid="stChatInput"] textarea {{
    background: transparent !important;
    color: {INK} !important;
    font-size: 0.9rem !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
    color: #9c8e98 !important;
}}

.disclaimer-footer {{
    text-align: center;
    font-size: 0.72rem;
    color: #8f7e8b;
    margin-top: 0.6rem;
}}
</style>
""", unsafe_allow_html=True)


# ---------- Backend setup ----------
@st.cache_resource(show_spinner=False)
def get_vectordb():
    return load_index()


def get_api_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return config.GEMINI_API_KEY


api_key = get_api_key()
if not api_key:
    st.error(
        "GEMINI_API_KEY is not set. Add it to your `.env` file locally, "
        "or to Streamlit secrets if this app is deployed."
    )
    st.stop()
else:
    config.GEMINI_API_KEY = api_key

try:
    vectordb = get_vectordb()
except SystemExit:
    st.error(
        "No vector database found. Open a terminal in this project folder and run:\n\n"
        "`python ingest.py`\n\nThen refresh this page."
    )
    st.stop()


# ---------- Session state (Holds only the current active question/answer) ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- Chat input (docked at the bottom) ----------
typed = st.chat_input("Ask about symptoms, treatments, or share how you're feeling...")

if typed:
    # Immediately clear previous history and set only the new question
    st.session_state.messages = [{"role": "user", "content": typed}]

    user_dir = "rtl" if is_arabic(typed) else "ltr"
    with st.chat_message("user"):
        st.markdown(f'<div class="user-card" dir="{user_dir}">{typed}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="🎗️"):
        with st.spinner("Thinking..."):
            try:
                # Translate incoming question to English (auto-detect language)
                try:
                    english_question, lang = detect_and_translate_to_english(typed)
                except Exception:
                    english_question, lang = typed, "en"  # fail safe: proceed as-is

                results = retrieve(vectordb, english_question)
                response = generate_grounded_answer(english_question, results)
                answer_en = response.get("recommendation", "I couldn't find a grounded answer to that.")

                # Translate the answer back to the user's original language
                try:
                    answer_text = translate_from_english(answer_en, lang)
                except Exception:
                    answer_text = answer_en  # fail safe
            except Exception as e:
                answer_text = f"An error occurred: {e}"

        answer_dir = "rtl" if is_arabic(answer_text) else "ltr"
        st.markdown(f'<div class="assistant-card" dir="{answer_dir}">{answer_text}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer_text})

    st.markdown(
        '<div class="disclaimer-footer">ⓘ EndoCare AI provides educational information and does not diagnose or replace professional medical advice.</div>',
        unsafe_allow_html=True,
    )

elif st.session_state.messages:
    # Display the current single Q&A turn
    for msg in st.session_state.messages:
        direction = "rtl" if is_arabic(msg["content"]) else "ltr"
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f'<div class="user-card" dir="{direction}">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🎗️"):
                if msg.get("error"):
                    st.error(msg["error"])
                else:
                    st.markdown(f'<div class="assistant-card" dir="{direction}">{msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="disclaimer-footer">ⓘ EndoCare AI provides educational information and does not diagnose or replace professional medical advice.</div>',
        unsafe_allow_html=True,
    )

else:
    # Landing hero (shown only when no question has been asked yet)
    lines_js = str(SUPPORTIVE_LINES)
    hero_html = f"""
    <div style="font-family:'Inter',sans-serif; margin:0;">
    <style>
      .hero-wrap {{
          position: relative;
          border-radius: 24px;
          overflow: hidden;
          height: 220px;
          background:
              radial-gradient(ellipse 70% 90% at 20% 15%, {DEEP} 0%, transparent 65%),
              radial-gradient(ellipse 80% 90% at 40% 100%, {MID} 0%, transparent 70%),
              radial-gradient(ellipse 70% 70% at 90% 30%, {LIGHT} 0%, transparent 60%),
              linear-gradient(135deg, {DEEP} 0%, {MID} 55%, {LIGHT} 100%);
      }}
      .glass-card {{
          position: absolute;
          top: 50%; left: 50%;
          transform: translate(-50%, -50%);
          width: 84%;
          max-width: 440px;
          background: rgba(255,255,255,0.85);
          backdrop-filter: blur(14px);
          -webkit-backdrop-filter: blur(14px);
          border-radius: 20px;
          padding: 1.4rem 1.6rem;
          text-align: center;
          box-shadow: 0 8px 30px rgba(0,0,0,0.15);
      }}
      .glass-card h1 {{
          font-family: 'Poppins', sans-serif;
          font-size: 1.15rem;
          color: {INK};
          margin: 0 0 0.6rem 0;
      }}
      #rotating-line {{
          font-size: 0.88rem;
          color: {MID};
          min-height: 1.4em;
          transition: opacity 0.6s ease;
          font-weight: 500;
      }}
    </style>
    <div class="hero-wrap">
      <div class="glass-card">
        <h1>Hi, I'm EndoCare AI 🎗️</h1>
        <div id="rotating-line"></div>
      </div>
    </div>
    <script>
      const lines = {lines_js};
      let idx = 0;
      const el = document.getElementById('rotating-line');
      function showLine() {{
        el.style.opacity = 0;
        setTimeout(() => {{
          el.textContent = lines[idx];
          el.style.opacity = 1;
          idx = (idx + 1) % lines.length;
        }}, 400);
      }}
      showLine();
      setInterval(showLine, 3800);
    </script>
    </div>
    """
    components.html(hero_html, height=250, scrolling=False)

    st.markdown("""
    <div class="trust-strip">
        <div class="trust-pill">🔒 Private &amp; confidential</div>
        <div class="trust-pill">📚 Grounded in clinical guidelines</div>
        <div class="trust-pill">🤝 A judgment-free space</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="disclaimer-footer">ⓘ EndoCare AI provides educational information and does not diagnose or replace professional medical advice.</div>',
        unsafe_allow_html=True,
    )