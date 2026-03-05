# cardinal_assist_app_v1.py - Fixed Version
# Streamlit App
import streamlit as st
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datetime import datetime
import base64
from pathlib import Path

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Cardinal Assist",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"  # ✅ Sidebar expanded by default
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_base64_image(image_path):
    """Convert image to base64 for CSS background"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None


def load_css():
    """Load custom CSS for modern, futuristic chatbot UI"""
    bg_image = get_base64_image("background.jpg")

    css = f"""
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global styles */
    * {{
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Animated gradient background */
    .stApp {{
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0f0c29);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }}

    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* Cyber grid overlay */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px),
            {f'url(data:image/jpeg;base64,{bg_image})' if bg_image else 'none'};
        background-size: 50px 50px, 50px 50px, cover;
        background-position: center;
        opacity: 0.3;
        z-index: 0;
        pointer-events: none;
    }}

    /* Glassmorphism header */
    .main-header {{
        background: rgba(15, 12, 41, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 255, 255, 0.2);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 40px rgba(0, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }}

    .main-header::before {{
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.1), transparent);
        animation: shine 3s infinite;
    }}

    @keyframes shine {{
        0% {{ left: -100%; }}
        100% {{ left: 200%; }}
    }}

    .main-header h1 {{
        margin: 0;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00fff5 0%, #a78bff 50%, #ff6ec7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 40px rgba(0, 255, 245, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }}

    @keyframes glow {{
        from {{ filter: drop-shadow(0 0 20px rgba(0, 255, 245, 0.5)); }}
        to {{ filter: drop-shadow(0 0 30px rgba(167, 139, 255, 0.8)); }}
    }}

    .main-header p {{
        margin: 0.8rem 0 0 0;
        font-size: 1.3rem;
        color: #a78bff;
        font-weight: 300;
        letter-spacing: 2px;
    }}

    /* Futuristic logo container */
    .logo-container {{
        text-align: center;
        padding: 1.5rem;
        background: rgba(15, 12, 41, 0.6);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 
            0 8px 25px rgba(0, 0, 0, 0.3),
            0 0 30px rgba(0, 255, 255, 0.1);
        transition: all 0.3s ease;
    }}

    .logo-container:hover {{
        transform: translateY(-5px);
        box-shadow: 
            0 12px 35px rgba(0, 0, 0, 0.4),
            0 0 50px rgba(0, 255, 255, 0.2);
    }}

    # /* Modern chat container - optimized */
    # .chat-container {{
    #     background: rgba(15, 12, 41, 0.4);
    #     backdrop-filter: blur(20px);
    #     border: 1px solid rgba(0, 255, 255, 0.15);
    #     border-radius: 24px;
    #     padding: 2rem;
    #     box-shadow: 
    #         0 8px 32px rgba(0, 0, 0, 0.4),
    #         inset 0 1px 0 rgba(255, 255, 255, 0.1);
    #     min-height: 500px;
    #     max-height: 650px;
    #     overflow-y: auto;
    #     margin-bottom: 2rem;
    #     position: relative;
    # }}

    /* Neon user message bubble */
    .user-message {{
        background: linear-gradient(135deg, rgba(0, 255, 245, 0.15) 0%, rgba(167, 139, 255, 0.15) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 245, 0.4);
        color: #ffffff;
        padding: 1.2rem 1.8rem;
        border-radius: 24px 24px 4px 24px;
        margin: 1.2rem 0 1.2rem 15%;
        box-shadow: 
            0 4px 20px rgba(0, 255, 245, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        animation: slideInRight 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        position: relative;
    }}

    .user-message::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00fff5, transparent);
        opacity: 0.6;
    }}

    /* Neon assistant message bubble */
    .assistant-message {{
        background: linear-gradient(135deg, rgba(255, 110, 199, 0.15) 0%, rgba(167, 139, 255, 0.15) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 110, 199, 0.4);
        color: #ffffff;
        padding: 1.2rem 1.8rem;
        border-radius: 24px 24px 24px 4px;
        margin: 1.2rem 15% 1.2rem 0;
        box-shadow: 
            0 4px 20px rgba(255, 110, 199, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        animation: slideInLeft 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        position: relative;
    }}

    .assistant-message::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #ff6ec7, transparent);
        opacity: 0.6;
    }}

    /* Enhanced message label */
    .message-label {{
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #00fff5;
        opacity: 0.9;
    }}

    .assistant-message .message-label {{
        color: #ff6ec7;
    }}

    /* Enhanced message text */
    .message-text {{
        font-size: 1.05rem;
        line-height: 1.7;
        color: #e0e0e0;
        font-weight: 300;
    }}

    /* Futuristic timestamp */
    .timestamp {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        opacity: 0.6;
        text-align: right;
        margin-top: 0.8rem;
        color: #00fff5;
        letter-spacing: 1px;
    }}

    /* Neon sources section */
    .sources-section {{
        background: rgba(0, 255, 245, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 245, 0.3);
        padding: 1.2rem;
        border-radius: 16px;
        margin-top: 1rem;
        box-shadow: 
            0 4px 15px rgba(0, 255, 245, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }}

    .sources-section h4 {{
        margin: 0 0 0.8rem 0;
        color: #00fff5;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }}

    .source-item {{
        font-size: 0.9rem;
        padding: 0.8rem;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 255, 245, 0.2);
        border-radius: 10px;
        margin: 0.6rem 0;
        color: #b0b0b0;
        transition: all 0.3s ease;
    }}

    .source-item:hover {{
        background: rgba(0, 255, 245, 0.08);
        border-color: rgba(0, 255, 245, 0.4);
        transform: translateX(5px);
    }}

    /* Streamlit input field customization */
    .stTextInput > div > div > input {{
        background: rgba(15, 12, 41, 0.6) !important;
        border: 1px solid rgba(0, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 1.05rem !important;
        padding: 0.75rem 1.2rem !important;
        transition: all 0.3s ease !important;
        height: 50px !important;
    }}

    .stTextInput > div > div > input:focus {{
        border-color: #00fff5 !important;
        box-shadow: 0 0 20px rgba(0, 255, 245, 0.3) !important;
        outline: none !important;
    }}

    .stTextInput > div > div > input::placeholder {{
        color: rgba(255, 255, 255, 0.4) !important;
    }}

    /* Perfect button alignment */
    .stButton > button {{
        background: linear-gradient(135deg, #00fff5 0%, #a78bff 100%) !important;
        color: #0f0c29 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 4px 20px rgba(0, 255, 245, 0.4) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
        height: 50px !important;
        min-height: 50px !important;
        line-height: 50px !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 30px rgba(0, 255, 245, 0.6) !important;
    }}

    .stButton > button:active {{
        transform: translateY(0) !important;
    }}

    /* Remove extra padding from button container */
    .stButton {{
        padding-top: 0 !important;
        margin-top: 0 !important;
    }}

    /* Align columns properly */
    .row-widget.stHorizontal {{
        align-items: flex-end !important;
        gap: 0.5rem !important;
    }}

    /* Secondary buttons (sidebar) - same height for consistency */
    .stButton > button[kind="secondary"] {{
        background: rgba(255, 110, 199, 0.2) !important;
        color: #ff6ec7 !important;
        border: 1px solid rgba(255, 110, 199, 0.4) !important;
        height: auto !important;
        min-height: auto !important;
        line-height: normal !important;
        padding: 0.8rem 1rem !important;
    }}

    .stButton > button[kind="secondary"]:hover {{
        background: rgba(255, 110, 199, 0.3) !important;
        border-color: #ff6ec7 !important;
    }}

    /* Selectbox styling */
    .stSelectbox > div > div {{
        background: rgba(15, 12, 41, 0.6) !important;
        border: 1px solid rgba(0, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
    }}

    /* Slider styling */
    .stSlider > div > div > div {{
        background: linear-gradient(90deg, #00fff5, #a78bff) !important;
    }}

    /* Checkbox styling */
    .stCheckbox {{
        color: #ffffff !important;
    }}

    /* Sidebar enhancements */
    section[data-testid="stSidebar"] {{
        background: rgba(15, 12, 41, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(0, 255, 255, 0.2) !important;
    }}

    section[data-testid="stSidebar"] > div {{
        background: transparent !important;
    }}

    /* Sidebar text */
    section[data-testid="stSidebar"] * {{
        color: #e0e0e0 !important;
    }}

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #00fff5 !important;
        text-shadow: 0 0 10px rgba(0, 255, 245, 0.5) !important;
    }}

    /* Smooth animations */
    @keyframes slideInRight {{
        from {{
            opacity: 0;
            transform: translateX(50px) scale(0.95);
        }}
        to {{
            opacity: 1;
            transform: translateX(0) scale(1);
        }}
    }}

    @keyframes slideInLeft {{
        from {{
            opacity: 0;
            transform: translateX(-50px) scale(0.95);
        }}
        to {{
            opacity: 1;
            transform: translateX(0) scale(1);
        }}
    }}

    /* Futuristic welcome message */
    .welcome-message {{
        text-align: center;
        padding: 4rem 2rem;
        color: #e0e0e0;
    }}

    .welcome-message h2 {{
        background: linear-gradient(135deg, #00fff5 0%, #a78bff 50%, #ff6ec7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        animation: glow 2s ease-in-out infinite alternate;
    }}

    .welcome-message p {{
        font-size: 1.15rem;
        line-height: 2;
        color: #b0b0b0;
        font-weight: 300;
    }}

    /* Neon scrollbar */
    .chat-container::-webkit-scrollbar {{
        width: 10px;
    }}

    .chat-container::-webkit-scrollbar-track {{
        background: rgba(15, 12, 41, 0.5);
        border-radius: 10px;
    }}

    .chat-container::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, #00fff5, #a78bff);
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 255, 245, 0.5);
    }}

    .chat-container::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(180deg, #a78bff, #ff6ec7);
        box-shadow: 0 0 15px rgba(255, 110, 199, 0.6);
    }}

    /* Hide default Streamlit elements */
    MainMenu {{visibility: visible;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    # MainMenu {{visibility: visible;}}
    # footer {{visibility: visible;}}
    # header {{visibility: visible;}}

    /* Neon stats box */
    .stats-box {{
        background: linear-gradient(135deg, rgba(0, 255, 245, 0.1) 0%, rgba(167, 139, 255, 0.1) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 255, 245, 0.4);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 
            0 4px 20px rgba(0, 255, 245, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }}

    .stats-box::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00fff5, transparent);
    }}

    .stats-box h3 {{
        margin: 0 0 0.8rem 0;
        font-size: 1.2rem;
        color: #00fff5 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}

    .stats-box p {{
        margin: 0;
        font-size: 1.1rem;
        color: #e0e0e0 !important;
        font-weight: 300;
    }}

    /* Responsive design */
    @media (max-width: 768px) {{
        .main-header h1 {{
            font-size: 2.5rem;
        }}

        .user-message,
        .assistant-message {{
            margin-left: 5%;
            margin-right: 5%;
        }}

        .welcome-message {{
            padding: 2rem 1rem;
        }}

        .welcome-message h2 {{
            font-size: 2rem;
        }}
    }}

    /* Pulsing cursor effect */
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}

    /* Loading indicator */
    .stSpinner > div {{
        border-color: #00fff5 transparent transparent transparent !important;
    }}
    </style>

    <script>
    // Handle Enter key press in text input
    const doc = window.parent.document;

    function setupEnterKeyHandler() {{
        const textInput = doc.querySelector('.stTextInput input');

        if (textInput && !textInput.hasEnterHandler) {{
            textInput.hasEnterHandler = true;
            textInput.addEventListener('keypress', function(e) {{
                if (e.key === 'Enter') {{
                    e.preventDefault();
                    const goButton = doc.querySelector('.stButton button[kind="primary"]');
                    if (goButton) {{
                        goButton.click();
                    }}
                }}
            }});
        }}
    }}

    // Run setup on load and after Streamlit reruns
    setupEnterKeyHandler();
    setInterval(setupEnterKeyHandler, 1000);
    </script>
    """
    st.markdown(css, unsafe_allow_html=True)


FAQ_CATEGORIES = {
    "🎓 Admissions": [
        "What are the admission requirements for undergraduate students?",
        "What are the admission requirements for graduate students?",
        "What is the application deadline for fall semester?",
        "How do I apply for financial aid?",
        "What is the application process for first-year students?"
    ],
    "🌍 International Students": [
        "What is the I-20 processing time for new F-1 visa students?",
        "What documents do international students need for enrollment?",
        "What are the English language proficiency requirements?",
        "How do I request my I-20 document?",
        "What visa types are accepted for international students?"
    ],
    "📚 Registration & Enrollment": [
        "How do I register for classes as a new student?",
        "When does registration open for each semester?",
        "What is the course add/drop deadline?",
        "How do I change my major or minor?",
        "What is the minimum credit requirement per semester?"
    ],
    "💰 Financial Aid & Tuition": [
        "What are the tuition costs for undergraduate students?",
        "What are the tuition costs for international students?",
        "What scholarships are available?",
        "How do I apply for student loans?",
        "What is the refund policy for tuition?"
    ],
    "🏫 Campus Life": [
        "What housing options are available?",
        "What dining plans are offered?",
        "What student organizations can I join?",
        "Where is the student health center?",
        "What recreation facilities are available?"
    ]
}


# ============================================================================
# CARDINAL ASSISTANT CLASS
# ============================================================================

@st.cache_resource
def load_cardinal_assistant():
    """Load and cache the Cardinal Assistant"""

    class CardinalAssistant:
        def __init__(self):
            # Load embedding model
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Load FAISS index and metadata
            if os.path.exists('rag_index.faiss') and os.path.exists('rag_meta.pkl'):
                self.index = faiss.read_index('rag_index.faiss')
                with open('rag_meta.pkl', 'rb') as f:
                    self.meta = pickle.load(f)
            else:
                raise FileNotFoundError(
                    "Index files not found! Make sure 'rag_index.faiss' and 'rag_meta.pkl' are in the same directory.")

            # Load generation model
            model_name = 'google/flan-t5-base'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Use appropriate device
            if torch.cuda.is_available():
                self.generator = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                device = 0
            else:
                self.generator = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                device = -1

            self.gen_pipeline = pipeline(
                'text2text-generation',
                model=self.generator,
                tokenizer=self.tokenizer,
                max_length=512,
                device=device
            )

        def retrieve(self, query, k=5):
            """Retrieve relevant chunks"""
            if self.index is None or self.index.ntotal == 0:
                return []

            q_emb = self.embed_model.encode([query], convert_to_numpy=True)
            D, I = self.index.search(q_emb, k)

            results = []
            for idx, dist in zip(I[0], D[0]):
                if 0 <= idx < len(self.meta):
                    results.append({
                        'text': self.meta[idx]['text'],
                        'source': self.meta[idx]['source'],
                        'score': float(dist)
                    })
            return results

        def generate_answer(self, question, chunks, max_context=800):
            """Generate answer from retrieved chunks"""
            if not chunks:
                return "I don't have enough information to answer that question. Please try rephrasing or contact the admissions office for more details."

            # Build context
            context_parts = []
            total_len = 0

            for chunk in chunks:
                text = chunk['text']
                if total_len + len(text) <= max_context:
                    context_parts.append(text)
                    total_len += len(text)
                else:
                    remaining = max_context - total_len
                    if remaining > 100:
                        context_parts.append(text[:remaining] + "...")
                    break

            context = "\n\n".join(context_parts)

            # Create prompt
            prompt = f"""You are Cardinal Assist, a helpful AI assistant for The Catholic University of America. Answer the student's question using only the information provided in the context below. Be concise, accurate, and helpful.

Context:
{context}

Question: {question}

Answer:"""

            try:
                result = self.gen_pipeline(prompt, max_length=256, do_sample=False)
                answer = result[0]['generated_text'].strip()
                return answer
            except Exception as e:
                return f"I encountered an error while generating the answer. Please try again."

        def ask(self, question, k=5):
            """Full pipeline: retrieve + generate"""
            retrieved = self.retrieve(question, k)
            answer = self.generate_answer(question, retrieved)

            return {
                'question': question,
                'answer': answer,
                'chunks': retrieved,
                'timestamp': datetime.now().strftime("%I:%M %p")
            }

    return CardinalAssistant()


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load CSS
    load_css()

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""

    # Load assistant
    if st.session_state.assistant is None:
        with st.spinner("🔄 Initializing Cardinal Assist..."):
            try:
                st.session_state.assistant = load_cardinal_assistant()
            except Exception as e:
                st.error(f"❌ Error loading assistant: {e}")
                st.info("Make sure 'rag_index.faiss' and 'rag_meta.pkl' are in the same directory as this app.")
                st.stop()

    # Sidebar
    with st.sidebar:
        # Logo section
        # st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        # Try to load logo, fallback to text
        if os.path.exists("logo.png"):
            st.image("logo.png", width='stretch')
        else:
            st.markdown("### 🎓 The Catholic University of America")
        st.markdown('</div>', unsafe_allow_html=True)

        # Stats
        st.markdown(f"""
        <div class="stats-box">
            <h3 style="margin: 0;">📊 Stats</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                {len(st.session_state.chat_history)} conversations<br>
                {st.session_state.assistant.index.ntotal} knowledge chunks
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # FAQ Categories
        st.markdown("### 📋 Browse by Category")

        selected_category = st.selectbox(
            "Choose a category:",
            [""] + list(FAQ_CATEGORIES.keys()),
            format_func=lambda x: "Select a category..." if x == "" else x
        )

        if selected_category and selected_category != "":
            st.markdown("#### Frequently Asked Questions")
            for faq in FAQ_CATEGORIES[selected_category]:
                # ✅ When clicked, populate text box (don't auto-submit)
                if st.button(faq, key=f"faq_{faq}", width='stretch'):
                    st.session_state.current_question = faq
                    st.rerun()  # update the state

        st.markdown("---")

        # Settings
        st.markdown("### ⚙️ Settings")
        num_sources = st.slider("Sources to retrieve", 1, 10, 5)
        show_sources = st.checkbox("Show sources", value=False)  # ✅ OFF by default

        st.markdown("---")

        # Clear button
        if st.button("🗑️ Clear Conversation", width='stretch', type="secondary"):
            st.session_state.chat_history = []
            st.session_state.current_question = ""
            st.rerun()

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; font-size: 0.85rem; color: #666;'>
            <p><strong>Cardinal Assist</strong><br>
            Powered by AI & RAG<br>
            © 2025</p>
        </div>
        """, unsafe_allow_html=True)

    # Main content
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Cardinal Assist</h1>
        <p>Your AI-Powered University Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        # Welcome message
        st.markdown("""
        <div class="welcome-message">
            <h2>👋 Welcome to Cardinal Assist!</h2>
            <p>I'm here to help you with questions about:</p>
            <p>
                🎓 <strong>Admissions</strong> • 
                🌍 <strong>International Students</strong> • 
                📚 <strong>Registration</strong><br>
                💰 <strong>Financial Aid</strong> • 
                🏫 <strong>Campus Life</strong>
            </p>
            <p style="margin-top: 2rem; font-size: 0.95rem; color: #888;">
                Browse FAQs by category in the sidebar, or type your question below!
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display chat history
        for chat in st.session_state.chat_history:
            # User message
            st.markdown(f"""
            <div class="user-message">
                <div class="message-label">👤 You</div>
                <div class="message-text">{chat['question']}</div>
                <div class="timestamp">{chat['timestamp']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Assistant message
            st.markdown(f"""
            <div class="assistant-message">
                <div class="message-label">🎓 Cardinal Assist</div>
                <div class="message-text">{chat['answer']}</div>
                <div class="timestamp">{chat['timestamp']}</div>
            """, unsafe_allow_html=True)

            # Sources
            if show_sources and chat.get('chunks'):
                sources_html = '<div class="sources-section"><h4>📚 Sources:</h4>'
                for i, chunk in enumerate(chat['chunks'][:3], 1):
                    sources_html += f"""
                    <div class="source-item">
                        <strong>{i}.</strong> {chunk['source']} 
                        <em>(relevance: {chunk['score']:.2f})</em>
                    </div>
                    """
                sources_html += '</div>'
                st.markdown(sources_html, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Input section
    col1, col2 = st.columns([6, 1])

    # with col1:
    #     question = st.text_input(
    #         "Type your question here...",
    #         value=st.session_state.current_question,
    #         placeholder="e.g., What are the admission requirements for graduate students?",
    #         label_visibility="collapsed",
    #         key="question_input"
    #     )

    with col1:
        # Dynamic placeholder based on whether FAQ was clicked
        dynamic_placeholder = (
            st.session_state.current_question
            if st.session_state.current_question
            else "e.g., What are the admission requirements for graduate students?"
        )

        question = st.text_input(
            "Type your question here...",
            value="",  # Keep empty so placeholder shows
            placeholder=dynamic_placeholder,  # ✅ Shows FAQ question as placeholder
            label_visibility="collapsed",
            key="question_input"
        )

        # If placeholder has FAQ and user hasn't typed, use it
        if not question and st.session_state.current_question:
            question = st.session_state.current_question

    with col2:
        go_button = st.button("🚀 Go", width='stretch', type="primary")

    # Process question (triggered by Go button or Enter key)
    if go_button and question:
        with st.spinner("🤔 Thinking..."):
            # Get answer
            result = st.session_state.assistant.ask(question, k=num_sources)

            # Add to history
            st.session_state.chat_history.append(result)

            # Clear input
            st.session_state.current_question = ""

            # Rerun to display new message
            st.rerun()


if __name__ == "__main__":
    main()
