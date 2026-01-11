"""
TruthCheck AI - Production-Grade Fake News Detection
A modern, glassmorphism-styled Streamlit application for analyzing news authenticity.
"""

import streamlit as st

import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from typing import Optional, Dict

# ══════════════════════════════════════════════════════════════
# 🎨 CUSTOM CSS - GLASSMORPHISM DESIGN
# ══════════════════════════════════════════════════════════════

def inject_custom_css():
    """Inject modern glassmorphism CSS styling into the Streamlit app."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: #f8f9fa;
        padding: 2rem;
    }
    
    /* Hero Header */
    .hero-header {
        text-align: center;
        padding: 2.5rem 0;
        margin-bottom: 2.5rem;
        background: white;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 0.5rem;
    }
    
    .hero-tagline {
        font-size: 1rem;
        color: #718096;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    /* Glass Cards */
    .glass-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }
    
    .glass-card-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Input Section */
    .stTextArea textarea {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #2d3748 !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #a0aec0 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #4299e1 !important;
        box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1) !important;
    }
    
    /* Custom Button */
    .stButton > button {
        background: #3182ce;
        color: white;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(49, 130, 206, 0.2);
    }
    
    .stButton > button:hover {
        background: #2c5282;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(49, 130, 206, 0.3);
    }
    
    /* Result Card */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin-top: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    .result-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .result-content {
        color: #2d3748;
        font-size: 0.95rem;
        line-height: 1.8;
    }
    
    .result-section {
        margin-bottom: 1.5rem;
    }
    
    .result-label {
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    
    .result-value {
        color: #4a5568;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .result-list {
        margin-left: 0;
        padding-left: 0;
    }
    
    .result-list-item {
        margin-bottom: 1rem;
        padding-left: 1.5rem;
        position: relative;
    }
    
    .result-list-item::before {
        content: "•";
        position: absolute;
        left: 0.5rem;
        color: #3182ce;
        font-weight: bold;
    }
    
    .result-list-label {
        font-weight: 600;
        color: #2d3748;
    }
    
    .result-list-text {
        color: #4a5568;
        margin-top: 0.25rem;
    }
    
    .reliability-score {
        display: inline-block;
        background: #edf2f7;
        color: #2d3748;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .classification-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    .classification-fake {
        background: #fed7d7;
        color: #c53030;
    }
    
    .classification-possibly {
        background: #feebc8;
        color: #c05621;
    }
    
    .classification-real {
        background: #c6f6d5;
        color: #2f855a;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] .element-container {
        color: #2d3748;
    }
    
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
    }
    
    /* Input Fields in Sidebar */
    .stTextInput input {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        color: #2d3748 !important;
    }
    
    .stTextInput input:focus {
        border-color: #4299e1 !important;
        box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1) !important;
    }
    
    /* Error/Warning Messages */
    .stAlert {
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #ffffff !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #718096;
        font-size: 0.875rem;
        margin-top: 3rem;
    }
    
    .footer a {
        color: #3182ce;
        text-decoration: none;
        font-weight: 500;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f7fafc;
        border-radius: 8px;
        color: #2d3748 !important;
        font-weight: 600;
        border: 1px solid #e2e8f0;
    }
    
    /* Markdown in sidebar */
    [data-testid="stSidebar"] h3 {
        color: #2d3748 !important;
        font-size: 1rem !important;
        margin-top: 1rem !important;
    }
    
    [data-testid="stSidebar"] p {
        color: #4a5568 !important;
        font-size: 0.875rem !important;
        line-height: 1.6 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 🧩 UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════

def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()


def get_analysis_prompt(news_text: str) -> str:
    """
    Generate the analysis prompt for fake news detection.
    
    Args:
        news_text: The news article or claim to analyze
        
    Returns:
        Formatted prompt string for the LLM
    """
    prompt = f"""
You are an advanced Fake News Detection AI.

Analyze the following text for:
- Misinformation patterns
- Emotional manipulation
- Logical consistency
- Source reliability
- Plausibility of claims

Output:
- Reliability score (0–100)
- Classification: Likely Fake / Possibly Fake / Likely Real
- A short explanation

Text:
\"\"\"{news_text}\"\"\"
"""
    return prompt


def analyze_news(api_key: str, news_text: str, model: str) -> Optional[str]:
    """
    Analyze news article for authenticity using Groq LLM.
    
    Args:
        api_key: Groq API key
        news_text: The news content to analyze
        model: Model name to use
        
    Returns:
        Analysis result as string, or None if error occurs
        
    Raises:
        Exception: Any error during LLM invocation
    """
    llm = ChatGroq(
        groq_api_key=api_key,
        model=model,
        temperature=0.2
    )
    
    prompt = get_analysis_prompt(news_text)
    response = llm.invoke(prompt)
    
    return response.content


# ══════════════════════════════════════════════════════════════
# 🎨 UI COMPONENTS
# ══════════════════════════════════════════════════════════════

def render_hero_section():
    """Render the hero header section."""
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🛡️ TruthCheck AI</div>
        <div class="hero-tagline">Advanced AI-Powered Fake News Detection</div>
    </div>
    """, unsafe_allow_html=True)


def render_input_card() -> str:
    """
    Render the input card for news article.
    
    Returns:
        User input text
    """
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-header">📝 Enter News Article</div>', unsafe_allow_html=True)
    news_input = st.text_area(
        "",
        height=200,
        placeholder="Paste your news article or claim here...",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    return news_input


def parse_and_render_result(result: str):
    """
    Parse the LLM result and render it in a structured format.
    
    Args:
        result: Raw analysis result text from LLM
    """
    lines = result.strip().split('\n')
    
    reliability_score = None
    classification = None
    explanation_started = False
    explanation_items = []
    current_item = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Parse reliability score
        if 'reliability score' in line.lower():
            parts = line.split(':')
            if len(parts) > 1:
                score_text = parts[1].strip()
                # Extract number from text like "60" or "60/100"
                import re
                match = re.search(r'\d+', score_text)
                if match:
                    reliability_score = match.group()
        
        # Parse classification
        elif 'classification' in line.lower():
            parts = line.split(':')
            if len(parts) > 1:
                classification = parts[1].strip()
        
        # Parse explanation section
        elif 'explanation' in line.lower():
            explanation_started = True
        
        elif explanation_started:
            # Check if this is a new item (starts with -)
            if line.startswith('-') or line.startswith('•'):
                if current_item:
                    explanation_items.append(current_item)
                current_item = {'label': '', 'text': ''}
                # Remove bullet and parse
                cleaned = line.lstrip('-•').strip()
                if ':' in cleaned:
                    label, text = cleaned.split(':', 1)
                    current_item['label'] = label.strip()
                    current_item['text'] = text.strip()
                else:
                    current_item['text'] = cleaned
            elif current_item:
                # Continue previous item
                current_item['text'] += ' ' + line
    
    # Add last item
    if current_item:
        explanation_items.append(current_item)
    
    # Render the structured output
    st.markdown("""
    <div class="result-card">
        <div class="result-header">🧠 Analysis Result</div>
    """, unsafe_allow_html=True)
    
    # Reliability Score
    if reliability_score:
        st.markdown(f"""
        <div class="result-section">
            <div class="result-label">Reliability Score:</div>
            <div class="reliability-score">{reliability_score}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Classification
    if classification:
        classification_lower = classification.lower()
        badge_class = "classification-possibly"
        if "fake" in classification_lower and "possibly" not in classification_lower:
            badge_class = "classification-fake"
        elif "real" in classification_lower:
            badge_class = "classification-real"
            
        st.markdown(f"""
        <div class="result-section">
            <div class="result-label">Classification:</div>
            <div class="classification-badge {badge_class}">{classification}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Explanation
    if explanation_items:
        st.markdown('<div class="result-section"><div class="result-label">Explanation:</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-list">', unsafe_allow_html=True)
        
        for item in explanation_items:
            if item['label']:
                st.markdown(f"""
                <div class="result-list-item">
                    <div class="result-list-label">{item['label']}:</div>
                    <div class="result-list-text">{item['text']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-list-item">
                    <div class="result-list-text">{item['text']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_footer():
    """Render application footer."""
    st.markdown("""
    <div class="footer">
        Powered by <a href="https://groq.com" target="_blank">Groq</a> & LangChain
        <br>
        Built with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 🚀 MAIN APPLICATION
# ══════════════════════════════════════════════════════════════

def main():
    """Main application entry point."""
    
    # Load environment
    load_environment()
    
    # Page configuration
    st.set_page_config(
        page_title="TruthCheck AI - Fake News Detector",
        page_icon="🛡️",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # ═══════════════════════════════════════════════════════════
    # 📌 SIDEBAR
    # ═══════════════════════════════════════════════════════════
    
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🔐 API Configuration</div>', unsafe_allow_html=True)
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="Enter your API key..."
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **TruthCheck AI** uses advanced language models to analyze news articles and detect potential misinformation.
        
        **Model:** `openai/gpt-oss-20b`
        
        **Features:**
        - Pattern detection
        - Emotional analysis
        - Logical consistency check
        - Source reliability scoring
        """)
    
    # ═══════════════════════════════════════════════════════════
    # 🎨 MAIN CONTENT
    # ═══════════════════════════════════════════════════════════
    
    # Hero section
    render_hero_section()
    
    # Input card
    news_input = render_input_card()
    
    # Force model to OSS (unchanged from original)
    MODEL_TO_USE = "openai/gpt-oss-20b"
    
    # Analyze button
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    analyze_button = st.button("🔍 Analyze News Article")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════
    # ⚙️ ANALYSIS LOGIC (UNCHANGED)
    # ═══════════════════════════════════════════════════════════
    
    if analyze_button:
        
        # Validation: API Key
        if not api_key:
            st.error("Please enter your Groq API Key.")
            st.stop()
        
        # Validation: Input text
        if not news_input.strip():
            st.error("Please paste some text to analyze.")
            st.stop()
        
        try:
            with st.spinner("🔄 Analyzing article... Please wait"):
                result = analyze_news(api_key, news_input, MODEL_TO_USE)
            
            # Display result with structured formatting
            parse_and_render_result(result)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Footer
    render_footer()


# ══════════════════════════════════════════════════════════════
# 🎯 ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
