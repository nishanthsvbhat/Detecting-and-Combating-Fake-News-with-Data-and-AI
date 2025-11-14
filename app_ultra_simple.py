"""
ULTRA SIMPLE FAKE NEWS DETECTOR
================================
Minimal, clean interface
Shows only: TRUE or FALSE + Confidence
Nothing else matters
"""

import streamlit as st
import pickle
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="News Verdict", layout="centered")

# Minimal CSS
st.markdown("""
    <style>
        .main { padding-top: 3rem; }
        .verdict-true { 
            font-size: 100px; text-align: center; color: #10b981; font-weight: 900; letter-spacing: 5px;
            padding: 40px; margin: 30px 0; text-shadow: 0 2px 10px rgba(16,185,129,0.3);
        }
        .verdict-false { 
            font-size: 100px; text-align: center; color: #ef4444; font-weight: 900; letter-spacing: 5px;
            padding: 40px; margin: 30px 0; text-shadow: 0 2px 10px rgba(239,68,68,0.3);
        }
        .confidence { 
            font-size: 28px; text-align: center; color: #1f2937; font-weight: bold; margin: 20px 0;
        }
        .title { 
            font-size: 36px; font-weight: 900; text-align: center; margin-bottom: 30px; color: #1f2937;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model_dir = Path('model_artifacts_multi_dataset')
        if model_dir.exists():
            with open(model_dir / 'ensemble_multi.pkl', 'rb') as f:
                model = pickle.load(f)
            with open(model_dir / 'vectorizer_multi.pkl', 'rb') as f:
                vec = pickle.load(f)
            return model, vec
    except:
        pass
    
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vec = pickle.load(f)
        return model, vec
    except:
        return None, None

# HEADER
st.markdown('<div class="title">📰 NEWS VERDICT</div>', unsafe_allow_html=True)

# INPUT
text = st.text_area("Enter news text:", height=120, placeholder="Paste your article...", label_visibility="collapsed")

# BUTTONS
col1, col2 = st.columns(2)
with col1:
    analyze = st.button("🔍 ANALYZE", use_container_width=True)
with col2:
    demo = st.button("📋 DEMO", use_container_width=True)

if demo:
    text = """CEO Exposed: Secret Accounts Found
    Shocking revelations emerged about massive financial scandal with billions in offshore accounts.
    Investigators call it the biggest financial crime ever. Everyone knows this will destroy the company."""

if analyze and text and len(text.strip()) > 0:
    model, vectorizer = load_models()
    
    if model and vectorizer:
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        conf = max(model.predict_proba(X)[0])
        
        # VERDICT
        if pred == 1:
            st.markdown('<div class="verdict-true">TRUE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict-false">FALSE</div>', unsafe_allow_html=True)
        
        # CONFIDENCE
        st.markdown(f'<div class="confidence">{conf*100:.0f}% Confidence</div>', unsafe_allow_html=True)
    else:
        st.error("❌ Train models first")
elif analyze:
    st.warning("⚠️ Enter text")
