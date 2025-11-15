"""
WORKING FAKE NEWS DETECTOR
===========================
Simple, clean, effective
"""

import streamlit as st
import pickle
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="📰 News Verdict", layout="centered", initial_sidebar_state="collapsed")

# CSS STYLING
st.markdown("""
    <style>
        .main { padding-top: 2rem; }
        .verdict-true { 
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            font-size: 72px;
            font-weight: 900;
            letter-spacing: 3px;
            box-shadow: 0 10px 40px rgba(16, 185, 129, 0.3);
            margin: 20px 0;
        }
        .verdict-false { 
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            font-size: 72px;
            font-weight: 900;
            letter-spacing: 3px;
            box-shadow: 0 10px 40px rgba(239, 68, 68, 0.3);
            margin: 20px 0;
        }
        .confidence { 
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 15px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .confidence-number {
            font-size: 36px;
            font-weight: bold;
            color: #1f2937;
        }
        .title { 
            text-align: center;
            font-size: 48px;
            font-weight: 900;
            color: #1f2937;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# TITLE
st.markdown('<div class="title">📰 NEWS VERDICT</div>', unsafe_allow_html=True)

# LOAD MODELS
@st.cache_resource
def load_models():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

# INPUT AREA
text_input = st.text_area("📝 Enter news text to analyze:", height=150, placeholder="Paste your article here...")

# BUTTONS
col1, col2 = st.columns([1, 1])

with col1:
    analyze_btn = st.button("🔍 ANALYZE", use_container_width=True)

with col2:
    demo_btn = st.button("📋 DEMO", use_container_width=True)

# DEMO TEXT
if demo_btn:
    text_input = """CEO Exposed: Shocking revelations emerged about massive financial scandal with billions in offshore accounts. 
    Investigators say this is the biggest financial crime ever. Everyone is shocked by this massive scandal."""

# ANALYSIS
if analyze_btn:
    if not text_input or len(text_input.strip()) == 0:
        st.warning("⚠️ Please enter some text to analyze")
    else:
        model, vectorizer = load_models()
        
        if model and vectorizer:
            try:
                # Vectorize text
                X = vectorizer.transform([text_input])
                
                # Predict
                prediction = model.predict(X)[0]
                confidence = max(model.predict_proba(X)[0])
                
                # Display verdict
                if prediction == 1:
                    st.markdown('<div class="verdict-true">✅ TRUE</div>', unsafe_allow_html=True)
                    verdict_text = "Real News"
                else:
                    st.markdown('<div class="verdict-false">❌ FALSE</div>', unsafe_allow_html=True)
                    verdict_text = "Fake News"
                
                # Display confidence
                st.markdown(f'''
                    <div class="confidence">
                        <div class="confidence-number">{confidence*100:.0f}%</div>
                        <div style="font-size: 14px; color: #6b7280; margin-top: 5px;">Confidence Level</div>
                    </div>
                ''', unsafe_allow_html=True)
                
                # Show result
                st.success(f"**Verdict:** {verdict_text} ({confidence*100:.1f}% confident)")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
        else:
            st.error("❌ Models not loaded. Please check model files exist.")
