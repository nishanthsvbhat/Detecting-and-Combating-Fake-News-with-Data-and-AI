"""
🏆 FAKE NEWS DETECTION SYSTEM - OLLAMA INTEGRATED v4.0
======================================================
Streamlined with Local & Cloud LLMs:
✅ 5 ML Models (Ensemble Voting)
✅ Ollama LLM (Local - Fast & Private)
✅ Google Gemini (Cloud - Powerful)
✅ NewsAPI Integration
✅ Direct Text Input
✅ No Complex Options

Author: Nishanth
Repository: https://github.com/nishanthsvbhat/Detecting-and-Combating-Fake-News-with-Data-and-AI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

load_dotenv()

st.set_page_config(
    page_title="🔍 Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1rem;
        font-weight: bold;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .real-news {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .fake-news {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

MIN_TEXT_LENGTH = 50
MAX_TEXT_LENGTH = 10000

BIAS_KEYWORDS = {
    'emotional': ['disaster', 'miracle', 'shocking', 'heartbreaking', 'devastating', 'unbelievable', 'incredible'],
    'political': ['left', 'right', 'conservative', 'liberal', 'trump', 'biden', 'democrat', 'republican'],
    'hyperbolic': ['always', 'never', 'everyone', 'nobody', 'best', 'worst', 'incredible', 'terrible'],
    'source_attack': ['they', 'them', 'elites', 'establishment', 'conspiracy', 'cover-up'],
    'conspiracy': ['hoax', 'fake', 'lie', 'truth', 'exposed', 'hidden', 'secret']
}

# ============================================================================
# CACHING & MODEL LOADING
# ============================================================================

@st.cache_resource
def load_all_ml_models():
    """Load and train all ML models with error handling"""
    try:
        if not os.path.exists('True.csv') or not os.path.exists('Fake.csv'):
            st.error("❌ CSV files not found!")
            return None
        
        # Load data
        true_df = pd.read_csv('True.csv')
        fake_df = pd.read_csv('Fake.csv')
        
        # Detect columns
        title_col = 'title' if 'title' in true_df.columns else true_df.columns[0]
        text_col = 'text' if 'text' in true_df.columns else true_df.columns[1] if len(true_df.columns) > 1 else title_col
        
        # Prepare data
        X_real = (true_df[title_col].fillna('') + ' ' + true_df[text_col].fillna('').astype(str)).str[:500]
        X_fake = (fake_df[title_col].fillna('') + ' ' + fake_df[text_col].fillna('').astype(str)).str[:500]
        
        X = pd.concat([X_real, X_fake], ignore_index=True)
        y = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_fake))])
        
        # Vectorizer
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        X_vectorized = vectorizer.fit_transform(X)
        
        # Train models
        models = {}
        
        # 1. PassiveAggressive
        pa = PassiveAggressiveClassifier(max_iter=1000, random_state=42, n_jobs=-1)
        pa.fit(X_vectorized, y)
        models['pa'] = pa
        
        # 2. RandomForest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
        rf.fit(X_vectorized, y)
        models['rf'] = rf
        
        # 3. SVM
        svm = LinearSVC(max_iter=1000, random_state=42)
        svm.fit(X_vectorized, y)
        models['svm'] = svm
        
        # 4. Naive Bayes
        nb = MultinomialNB()
        nb.fit(X_vectorized, y)
        models['nb'] = nb
        
        # 5. XGBoost
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_vectorized, y)
        models['xgb'] = xgb_model
        
        return {
            'models': models,
            'vectorizer': vectorizer,
            'real_count': len(X_real),
            'fake_count': len(X_fake)
        }
    
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None


def initialize_gemini():
    """Initialize Google Gemini API"""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return False
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        return False


def check_ollama_available():
    """Check if Ollama is running and available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


# ============================================================================
# PREDICTION & ANALYSIS FUNCTIONS
# ============================================================================

def predict_with_ensemble(text: str, model_data: Dict) -> Dict:
    """Predict using ensemble of all 5 models"""
    try:
        if not model_data or 'models' not in model_data:
            return {'error': 'Models not loaded'}
        
        # Vectorize text
        X = model_data['vectorizer'].transform([text[:500]])
        
        # Get predictions from all models
        predictions = {}
        confidences = {}
        
        # PassiveAggressive
        pred_pa = model_data['models']['pa'].predict(X)[0]
        prob_pa = model_data['models']['pa'].decision_function(X)[0]
        predictions['PassiveAggressive'] = 'REAL' if pred_pa == 1 else 'FAKE'
        confidences['PassiveAggressive'] = abs(prob_pa)
        
        # RandomForest
        pred_rf = model_data['models']['rf'].predict(X)[0]
        prob_rf = model_data['models']['rf'].predict_proba(X)[0]
        predictions['RandomForest'] = 'REAL' if pred_rf == 1 else 'FAKE'
        confidences['RandomForest'] = max(prob_rf)
        
        # SVM
        pred_svm = model_data['models']['svm'].predict(X)[0]
        prob_svm = model_data['models']['svm'].decision_function(X)[0]
        predictions['SVM'] = 'REAL' if pred_svm == 1 else 'FAKE'
        confidences['SVM'] = abs(prob_svm)
        
        # Naive Bayes
        pred_nb = model_data['models']['nb'].predict(X)[0]
        prob_nb = model_data['models']['nb'].predict_proba(X)[0]
        predictions['NaiveBayes'] = 'REAL' if pred_nb == 1 else 'FAKE'
        confidences['NaiveBayes'] = max(prob_nb)
        
        # XGBoost
        pred_xgb = model_data['models']['xgb'].predict(X)[0]
        prob_xgb = model_data['models']['xgb'].predict_proba(X)[0]
        predictions['XGBoost'] = 'REAL' if pred_xgb == 1 else 'FAKE'
        confidences['XGBoost'] = max(prob_xgb)
        
        # Ensemble voting
        real_votes = sum(1 for v in predictions.values() if v == 'REAL')
        fake_votes = sum(1 for v in predictions.values() if v == 'FAKE')
        ensemble_verdict = 'REAL' if real_votes >= 3 else 'FAKE'
        
        # Average confidence
        avg_confidence = np.mean(list(confidences.values()))
        
        # Risk level
        if ensemble_verdict == 'FAKE':
            risk_level = 'HIGH' if avg_confidence > 0.7 else 'MEDIUM'
        else:
            risk_level = 'LOW' if avg_confidence > 0.7 else 'MEDIUM'
        
        return {
            'verdict': ensemble_verdict,
            'confidence': min(100, int(avg_confidence * 100)),
            'real_votes': real_votes,
            'fake_votes': fake_votes,
            'risk_level': risk_level,
            'individual_predictions': predictions,
            'confidences': confidences
        }
    
    except Exception as e:
        return {'error': f'Prediction error: {str(e)}'}


def analyze_with_ollama(text: str, prediction: Dict) -> str:
    """Analyze with Ollama (Local LLM)"""
    try:
        if not check_ollama_available():
            return "❌ Ollama not running. Start with: ollama serve"
        
        prompt = f"""Analyze this news article for authenticity and bias:

Article: {text[:1000]}

ML Analysis Result: {prediction['verdict']} (Confidence: {prediction['confidence']}%)

Please provide:
1. Brief authenticity assessment
2. Language tone analysis
3. Potential bias indicators
4. Key claims that should be verified
5. Overall trustworthiness score (0-100)

Keep response concise and actionable."""
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2",  # or mistral, neural-chat, etc.
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', "❌ No response from Ollama")
        else:
            return f"❌ Ollama error: {response.status_code}"
    
    except requests.exceptions.Timeout:
        return "⏱️ Ollama analysis timed out (>30s). Try shorter text or run on faster hardware."
    except Exception as e:
        return f"⚠️ Ollama analysis failed: {str(e)}"


def analyze_with_gemini(text: str, prediction: Dict) -> str:
    """Analyze with Google Gemini"""
    try:
        if not initialize_gemini():
            return "❌ Gemini API not configured"
        
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""Analyze this news article for authenticity and bias:

Article: {text[:1000]}

ML Analysis Result: {prediction['verdict']} (Confidence: {prediction['confidence']}%)

Please provide:
1. Brief authenticity assessment
2. Language tone analysis
3. Potential bias indicators
4. Key claims that should be verified
5. Overall trustworthiness score (0-100)

Keep response concise and actionable."""
        
        response = model.generate_content(prompt, stream=False)
        return response.text if response else "❌ No response from Gemini"
    
    except Exception as e:
        return f"⚠️ Gemini analysis failed: {str(e)}"


def detect_bias(text: str) -> Dict:
    """Detect bias in text"""
    text_lower = text.lower()
    bias_detected = {}
    
    for category, keywords in BIAS_KEYWORDS.items():
        found = [kw for kw in keywords if kw in text_lower]
        if found:
            bias_detected[category] = found
    
    return bias_detected


def fetch_related_articles(query: str, limit: int = 5) -> List[Dict]:
    """Fetch related articles from NewsAPI"""
    try:
        api_key = os.getenv('NEWS_API_KEY')
        if not api_key:
            return []
        
        # Use first 10 words as search query
        search_query = ' '.join(query.split()[:10])
        
        url = f"https://newsapi.org/v2/everything?q={search_query}&sortBy=relevancy&language=en&pageSize={limit}"
        headers = {'X-Api-Key': api_key}
        
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            return [
                {
                    'title': a.get('title', 'N/A'),
                    'source': a.get('source', {}).get('name', 'N/A'),
                    'url': a.get('url', '#')
                }
                for a in articles[:limit]
            ]
        return []
    
    except Exception as e:
        return []


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    # Header
    st.markdown("# 🔍 Fake News Detection System")
    
    # Check LLM availability
    ollama_available = check_ollama_available()
    gemini_available = initialize_gemini()
    
    status_text = "**Analyze news with 5 ML models + "
    if ollama_available:
        status_text += "Ollama (Local)"
        if gemini_available:
            status_text += " + Gemini (Cloud)"
    elif gemini_available:
        status_text += "Gemini (Cloud)"
    else:
        status_text += "LLM (Not available)"
    status_text += "**"
    
    st.markdown(status_text)
    st.divider()
    
    # Load models
    with st.spinner('⏳ Loading ML models...'):
        model_data = load_all_ml_models()
    
    if not model_data:
        st.error("❌ Failed to load models")
        return
    
    # Initialize session state
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    
    # ================================================================
    # MAIN CONTENT
    # ================================================================
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        st.subheader("📝 Enter Article Text")
        text_input = st.text_area(
            "Paste or type your article here:",
            height=200,
            placeholder="Type your article text here...",
            label_visibility="collapsed"
        )
        
        # Character count
        char_count = len(text_input)
        if char_count < MIN_TEXT_LENGTH:
            st.info(f"📝 {char_count}/{MIN_TEXT_LENGTH} characters (need {MIN_TEXT_LENGTH - char_count} more)")
    
    with col2:
        st.subheader("⚙️ Options")
        detect_bias_checkbox = st.checkbox("🔍 Detect Bias", value=True)
        fetch_articles_checkbox = st.checkbox("📰 Find Related", value=True)
        
        # LLM selection
        st.write("**LLM:**")
        llm_options = []
        if ollama_available:
            llm_options.append("🟢 Ollama (Local)")
        if gemini_available:
            llm_options.append("🔵 Gemini (Cloud)")
        
        if not llm_options:
            st.warning("⚠️ No LLM available")
            selected_llm = None
        elif len(llm_options) == 1:
            selected_llm = llm_options[0]
            st.success(f"✅ {selected_llm}")
        else:
            selected_llm = st.radio("Choose LLM:", llm_options, label_visibility="collapsed")
    
    st.divider()
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.analysis_result = None
            st.session_state.text_input = ""
            st.rerun()
    
    # ================================================================
    # ANALYSIS
    # ================================================================
    
    if analyze_button:
        if char_count < MIN_TEXT_LENGTH:
            st.error(f"❌ Please provide at least {MIN_TEXT_LENGTH} characters of text")
        elif char_count > MAX_TEXT_LENGTH:
            st.error(f"❌ Text exceeds {MAX_TEXT_LENGTH} characters")
        else:
            with st.spinner('🔄 Analyzing article...'):
                # ML Prediction
                prediction = predict_with_ensemble(text_input, model_data)
                st.session_state.analysis_result = {
                    'prediction': prediction,
                    'text': text_input,
                    'detect_bias': detect_bias_checkbox,
                    'fetch_articles': fetch_articles_checkbox,
                    'selected_llm': selected_llm
                }
    
    # ================================================================
    # DISPLAY RESULTS
    # ================================================================
    
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        prediction = result['prediction']
        
        if 'error' in prediction:
            st.error(f"❌ {prediction['error']}")
            return
        
        st.divider()
        st.subheader("📊 Analysis Results")
        
        # Main verdict
        verdict = prediction['verdict']
        confidence = prediction['confidence']
        risk_level = prediction['risk_level']
        
        # Color-coded result box
        if verdict == 'REAL':
            st.markdown(f"""
            <div class="result-box real-news">
            <h3>✅ VERDICT: REAL NEWS</h3>
            <p><b>Confidence:</b> {confidence}%</p>
            <p><b>Risk Level:</b> {risk_level}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box fake-news">
            <h3>❌ VERDICT: FAKE NEWS</h3>
            <p><b>Confidence:</b> {confidence}%</p>
            <p><b>Risk Level:</b> {risk_level}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["🤖 Model Breakdown", "🧠 AI Analysis", "🔍 Bias Detection", "📰 Related Articles"])
        
        with tab1:
            st.subheader("Individual Model Predictions")
            
            # Model consensus
            real_votes = prediction['real_votes']
            fake_votes = prediction['fake_votes']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Models Voting REAL", real_votes)
            with col2:
                st.metric("Models Voting FAKE", fake_votes)
            with col3:
                st.metric("Ensemble Vote", f"{real_votes}/5")
            
            st.divider()
            
            # Individual predictions table
            st.write("**Individual Model Verdicts:**")
            
            predictions_data = []
            for model_name, verdict in prediction['individual_predictions'].items():
                confidence_val = prediction['confidences'].get(model_name, 0)
                confidence_pct = int(confidence_val * 100) if confidence_val < 1 else int(confidence_val)
                predictions_data.append({
                    'Model': model_name,
                    'Verdict': verdict,
                    'Confidence': f"{confidence_pct}%"
                })
            
            df_predictions = pd.DataFrame(predictions_data)
            st.dataframe(df_predictions, use_container_width=True, hide_index=True)
            
            # Confidence visualization
            st.write("**Confidence Scores:**")
            fig = go.Figure(data=[
                go.Bar(
                    x=list(prediction['confidences'].keys()),
                    y=[v * 100 if v < 1 else v for v in prediction['confidences'].values()],
                    marker_color=['#28a745' if prediction['individual_predictions'][k] == 'REAL' else '#dc3545' 
                                 for k in prediction['confidences'].keys()]
                )
            ])
            fig.update_layout(
                title="Model Confidence Scores",
                xaxis_title="Model",
                yaxis_title="Confidence (%)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("🧠 AI Analysis")
            
            if not result['selected_llm']:
                st.warning("❌ No LLM configured or available")
            else:
                with st.spinner('🤖 Analyzing with AI...'):
                    if "Ollama" in result['selected_llm']:
                        ai_analysis = analyze_with_ollama(result['text'], prediction)
                    else:
                        ai_analysis = analyze_with_gemini(result['text'], prediction)
                
                st.markdown(ai_analysis)
        
        with tab3:
            if result['detect_bias']:
                st.subheader("🔍 Bias Detection")
                bias_found = detect_bias(result['text'])
                
                if bias_found:
                    st.warning("⚠️ Potential bias indicators found:")
                    for category, keywords in bias_found.items():
                        st.write(f"**{category.title()}:** {', '.join(keywords)}")
                else:
                    st.success("✅ No obvious bias indicators detected")
            else:
                st.info("Bias detection not enabled")
        
        with tab4:
            if result['fetch_articles']:
                st.subheader("📰 Related Articles from NewsAPI")
                with st.spinner('📰 Fetching related articles...'):
                    related = fetch_related_articles(result['text'][:100])
                
                if related:
                    for i, article in enumerate(related, 1):
                        st.write(f"**{i}. {article['title']}**")
                        st.caption(f"Source: {article['source']}")
                        st.write(f"[Read more →]({article['url']})")
                        st.divider()
                else:
                    st.info("No related articles found")
            else:
                st.info("Related articles not enabled")


if __name__ == '__main__':
    main()
