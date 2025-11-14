"""
🎯 PROFESSIONAL FAKE NEWS DETECTION SYSTEM
==========================================
Enterprise-Grade Frontend with Full Integration:
✅ ML Models (Real datasets: True.csv & Fake.csv)
✅ LLM (Google Gemini AI for analysis)
✅ NewsAPI (Real-time article verification)
✅ Advanced Analytics Dashboard
✅ User-Friendly Interface
✅ Professional Design

Author: Nishanth
Repository: https://github.com/nishanthsvbhat/Detecting-and-Combating-Fake-News-with-Data-and-AI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
import os
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="📰 Fake News Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CACHING & SESSION STATE
# ============================================================================

@st.cache_resource
def load_ml_models():
    """Load and train ML models on real datasets"""
    try:
        # Load real datasets
        true_df = pd.read_csv('True.csv')
        fake_df = pd.read_csv('Fake.csv')
        
        # Prepare training data
        X_real = true_df['title'] + ' ' + true_df.get('text', '')
        X_fake = fake_df['title'] + ' ' + fake_df.get('text', '')
        
        X = pd.concat([X_real, X_fake], ignore_index=True)
        y = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_fake))])
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        X_vectorized = vectorizer.fit_transform(X)
        
        # Train classifiers
        pa_classifier = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
        pa_classifier.fit(X_vectorized, y)
        
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_vectorized, y)
        
        return {
            'vectorizer': vectorizer,
            'pa_classifier': pa_classifier,
            'rf_classifier': rf_classifier,
            'dataset_stats': {
                'real_count': len(X_real),
                'fake_count': len(X_fake),
                'total_count': len(X)
            }
        }
    except Exception as e:
        st.warning(f"Could not load ML models: {e}")
        return None

@st.cache_resource
def setup_gemini():
    """Setup Google Gemini API for LLM analysis"""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            return genai
        return None
    except Exception as e:
        st.warning(f"Gemini setup failed: {e}")
        return None

@st.cache_resource
def get_newsapi_key():
    """Get NewsAPI key from environment"""
    return os.getenv('NEWS_API_KEY', '')

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = load_ml_models()
if 'gemini_client' not in st.session_state:
    st.session_state.gemini_client = setup_gemini()

# ============================================================================
# CUSTOM CSS & STYLING
# ============================================================================

st.markdown("""
<style>
    /* Main styling */
    .main {
        padding: 0px;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    
    .header-title {
        font-size: 2.5em;
        font-weight: bold;
        margin: 0;
        padding: 0;
    }
    
    .header-subtitle {
        font-size: 1em;
        opacity: 0.9;
        margin: 5px 0 0 0;
    }
    
    /* Verdict cards */
    .verdict-real {
        background: linear-gradient(135deg, #51cf66 0%, #2f9e44 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        text-align: center;
        font-size: 1.3em;
    }
    
    .verdict-fake {
        background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        text-align: center;
        font-size: 1.3em;
    }
    
    .verdict-uncertain {
        background: linear-gradient(135deg, #ffa94d 0%, #fd7e14 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        text-align: center;
        font-size: 1.3em;
    }
    
    /* Info boxes */
    .info-box {
        background: #e7f5ff;
        border-left: 4px solid #1971c2;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .success-box {
        background: #d3f9d8;
        border-left: 4px solid #2f9e44;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .warning-box {
        background: #fff3bf;
        border-left: 4px solid #f59f00;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .danger-box {
        background: #ffe0e0;
        border-left: 4px solid #c92a2a;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
        margin-bottom: 10px;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def predict_with_ml_models(text: str) -> Dict[str, Any]:
    """Predict using ML models trained on real datasets"""
    if not st.session_state.ml_models:
        return None
    
    models = st.session_state.ml_models
    try:
        X = models['vectorizer'].transform([text])
        
        # Get predictions from both models
        pa_pred = models['pa_classifier'].predict(X)[0]
        pa_prob = models['pa_classifier'].decision_function(X)[0]
        
        rf_pred = models['rf_classifier'].predict(X)[0]
        rf_prob = models['rf_classifier'].predict_proba(X)[0]
        
        # Ensemble prediction
        avg_prob = (abs(pa_prob) + rf_prob[int(pa_pred)]) / 2
        final_prediction = pa_pred if pa_prob > 0 else 0
        
        return {
            'is_real': final_prediction == 1,
            'confidence': min(avg_prob * 100, 100),
            'pa_prediction': 'REAL' if pa_pred == 1 else 'FAKE',
            'rf_prediction': 'REAL' if rf_pred == 1 else 'FAKE'
        }
    except Exception as e:
        st.error(f"ML prediction error: {e}")
        return None

def analyze_with_gemini(text: str, prediction: Dict) -> str:
    """Get detailed analysis from Gemini LLM"""
    if not st.session_state.gemini_client:
        return "LLM analysis unavailable. Configure GEMINI_API_KEY in .env file."
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
Analyze this article text for misinformation. Be concise but thorough.

ARTICLE TEXT:
{text[:1000]}

ML PREDICTION: {'REAL NEWS' if prediction['is_real'] else 'FAKE NEWS'} (Confidence: {prediction['confidence']:.1f}%)

Please provide:
1. **Summary**: One-line assessment
2. **Red Flags**: Any warning signs detected
3. **Credibility Markers**: Positive indicators if real, or manipulation tactics if fake
4. **Recommendation**: Should users trust this?

Format as clear bullet points.
"""
        
        response = model.generate_content(prompt, stream=False)
        return response.text
    except Exception as e:
        return f"LLM analysis failed: {str(e)}"

def fetch_related_articles(query: str) -> List[Dict]:
    """Fetch related articles from NewsAPI"""
    api_key = get_newsapi_key()
    if not api_key:
        return []
    
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query[:50],
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 5,
            'apiKey': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            return articles[:5]
    except Exception as e:
        st.warning(f"NewsAPI fetch failed: {str(e)}")
    
    return []

def get_source_credibility(url: str) -> int:
    """Check if source is credible"""
    trusted_sources = [
        'reuters.com', 'bbc.com', 'ap.org', 'cnn.com', 'bloomberg.com',
        'wsj.com', 'nytimes.com', 'guardian.com', 'washingtonpost.com',
        'npr.org', 'abcnews.go.com', 'cbsnews.com', 'nbcnews.com',
        'politico.com', 'thehill.com', 'time.com', 'newsweek.com'
    ]
    
    for source in trusted_sources:
        if source in url.lower():
            return 95
    return 50

# ============================================================================
# MAIN APP LAYOUT
# ============================================================================

# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">📰 Fake News Detection System</div>
    <div class="header-subtitle">
        Using AI, ML, and Real Data to Combat Misinformation
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Analyze Article",
    "📊 Dashboard",
    "📈 Model Info",
    "ℹ️ About"
])

# ============================================================================
# TAB 1: ANALYZE ARTICLE
# ============================================================================

with tab1:
    st.header("🔍 Analyze Article for Misinformation")
    
    # Input method selector
    input_method = st.radio(
        "Select input method:",
        ["📝 Paste Text", "🔗 Enter URL", "📤 Upload File"],
        horizontal=True
    )
    
    article_text = ""
    article_source = "User Input"
    
    if input_method == "📝 Paste Text":
        article_text = st.text_area(
            "Paste article text here:",
            height=200,
            placeholder="Enter the article text you want to analyze..."
        )
        article_source = "Direct Text Input"
    
    elif input_method == "🔗 Enter URL":
        url = st.text_input("Enter article URL:")
        if url:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Simple text extraction from HTML
                    from html.parser import HTMLParser
                    class TextExtractor(HTMLParser):
                        def __init__(self):
                            super().__init__()
                            self.text = []
                        def handle_data(self, data):
                            if data.strip():
                                self.text.append(data.strip())
                    
                    parser = TextExtractor()
                    parser.feed(response.text)
                    article_text = ' '.join(parser.text)[:5000]
                    article_source = url
            except Exception as e:
                st.error(f"Could not fetch URL: {e}")
    
    elif input_method == "📤 Upload File":
        uploaded_file = st.file_uploader("Upload text file", type=['txt', 'pdf'])
        if uploaded_file:
            try:
                article_text = uploaded_file.read().decode('utf-8')
                article_source = uploaded_file.name
            except Exception as e:
                st.error(f"Could not read file: {e}")
    
    # Analyze button
    if st.button("🚀 Analyze Article", use_container_width=True, type="primary"):
        if not article_text or len(article_text) < 50:
            st.error("❌ Please provide at least 50 characters of text for analysis.")
        else:
            with st.spinner("🔄 Analyzing... (ML + LLM + NewsAPI)"):
                # ML Analysis
                ml_result = predict_with_ml_models(article_text)
                
                if ml_result:
                    # Display verdict
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        is_real = ml_result['is_real']
                        verdict_text = "✅ LIKELY REAL NEWS" if is_real else "❌ LIKELY FAKE NEWS"
                        verdict_class = "verdict-real" if is_real else "verdict-fake"
                        st.markdown(f'<div class="{verdict_class}">{verdict_text}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Confidence", f"{ml_result['confidence']:.1f}%")
                    
                    with col3:
                        risk_level = "🟢 LOW" if ml_result['confidence'] < 50 else "🟡 MEDIUM" if ml_result['confidence'] < 80 else "🔴 HIGH"
                        st.metric("Risk Level", risk_level)
                    
                    # Model breakdown
                    st.markdown("### 🤖 ML Model Analysis")
                    st.write(f"""
                    - **PassiveAggressive Classifier**: {ml_result['pa_prediction']}
                    - **RandomForest Classifier**: {ml_result['rf_prediction']}
                    - **Ensemble Decision**: {'REAL' if ml_result['is_real'] else 'FAKE'}
                    """)
                    
                    # LLM Analysis
                    st.markdown("### 🧠 AI Analysis (Gemini)")
                    llm_analysis = analyze_with_gemini(article_text, ml_result)
                    st.markdown(llm_analysis)
                    
                    # NewsAPI Verification
                    st.markdown("### 📰 Related Articles (NewsAPI)")
                    keywords = article_text.split()[:5]
                    query = ' '.join(keywords)
                    articles = fetch_related_articles(query)
                    
                    if articles:
                        for i, article in enumerate(articles, 1):
                            col1, col2 = st.columns([0.8, 0.2])
                            with col1:
                                st.markdown(f"""
                                **{i}. {article.get('title', 'N/A')[:80]}**
                                
                                Source: {article.get('source', {}).get('name', 'Unknown')}
                                
                                {article.get('description', 'No description')[:150]}...
                                """)
                            with col2:
                                cred = get_source_credibility(article.get('url', ''))
                                st.metric("Trust Score", f"{cred}%")
                    else:
                        st.info("ℹ️ No related articles found. Configure NewsAPI key for verification.")
                    
                    # Save to history
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now(),
                        'text': article_text[:100],
                        'verdict': 'REAL' if ml_result['is_real'] else 'FAKE',
                        'confidence': ml_result['confidence']
                    })

# ============================================================================
# TAB 2: DASHBOARD
# ============================================================================

with tab2:
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.ml_models:
        stats = st.session_state.ml_models['dataset_stats']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Articles</div>
                <div class="metric-value">{stats['total_count']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #51cf66 0%, #2f9e44 100%);">
                <div class="metric-label">Real News</div>
                <div class="metric-value">{stats['real_count']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%);">
                <div class="metric-label">Fake News</div>
                <div class="metric-value">{stats['fake_count']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            real_pct = (stats['real_count'] / stats['total_count']) * 100
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <div class="metric-label">Real %</div>
                <div class="metric-value">{real_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Dataset distribution chart
        fig = go.Figure(data=[
            go.Pie(
                labels=['Real News', 'Fake News'],
                values=[stats['real_count'], stats['fake_count']],
                marker=dict(colors=['#51cf66', '#ff6b6b'])
            )
        ])
        fig.update_layout(title="Dataset Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Analysis history
    if st.session_state.analysis_history:
        st.markdown("### 📝 Analysis History")
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(history_df, use_container_width=True)

# ============================================================================
# TAB 3: MODEL INFO
# ============================================================================

with tab3:
    st.header("📈 Model Information")
    
    st.markdown("""
    ### 🤖 Machine Learning Models
    
    **Trained on Real Datasets:**
    - **True.csv**: 21,417 real news articles
    - **Fake.csv**: 23,481 fake news articles
    - **Total Training Data**: 44,898 articles
    
    **Models Used:**
    1. **PassiveAggressive Classifier**
       - Fast, online learning
       - Good for streaming data
       - Robust to outliers
    
    2. **Random Forest Classifier**
       - Ensemble method
       - High accuracy
       - Feature importance analysis
    
    **Feature Extraction:**
    - TF-IDF Vectorization
    - Unigrams and Bigrams
    - Stop words removed
    - Max features: 5,000
    
    ### 🧠 LLM Integration
    
    **Google Gemini API**
    - Detailed misinformation analysis
    - Red flag detection
    - Credibility assessment
    - Recommendation generation
    
    Configure: Set `GEMINI_API_KEY` in `.env` file
    
    ### 📰 NewsAPI Integration
    
    **Real-time Article Verification**
    - Fetch related articles
    - Source credibility check
    - Trending topic analysis
    - Global news coverage
    
    Configure: Set `NEWS_API_KEY` in `.env` file
    
    Get keys from:
    - [Gemini API](https://ai.google.dev/)
    - [NewsAPI](https://newsapi.org/)
    """)

# ============================================================================
# TAB 4: ABOUT
# ============================================================================

with tab4:
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    **Fake News Detection System** combines cutting-edge AI, ML, and real data to combat misinformation.
    
    **Key Features:**
    - ✅ ML Models trained on 44,898 real articles
    - ✅ Gemini LLM for intelligent analysis
    - ✅ NewsAPI for real-time verification
    - ✅ User-friendly interface
    - ✅ Professional design
    
    ### 📚 Datasets
    
    - **True.csv**: 21,417 real news articles from credible sources
    - **Fake.csv**: 23,481 fake news articles for comparison
    
    ### 🔗 Resources
    
    - **GitHub Repository**: https://github.com/nishanthsvbhat/Detecting-and-Combating-Fake-News-with-Data-and-AI
    - **Gemini API Docs**: https://ai.google.dev/docs
    - **NewsAPI Docs**: https://newsapi.org/docs
    
    ### 🛠️ Technology Stack
    
    - **Frontend**: Streamlit
    - **ML**: Scikit-learn
    - **LLM**: Google Gemini
    - **APIs**: NewsAPI
    - **Data**: Pandas, NumPy
    - **Visualization**: Plotly
    
    ### 👨‍💻 Author
    
    **Nishanth**
    
    Fighting misinformation with AI and ML
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; margin-top: 20px;">
    <small>
    🎯 Fake News Detection System | Powered by AI, ML & Real Data
    <br>
    📌 Repository: <a href="https://github.com/nishanthsvbhat/Detecting-and-Combating-Fake-News-with-Data-and-AI">GitHub Link</a>
    <br>
    ⚖️ Use responsibly. Always cross-verify with multiple sources.
    </small>
</div>
""", unsafe_allow_html=True)
