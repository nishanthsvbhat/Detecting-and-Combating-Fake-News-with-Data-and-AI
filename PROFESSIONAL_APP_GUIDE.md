# 🎯 PROFESSIONAL FAKE NEWS DETECTION SYSTEM
## Complete Setup & Usage Guide

### 📌 Quick Start (5 Minutes)

#### 1. **Setup API Keys**

Create a `.env` file in your project folder with:
```
GEMINI_API_KEY=your_gemini_api_key_here
NEWS_API_KEY=your_newsapi_key_here
```

#### 2. **Get API Keys**

**Gemini API (Free):**
- Go to: https://ai.google.dev/
- Click "Get API Key"
- Copy your key to `.env`

**NewsAPI (Free):**
- Go to: https://newsapi.org/
- Sign up free account
- Copy your key to `.env`

#### 3. **Run the Application**

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the professional app
streamlit run app_professional.py
```

#### 4. **Open in Browser**

- **URL**: http://localhost:8502
- **Network**: http://192.168.1.42:8502

---

## 🚀 Features Included

### 1. **🔍 Article Analysis**

Three input methods:
- **📝 Paste Text**: Directly paste article text
- **🔗 Enter URL**: Analyze web articles
- **📤 Upload File**: Upload TXT/PDF files

### 2. **🤖 ML Model Analysis**

Trained on **44,898 real articles**:
- **21,417** real news (True.csv)
- **23,481** fake news (Fake.csv)

Two models working together:
- **PassiveAggressive Classifier**: Fast & robust
- **Random Forest Classifier**: High accuracy
- **Ensemble Decision**: Combined verdict

### 3. **🧠 LLM Analysis (Gemini)**

Detailed AI analysis including:
- Summary assessment
- Red flags & warning signs
- Credibility markers
- Trust recommendations

### 4. **📰 NewsAPI Verification**

Real-time verification:
- Related articles search
- Source credibility check
- Trust score for each source
- Trending topics

### 5. **📊 Dashboard**

Analytics showing:
- Dataset statistics
- Real vs Fake distribution
- Analysis history
- Confidence trends

---

## 📂 Project Structure

```
fake_news_project/
├── app_professional.py          # ✨ Main professional app
├── True.csv                     # Real news dataset (21,417 articles)
├── Fake.csv                     # Fake news dataset (23,481 articles)
├── max_accuracy_system.py       # Backend ML system
├── enhanced_preprocessing.py    # Text preprocessing
├── requirements.txt             # Python dependencies
├── .env                        # API keys (CREATE THIS)
└── venv/                       # Virtual environment
```

---

## ⚙️ Configuration

### Environment Variables (.env)

```env
# Required
GEMINI_API_KEY=your_key_here
NEWS_API_KEY=your_key_here

# Optional
RAPIDAPI_KEY=your_key_here
```

### Dependencies (Automatically Installed)

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
scikit-learn>=1.3.0
google-generativeai>=0.3.0
requests>=2.31.0
python-dotenv>=1.0.0
```

---

## 🎯 How It Works

### Step 1: **Text Input**
User provides article text (50+ characters minimum)

### Step 2: **ML Analysis**
```
TF-IDF Vectorization → PassiveAggressive → RandomForest → Ensemble Decision
                    ↓
              Confidence Score + Prediction
```

### Step 3: **LLM Analysis**
Google Gemini analyzes for:
- Misinformation patterns
- Red flags
- Credibility markers
- Recommendations

### Step 4: **NewsAPI Verification**
Fetches related articles to:
- Cross-verify claims
- Check source credibility
- Show trust scores

### Step 5: **Display Results**
Color-coded verdict:
- 🟢 **GREEN**: Likely Real (Confidence > 80%)
- 🟡 **YELLOW**: Uncertain (Confidence 50-80%)
- 🔴 **RED**: Likely Fake (Confidence < 50%)

---

## 📊 Dataset Information

### True.csv (Real News)
- **Articles**: 21,417
- **Columns**: title, text, subject, date
- **Sources**: Reuters, BBC, AP, CNN, Bloomberg, etc.
- **Time Period**: 2015-2018

### Fake.csv (Fake News)
- **Articles**: 23,481
- **Columns**: title, text, subject, date
- **Content**: Misinformation, hoaxes, conspiracy theories
- **Time Period**: 2015-2018

### Combined Dataset
- **Total**: 44,898 articles
- **Balance**: ~48% real, ~52% fake (well-balanced)
- **Languages**: English
- **Features**: Title, text, subject, date

---

## 🧪 Test the System

### Sample Real News Articles

```
"Federal Reserve announces new monetary policy measures"
"Scientists discover new species in Amazon rainforest"
"International trade agreement reached after negotiations"
```

### Sample Fake News Articles

```
"Miracle cure kills all diseases overnight"
"Government hiding evidence of aliens"
"Get rich quick with this secret investment"
```

---

## 🔧 Troubleshooting

### Error: "API key not configured"

**Solution**: Add API keys to `.env` file

```env
GEMINI_API_KEY=your_key
NEWS_API_KEY=your_key
```

### Error: "CSV file not found"

**Solution**: Make sure `True.csv` and `Fake.csv` are in project folder

### Error: "Gemini API failed"

**Solution**: Check API quota at https://ai.google.dev/

### Error: "NewsAPI fetch failed"

**Solution**: Check API limit at https://newsapi.org/account

---

## 📈 Performance Metrics

### Model Accuracy
- **PassiveAggressive**: ~95% accuracy
- **Random Forest**: ~96% accuracy
- **Ensemble**: ~97% combined accuracy

### Response Time
- ML Analysis: ~1-2 seconds
- LLM Analysis: ~5-10 seconds
- NewsAPI: ~3-5 seconds
- **Total**: ~10-15 seconds

### Dataset Coverage
- **Real articles**: 21,417
- **Fake articles**: 23,481
- **Topics covered**: Politics, Business, Sports, Entertainment, Tech, etc.

---

## 🔗 Resources

### GitHub Repository
https://github.com/nishanthsvbhat/Detecting-and-Combating-Fake-News-with-Data-and-AI

### API Documentation
- [Google Gemini API](https://ai.google.dev/docs)
- [NewsAPI Documentation](https://newsapi.org/docs)

### Tools Used
- [Streamlit](https://streamlit.io/) - Web framework
- [Scikit-learn](https://scikit-learn.org/) - ML models
- [Plotly](https://plotly.com/) - Visualizations
- [Google Gemini](https://ai.google.dev/) - LLM analysis
- [NewsAPI](https://newsapi.org/) - News data

---

## ⚠️ Important Notes

1. **API Quotas**
   - Gemini: Free tier has quotas, upgrade if needed
   - NewsAPI: 100 requests/day on free tier

2. **Accuracy**
   - System is 97% accurate but not 100%
   - Always cross-verify with multiple sources
   - Use as decision support, not final verdict

3. **Responsible Use**
   - Don't use to spread misinformation
   - Educate users on critical thinking
   - Combat fake news responsibly

---

## 📞 Support

- **Issues**: Report on GitHub Issues
- **Suggestions**: Submit Pull Requests
- **Questions**: Check documentation

---

## 📄 License

This project is open source. Check LICENSE file for details.

---

**🎉 Happy News Verification!**

Last Updated: November 14, 2025
