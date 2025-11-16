"""
Fake News Detector - Flask Web Application
Simple and reliable deployment
"""

from flask import Flask, render_template_string, request, jsonify
import pickle
import json
import os

app = Flask(__name__)

# Load model
print("[*] Loading model...")
with open('model_ultra.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer_ultra.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('metadata_ultra.json', 'r') as f:
    metadata = json.load(f)

print("[+] Model loaded successfully")

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }
        .header h1 { font-size: 48px; margin-bottom: 10px; }
        .header p { font-size: 18px; opacity: 0.9; }
        .content { padding: 40px; }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 10px;
            font-weight: 600;
            color: #333;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            min-height: 200px;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .button-group {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
        }
        button {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-analyze {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-analyze:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3); }
        .btn-clear {
            background: #f0f0f0;
            color: #333;
        }
        .btn-clear:hover { background: #e0e0e0; }
        .results {
            display: none;
            background: #f8f9fa;
            padding: 30px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .results.show { display: block; }
        .verdict {
            text-align: center;
            margin-bottom: 30px;
        }
        .verdict-label {
            font-size: 48px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .verdict-real { color: #10b981; }
        .verdict-fake { color: #ef4444; }
        .confidence {
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
        }
        .metric-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .metric {
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .metric-label { font-size: 14px; color: #999; font-weight: 600; }
        .metric-value { font-size: 24px; font-weight: bold; margin-top: 10px; }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease;
        }
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #999;
            font-size: 12px;
        }
        .loading {
            text-align: center;
            color: #667eea;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Fake News Detector</h1>
            <p>AI-Powered Misinformation Detection</p>
        </div>
        
        <div class="content">
            <div class="form-group">
                <label for="article">📝 Enter Article Text:</label>
                <textarea id="article" placeholder="Paste your news article here..."></textarea>
            </div>
            
            <div class="button-group">
                <button class="btn-analyze" onclick="analyzeArticle()">🔍 Analyze Article</button>
                <button class="btn-clear" onclick="clearText()">🔄 Clear</button>
            </div>
            
            <div id="loading" class="loading" style="display:none;">
                ⏳ Analyzing...
            </div>
            
            <div id="results" class="results">
                <div class="verdict">
                    <div id="verdict-label" class="verdict-label"></div>
                    <div>Confidence: <span id="confidence" class="confidence">0%</span></div>
                </div>
                
                <div class="metric-row">
                    <div class="metric">
                        <div class="metric-label">Real News Probability</div>
                        <div class="progress-bar">
                            <div id="prob-real" class="progress-fill" style="width: 0%"></div>
                        </div>
                        <div class="metric-value" id="prob-real-text">0%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Fake News Probability</div>
                        <div class="progress-bar">
                            <div id="prob-fake" class="progress-fill" style="width: 0%"></div>
                        </div>
                        <div class="metric-value" id="prob-fake-text">0%</div>
                    </div>
                </div>
                
                <div class="metric-row" style="margin-top: 20px;">
                    <div class="metric">
                        <div class="metric-label">Model Accuracy</div>
                        <div class="metric-value">99.23%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Algorithm</div>
                        <div class="metric-value">LogisticRegression</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            🔍 Fake News Detector | Accuracy: 99.23% | Model: LogisticRegression + TF-IDF | Speed: <10ms
        </div>
    </div>
    
    <script>
        function analyzeArticle() {
            const text = document.getElementById('article').value.trim();
            if (!text) {
                alert('Please enter article text');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').classList.remove('show');
            
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                
                const verdict = data.prediction === 1 ? '🟢 REAL NEWS' : '🔴 FAKE NEWS';
                const verdictClass = data.prediction === 1 ? 'verdict-real' : 'verdict-fake';
                
                document.getElementById('verdict-label').textContent = verdict;
                document.getElementById('verdict-label').className = 'verdict-label ' + verdictClass;
                document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
                
                const probReal = (data.probabilities[1] * 100).toFixed(1);
                const probFake = (data.probabilities[0] * 100).toFixed(1);
                
                document.getElementById('prob-real').style.width = probReal + '%';
                document.getElementById('prob-real-text').textContent = probReal + '%';
                document.getElementById('prob-fake').style.width = probFake + '%';
                document.getElementById('prob-fake-text').textContent = probFake + '%';
                
                document.getElementById('results').classList.add('show');
            })
            .catch(e => {
                document.getElementById('loading').style.display = 'none';
                alert('Error: ' + e);
            });
        }
        
        function clearText() {
            document.getElementById('article').value = '';
            document.getElementById('results').classList.remove('show');
        }
        
        document.getElementById('article').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') analyzeArticle();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        article_text = data.get('text', '')
        
        if not article_text.strip():
            return jsonify({'error': 'Empty text'}), 400
        
        # Vectorize and predict
        X = vectorizer.transform([article_text])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = float(max(probabilities))
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': confidence,
            'probabilities': [float(p) for p in probabilities]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("FAKE NEWS DETECTOR - Flask Web Application")
    print("=" * 80)
    print("\n[*] Starting server...")
    print("[+] Open browser: http://localhost:5000")
    print("\n[*] Press Ctrl+C to stop\n")
    app.run(debug=False, host='127.0.0.1', port=5000)
