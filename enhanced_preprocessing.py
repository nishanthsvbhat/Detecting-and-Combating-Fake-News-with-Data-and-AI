"""
Enhanced Text Preprocessing Module
Inspired by best practices from reference Fake News Detection project
Provides robust cleaning, tokenization, and feature extraction
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import numpy as np

# Download required NLTK data (silent fallback if already present)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class EnhancedPreprocessor:
    """Robust text preprocessing with multiple cleaning strategies"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Expand stop words with custom terms
        self.stop_words.update({
            'said', 'would', 'could', 'also', 'say', 'get', 'make', 'go',
            'know', 'take', 'see', 'come', 'think', 'time', 'very', 'when'
        })
        
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.html_pattern = re.compile(r'<.*?>')
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002500-\U00002BEF"  # chinese char
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+"
        )
        self.mention_pattern = re.compile(r'@\S+')
        self.hashtag_pattern = re.compile(r'#\S+')
        self.special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        
    def remove_urls(self, text: str) -> str:
        """Remove HTTP/HTTPS URLs and www links"""
        return self.url_pattern.sub(' ', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses"""
        return self.email_pattern.sub(' ', text)
    
    def remove_html(self, text: str) -> str:
        """Remove HTML tags"""
        return self.html_pattern.sub(' ', text)
    
    def remove_emojis(self, text: str) -> str:
        """Remove emoji characters"""
        return self.emoji_pattern.sub(' ', text)
    
    def remove_mentions(self, text: str) -> str:
        """Remove @mentions from social media"""
        return self.mention_pattern.sub(' ', text)
    
    def remove_hashtags(self, text: str) -> str:
        """Remove hashtags but keep words after them"""
        return self.hashtag_pattern.sub(lambda m: m.group(0)[1:], text)
    
    def expand_contractions(self, text: str) -> str:
        """Expand common contractions"""
        contractions_dict = {
            r"won't": "will not",
            r"can't": "can not",
            r"n't": " not",
            r"'re": " are",
            r"'ve": " have",
            r"'ll": " will",
            r"'d": " would",
            r"'m": " am"
        }
        
        for contraction, expansion in contractions_dict.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
        return text
    
    def remove_numbers(self, text: str, keep_structure: bool = False) -> str:
        """Remove numbers, optionally keeping word structure"""
        if keep_structure:
            return re.sub(r'\d+', 'NUM', text)
        else:
            return re.sub(r'\d+', ' ', text)
    
    def remove_special_chars(self, text: str) -> str:
        """Remove special characters but keep basic punctuation"""
        # Keep only letters, numbers, and basic spaces/punctuation
        text = self.special_chars_pattern.sub(' ', text)
        return text
    
    def lowercase(self, text: str) -> str:
        """Convert to lowercase"""
        return text.lower()
    
    def remove_extra_spaces(self, text: str) -> str:
        """Remove multiple consecutive spaces"""
        return re.sub(r'\s+', ' ', text).strip()
    
    def tokenize(self, text: str) -> list:
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: list, keep_negations: bool = True) -> list:
        """Remove stopwords, optionally keeping negations"""
        negations = {'no', 'not', 'nor', 'neither', "n't"}
        
        if keep_negations:
            return [token for token in tokens if token not in self.stop_words or token in negations]
        else:
            return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: list) -> list:
        """Apply lemmatization to tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem(self, tokens: list) -> list:
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """
        Comprehensive text cleaning pipeline
        
        Args:
            text: Input text to clean
            aggressive: If True, removes more content (numbers, special chars)
        
        Returns:
            Cleaned text
        """
        # Order matters - do structural cleaning first
        text = self.remove_html(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_mentions(text)
        text = self.remove_emojis(text)
        
        # Linguistic cleaning
        text = self.expand_contractions(text)
        text = self.lowercase(text)
        
        if aggressive:
            text = self.remove_hashtags(text)
            text = self.remove_numbers(text, keep_structure=True)
            text = self.remove_special_chars(text)
        
        text = self.remove_extra_spaces(text)
        return text
    
    def preprocess_full(self, text: str, apply_stem: bool = True, 
                       apply_lemma: bool = True, aggressive: bool = False) -> list:
        """
        Full preprocessing pipeline: cleaning → tokenization → stopword removal → lemma/stem
        
        Args:
            text: Input text
            apply_stem: Apply stemming (default: True)
            apply_lemma: Apply lemmatization (default: True)
            aggressive: Aggressive cleaning (default: False)
        
        Returns:
            List of processed tokens
        """
        # Clean text
        cleaned_text = self.clean_text(text, aggressive=aggressive)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens, keep_negations=True)
        
        # Lemmatization
        if apply_lemma:
            tokens = self.lemmatize(tokens)
        
        # Stemming (typically done after lemmatization or as alternative)
        if apply_stem and not apply_lemma:
            tokens = self.stem(tokens)
        
        # Filter empty tokens
        tokens = [t for t in tokens if t and len(t) > 1]
        
        return tokens
    
    def preprocess_to_string(self, text: str, aggressive: bool = False) -> str:
        """
        Preprocess to cleaned string (useful for TF-IDF vectorization)
        
        Args:
            text: Input text
            aggressive: Aggressive cleaning
        
        Returns:
            Cleaned and preprocessed text as string
        """
        tokens = self.preprocess_full(text, apply_stem=True, apply_lemma=False, aggressive=aggressive)
        return ' '.join(tokens)
    
    def extract_key_features(self, text: str) -> dict:
        """
        Extract linguistic and structural features from text
        
        Args:
            text: Input text
        
        Returns:
            Dictionary of extracted features
        """
        cleaned = self.clean_text(text, aggressive=False)
        tokens = self.tokenize(cleaned)
        
        features = {
            'word_count': len(tokens),
            'unique_words': len(set(tokens)),
            'avg_word_length': np.mean([len(t) for t in tokens]) if tokens else 0,
            'has_urls': bool(self.url_pattern.search(text)),
            'has_emails': bool(self.email_pattern.search(text)),
            'has_emojis': bool(self.emoji_pattern.search(text)),
            'has_mentions': bool(self.mention_pattern.search(text)),
            'has_hashtags': bool(self.hashtag_pattern.search(text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'punctuation_density': sum(1 for c in text if c in string.punctuation) / len(text) if text else 0,
        }
        
        return features


# Module-level convenience functions
preprocessor = EnhancedPreprocessor()

def clean_text(text: str, aggressive: bool = False) -> str:
    """Clean text using default preprocessor"""
    return preprocessor.clean_text(text, aggressive=aggressive)

def preprocess_full(text: str, apply_stem: bool = True, 
                   apply_lemma: bool = True, aggressive: bool = False) -> list:
    """Full preprocessing pipeline"""
    return preprocessor.preprocess_full(text, apply_stem, apply_lemma, aggressive)

def preprocess_to_string(text: str, aggressive: bool = False) -> str:
    """Preprocess to cleaned string"""
    return preprocessor.preprocess_to_string(text, aggressive=aggressive)

def extract_key_features(text: str) -> dict:
    """Extract linguistic features"""
    return preprocessor.extract_key_features(text)
