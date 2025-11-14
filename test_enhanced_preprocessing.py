#!/usr/bin/env python3
"""
Quick test to verify enhanced preprocessing module
Tests all major functions from enhanced_preprocessing.py
"""

import sys
sys.path.insert(0, '.')

from enhanced_preprocessing import EnhancedPreprocessor, preprocess_to_string, extract_key_features

def test_enhanced_preprocessing():
    """Test enhanced preprocessing functions"""
    
    print("=" * 70)
    print("ENHANCED PREPROCESSING MODULE TEST")
    print("=" * 70)
    
    # Initialize preprocessor
    preprocessor = EnhancedPreprocessor()
    
    # Test text with various elements
    test_text = """
    Check out this AMAZING opportunity! ✨🚀
    Visit https://example.com/scam or email me@fake.com
    Breaking: @ElonMusk tweets "Get RICH QUICK with #CryptoGold!"
    <html>This</html> miracle cure CURES EVERYTHING!
    Special offer: LIMITED TIME!!! Won't you join us???
    """
    
    print("\n1. ORIGINAL TEXT:")
    print(f"   {test_text[:100]}...")
    
    print("\n2. BASIC CLEANING:")
    cleaned = preprocessor.clean_text(test_text, aggressive=False)
    print(f"   {cleaned[:100]}...")
    
    print("\n3. AGGRESSIVE CLEANING:")
    cleaned_agg = preprocessor.clean_text(test_text, aggressive=True)
    print(f"   {cleaned_agg[:100]}...")
    
    print("\n4. FULL PREPROCESSING (tokens):")
    tokens = preprocessor.preprocess_full(test_text, apply_stem=True, apply_lemma=False)
    print(f"   Tokens: {tokens[:10]}")
    print(f"   Total tokens: {len(tokens)}")
    
    print("\n5. PREPROCESS TO STRING (for ML):")
    processed_str = preprocessor.preprocess_to_string(test_text)
    print(f"   {processed_str[:100]}...")
    
    print("\n6. EXTRACTED KEY FEATURES:")
    features = preprocessor.extract_key_features(test_text)
    for key, value in features.items():
        print(f"   {key}: {value}")
    
    print("\n7. SPECIFIC CLEANING TESTS:")
    
    # Test URL removal
    url_text = "Check https://example.com and www.site.com today"
    print(f"   Remove URLs: '{url_text}' → '{preprocessor.remove_urls(url_text).strip()}'")
    
    # Test email removal
    email_text = "Email me@example.com or admin@site.org"
    print(f"   Remove emails: '{email_text}' → '{preprocessor.remove_emails(email_text).strip()}'")
    
    # Test emoji removal
    emoji_text = "Great! 😊 Amazing! 🚀 Wow! ✨"
    print(f"   Remove emojis: '{emoji_text}' → '{preprocessor.remove_emojis(emoji_text).strip()}'")
    
    # Test mention removal
    mention_text = "Hey @john and @jane please read this"
    print(f"   Remove mentions: '{mention_text}' → '{preprocessor.remove_mentions(mention_text).strip()}'")
    
    # Test contraction expansion
    contraction_text = "I won't can't shouldn't"
    print(f"   Expand contractions: '{contraction_text}' → '{preprocessor.expand_contractions(contraction_text).strip()}'")
    
    print("\n8. REAL FAKE NEWS EXAMPLES:")
    
    fake_example = "URGENT: Miracle cure doctors don't want you to know about! Click here for guaranteed investment returns!!! Big pharma conspiracy!"
    real_example = "Apple reports quarterly earnings showing revenue growth. Reuters confirmed official statement from health authorities."
    
    print(f"\n   FAKE NEWS EXAMPLE:")
    print(f"   Original: {fake_example[:80]}...")
    fake_processed = preprocess_to_string(fake_example)
    print(f"   Processed: {fake_processed[:80]}...")
    
    print(f"\n   REAL NEWS EXAMPLE:")
    print(f"   Original: {real_example[:80]}...")
    real_processed = preprocess_to_string(real_example)
    print(f"   Processed: {real_processed[:80]}...")
    
    print("\n" + "=" * 70)
    print("✅ ENHANCED PREPROCESSING MODULE TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    try:
        test_enhanced_preprocessing()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
