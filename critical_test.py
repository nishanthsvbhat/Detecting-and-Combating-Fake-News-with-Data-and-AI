"""
Quick Critical Test - Verify key fixes are working
"""

import subprocess
import sys

# Critical test cases for submission
critical_tests = [
    {
        "text": "Rahul Gandhi is the new PM of India", 
        "expected": "FALSE",
        "description": "False political claim"
    },
    {
        "text": "War between China and Taiwan has started",
        "expected": "FALSE", 
        "description": "Conflict speculation"
    },
    {
        "text": "COVID vaccines contain microchips",
        "expected": "FALSE",
        "description": "Clear misinformation"
    },
    {
        "text": "Stock markets showed mixed results today",
        "expected": "TRUE",
        "description": "Legitimate news"
    }
]

def test_case(text, expected, description):
    """Test a single case"""
    print(f"\\n🧪 Testing: {description}")
    print(f"Input: \\\"{text[:50]}{'...' if len(text) > 50 else ''}\\\"")
    print(f"Expected: {expected}")
    
    try:
        cmd = [sys.executable, "max_accuracy_system.py", "--text", text]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"❌ ERROR: {result.stderr}")
            return False
        
        # Parse verdict - more robust parsing
        lines = result.stdout.split('\\n')
        verdict = None
        confidence = None
        
        for line in lines:
            if 'Final Verdict:' in line:
                verdict = line.split('Final Verdict:')[1].strip()
            elif 'Confidence:' in line:
                confidence = line.split('Confidence:')[1].strip().replace('%', '')
        
        print(f"Got verdict: '{verdict}', confidence: {confidence}%")
        
        # Normalize verdicts for comparison
        if verdict and expected:
            verdict_clean = verdict.strip().upper()
            expected_clean = expected.strip().upper()
            
            # Handle both exact and partial matches
            if verdict_clean == expected_clean:
                print(f"✅ PASSED - Exact match")
                return True
            elif expected_clean == "FALSE" and "FALSE" in verdict_clean:
                print(f"✅ PASSED - Contains expected FALSE")
                return True
            elif expected_clean == "TRUE" and ("TRUE" in verdict_clean or "LIKELY TRUE" in verdict_clean):
                print(f"✅ PASSED - Contains expected TRUE")
                return True
            else:
                print(f"❌ FAILED - Expected: {expected}, Got: {verdict}")
                return False
        else:
            print(f"❌ FAILED - Could not parse verdict from output")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run critical tests"""
    print("🚀 CRITICAL TEST - Submission Verification")
    print("=" * 50)
    
    passed = 0
    total = len(critical_tests)
    
    for test in critical_tests:
        if test_case(test["text"], test["expected"], test["description"]):
            passed += 1
    
    print("\\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("✅ ALL CRITICAL TESTS PASSED!")
        print("🎯 System ready for academic submission!")
    else:
        print("❌ CRITICAL ISSUES FOUND!")
        print("⚠️  Must fix before submission!")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)