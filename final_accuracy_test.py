"""
Final Accuracy Test - Comprehensive Test Suite for Academic Submission
Tests all critical edge cases to ensure maximum accuracy
"""

import subprocess
import sys
import time

# CRITICAL TEST CASES - These MUST pass for academic submission
test_cases = [
    {
        "category": "FALSE POLITICAL CLAIMS",
        "tests": [
            {
                "text": "Rahul Gandhi is the new PM of India",
                "expected": "FALSE",
                "reason": "Factually incorrect political claim"
            },
            {
                "text": "Joe Biden resigned as US President yesterday",
                "expected": "FALSE", 
                "reason": "False current events claim"
            },
            {
                "text": "Xi Jinping announced China's surrender to Taiwan",
                "expected": "FALSE",
                "reason": "False geopolitical claim"
            }
        ]
    },
    {
        "category": "CONFLICT SPECULATION",
        "tests": [
            {
                "text": "War between China and Taiwan has started with naval battles",
                "expected": "FALSE",
                "reason": "Unverified conflict claim"
            },
            {
                "text": "Russia launches new offensive against Ukraine cities",
                "expected": "FALSE",
                "reason": "Conflict speculation without verification"
            },
            {
                "text": "North Korea invades South Korea with ground forces",
                "expected": "FALSE", 
                "reason": "Unverified military action"
            }
        ]
    },
    {
        "category": "CLEAR MISINFORMATION",
        "tests": [
            {
                "text": "COVID-19 vaccines contain 5G microchips for mind control",
                "expected": "FALSE",
                "reason": "Medical misinformation"
            },
            {
                "text": "Earth is flat and NASA hides the truth from everyone",
                "expected": "FALSE",
                "reason": "Science denial conspiracy"
            },
            {
                "text": "Drinking bleach cures cancer according to new study",
                "expected": "FALSE",
                "reason": "Dangerous health misinformation"
            }
        ]
    },
    {
        "category": "LEGITIMATE NEWS",
        "tests": [
            {
                "text": "Stock markets showed mixed results in today's trading session",
                "expected": "TRUE",
                "reason": "General financial news"
            },
            {
                "text": "New smartphone technology improves battery life significantly",
                "expected": "TRUE",
                "reason": "Technology advancement news"
            },
            {
                "text": "Climate change continues to affect global weather patterns",
                "expected": "TRUE",
                "reason": "Established scientific consensus"
            }
        ]
    },
    {
        "category": "POLITICAL NEWS (LEGITIMATE)",
        "tests": [
            {
                "text": "Political parties prepare campaign strategies for upcoming elections",
                "expected": "TRUE",
                "reason": "General political process news"
            },
            {
                "text": "Government announces new policy for economic development",
                "expected": "TRUE",
                "reason": "Official policy announcement"
            }
        ]
    }
]

def run_test(text):
    """Run the misinformation detection system"""
    try:
        cmd = [sys.executable, "max_accuracy_system.py", "--text", text]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".", timeout=30)
        
        if result.returncode != 0:
            return {"error": f"Command failed: {result.stderr}"}
        
        output = result.stdout.strip()
        lines = output.split('\n')
        
        # Parse the output
        verdict = None
        confidence = None
        reasoning = []
        
        for line in lines:
            if 'Final Verdict:' in line:
                verdict = line.split('Final Verdict:')[1].strip()
            elif 'Confidence:' in line:
                confidence = line.split('Confidence:')[1].strip().replace('%', '')
            elif 'Reasoning:' in line:
                reasoning.append(line.split('Reasoning:')[1].strip())
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "full_output": output
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    """Run comprehensive accuracy test"""
    print("🚀 FINAL ACCURACY TEST - Academic Submission Ready")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for category_data in test_cases:
        category = category_data["category"]
        print(f"\n📂 Testing Category: {category}")
        print("-" * 40)
        
        for i, test in enumerate(category_data["tests"], 1):
            total_tests += 1
            print(f"\n🧪 Test {i}: {test['reason']}")
            print(f"Input: \"{test['text'][:60]}{'...' if len(test['text']) > 60 else ''}\"")
            print(f"Expected: {test['expected']}")
            
            # Run the test
            result = run_test(test['text'])
            
            if "error" in result:
                print(f"❌ ERROR: {result['error']}")
                failed_tests.append({
                    "category": category,
                    "test": test,
                    "error": result['error']
                })
                continue
            
            actual_verdict = result.get('verdict', 'UNKNOWN')
            confidence = result.get('confidence', 'N/A')
            
            # Check if test passed
            if actual_verdict == test['expected']:
                print(f"✅ PASSED - Verdict: {actual_verdict}, Confidence: {confidence}%")
                passed_tests += 1
            else:
                print(f"❌ FAILED - Expected: {test['expected']}, Got: {actual_verdict}, Confidence: {confidence}%")
                failed_tests.append({
                    "category": category,
                    "test": test,
                    "expected": test['expected'],
                    "actual": actual_verdict,
                    "confidence": confidence,
                    "reasoning": result.get('reasoning', [])
                })
            
            time.sleep(1)  # Brief pause between tests
    
    # Final Results Summary
    print("\n" + "=" * 60)
    print("🎯 FINAL ACCURACY RESULTS")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Accuracy: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests:
        print("\n❌ FAILED TESTS REQUIRING IMMEDIATE ATTENTION:")
        print("-" * 50)
        for i, failure in enumerate(failed_tests, 1):
            print(f"\n{i}. Category: {failure['category']}")
            print(f"   Test: {failure['test']['reason']}")
            print(f"   Input: {failure['test']['text'][:80]}{'...' if len(failure['test']['text']) > 80 else ''}")
            if 'expected' in failure:
                print(f"   Expected: {failure['expected']} | Actual: {failure['actual']} | Confidence: {failure['confidence']}%")
            if 'error' in failure:
                print(f"   Error: {failure['error']}")
    else:
        print("\n🎉 ALL TESTS PASSED! System ready for academic submission!")
        print("✅ Maximum accuracy achieved with zero critical errors!")
    
    print("\n" + "=" * 60)
    return len(failed_tests) == 0

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n⚠️  CRITICAL: Fix failed tests before submission!")
        sys.exit(1)
    else:
        print("✅ System verification complete - Ready for submission!")
        sys.exit(0)