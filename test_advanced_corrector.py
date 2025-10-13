# test_advanced_corrector.py
# -*- coding: utf-8 -*-
"""
Test script for advanced corrector
"""

import os
import sys
import json
from advanced_corrector import VietnamesePreprocessor, MultiDetector, AdvancedCorrector

def test_preprocessor():
    """Test preprocessing pipeline"""
    print("=" * 60)
    print("TEST 1: PREPROCESSOR")
    print("=" * 60)
    
    preprocessor = VietnamesePreprocessor(use_word_segmenter=False)
    
    test_cases = [
        "Xin chào! Tôi là một sinh viên.",
        "Email: test@example.com và website: https://example.com",
        "Câu này có lỗi chính tả: tôii đangg họcc tiếng Việt.",
        "Unicode test: café, naïve, Việt Nam",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n[Case {i}] Input: {text}")
        result = preprocessor.preprocess(text, protect_patterns=True)
        
        print(f"  Normalized: {result['normalized']}")
        print(f"  Sentences: {result['sentences']}")
        print(f"  Tokens: {result['tokens']}")
        if result['protected_map']:
            print(f"  Protected: {result['protected_map']}")

def test_detector():
    """Test multi-detector"""
    print("\n" + "=" * 60)
    print("TEST 2: MULTI-DETECTOR")
    print("=" * 60)
    
    # Check if detector model exists
    det_dir = os.environ.get("DET_DIR", "outputs/detector")
    if not os.path.exists(det_dir):
        print(f"\n[Warning] Detector model not found at: {det_dir}")
        print("Skipping detector test. Train detector first with:")
        print("  python vi_spell_pipeline_plus.py --do_train_detector")
        return
    
    # Check if lexicon exists
    lexicon_path = "data/vi_lexicon.txt"
    if not os.path.exists(lexicon_path):
        print(f"\n[Warning] Lexicon not found at: {lexicon_path}")
        print("Creating sample lexicon...")
        os.system("python prepare_data.py --create_sample")
    
    detector = MultiDetector(
        token_classifier_dir=det_dir,
        lexicon_path=lexicon_path if os.path.exists(lexicon_path) else None,
        weight_oov=0.4,
        weight_mlm=0.0,  # Disabled (slow)
        weight_classifier=0.6,
    )
    
    test_cases = [
        ["tôi", "đang", "học", "tiếng", "việt"],  # Correct
        ["tôii", "đangg", "họcc", "tiếng", "việt"],  # Errors: tôii, đangg, họcc
        ["xin", "chào", "bạn", "nhé"],  # Correct
        ["xinchào", "bạn", "nhé"],  # Error: xinchào (should be "xin chào")
    ]
    
    for i, tokens in enumerate(test_cases, 1):
        print(f"\n[Case {i}] Tokens: {tokens}")
        
        # Test individual detectors
        oov_scores = detector.detect_oov(tokens)
        clf_scores = detector.detect_token_classifier(tokens)
        
        print(f"  OOV scores:        {[f'{s:.2f}' for s in oov_scores]}")
        print(f"  Classifier scores: {[f'{s:.2f}' for s in clf_scores]}")
        
        # Test ensemble
        detections = detector.detect(tokens, threshold=0.5, use_mlm=False)
        
        if detections:
            print(f"  Detected errors:")
            for det in detections:
                print(f"    - Position {det.position}: '{det.token}' (conf={det.confidence:.2f})")
                print(f"      Scores: {det.detector_scores}")
        else:
            print(f"  No errors detected")

def test_advanced_corrector():
    """Test full advanced corrector pipeline"""
    print("\n" + "=" * 60)
    print("TEST 3: ADVANCED CORRECTOR (Full Pipeline)")
    print("=" * 60)
    
    det_dir = os.environ.get("DET_DIR", "outputs/detector")
    lexicon_path = "data/vi_lexicon.txt"
    
    if not os.path.exists(det_dir):
        print(f"\n[Warning] Detector model not found at: {det_dir}")
        print("Skipping test. Train detector first.")
        return
    
    if not os.path.exists(lexicon_path):
        print(f"\n[Info] Creating sample lexicon...")
        os.system("python prepare_data.py --create_sample")
    
    corrector = AdvancedCorrector(
        detector_dir=det_dir,
        lexicon_path=lexicon_path if os.path.exists(lexicon_path) else None,
        use_word_segmenter=False,
    )
    
    test_cases = [
        "Xin chào! Tôi đang học tiếng Việt.",
        "Tôii đangg họcc tiếng Việt.",
        "Email: test@example.com và website: https://example.com",
        "Hôm nay trời đẹpp quá!",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n[Case {i}] Input: {text}")
        
        result = corrector.correct(
            text,
            detection_threshold=0.5,
            protect_patterns=True,
        )
        
        print(f"  Preprocessed tokens: {result['preprocessed']['tokens']}")
        
        if result['detections']:
            print(f"  Detections:")
            for det in result['detections']:
                print(f"    - Position {det['position']}: '{det['token']}' (conf={det['confidence']:.2f})")
                print(f"      Detector scores: {det['detector_scores']}")
        else:
            print(f"  No errors detected")
        
        if result['corrections']:
            print(f"  Corrections:")
            for corr in result['corrections']:
                print(f"    - Position {corr['position']}: '{corr['original']}' → '{corr['correction']}'")
        else:
            print(f"  No corrections applied")
        
        print(f"  Final: {result['final']}")

def test_api_integration():
    """Test API integration (requires running server)"""
    print("\n" + "=" * 60)
    print("TEST 4: API INTEGRATION")
    print("=" * 60)
    
    try:
        import requests
    except ImportError:
        print("[Warning] requests library not found. Skipping API test.")
        print("Install with: pip install requests")
        return
    
    api_url = "http://localhost:8000"
    
    # Check if server is running
    try:
        response = requests.get(f"{api_url}/health", timeout=2)
        print(f"\n[Health Check] {response.json()}")
    except Exception as e:
        print(f"\n[Warning] Server not running at {api_url}")
        print(f"Error: {e}")
        print("\nStart server with: uvicorn app:app --host 0.0.0.0 --port 8000")
        return
    
    # Test /correct_v2 endpoint
    test_cases = [
        {
            "text": "Tôii đangg họcc tiếng Việt.",
            "detection_threshold": 0.5,
            "use_oov": True,
            "use_mlm": False,
            "use_classifier": True,
        },
        {
            "text": "Email: test@example.com và website: https://example.com",
            "detection_threshold": 0.5,
            "protect_patterns": True,
        },
    ]
    
    for i, payload in enumerate(test_cases, 1):
        print(f"\n[API Case {i}] Request: {payload['text']}")
        
        try:
            response = requests.post(
                f"{api_url}/correct_v2",
                json=payload,
                timeout=30,
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  Status: OK")
                print(f"  Detections: {len(result['detections'])}")
                print(f"  Corrections: {len(result['corrections'])}")
                print(f"  Final: {result['final']}")
                
                if result['detections']:
                    print(f"  Detection details:")
                    for det in result['detections']:
                        print(f"    - {det['token']} (conf={det['confidence']:.2f})")
            else:
                print(f"  Status: ERROR {response.status_code}")
                print(f"  Response: {response.text}")
        
        except Exception as e:
            print(f"  Error: {e}")

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ADVANCED CORRECTOR TEST SUITE")
    print("=" * 60)
    
    # Test 1: Preprocessor (always works)
    test_preprocessor()
    
    # Test 2: Detector (requires model)
    test_detector()
    
    # Test 3: Full pipeline (requires model)
    test_advanced_corrector()
    
    # Test 4: API (requires running server)
    test_api_integration()
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETED")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Create lexicon: python prepare_data.py --create_sample")
    print("2. Train detector: python vi_spell_pipeline_plus.py --do_train_detector")
    print("3. Start server: uvicorn app:app --host 0.0.0.0 --port 8000")
    print("4. Test API: python test_advanced_corrector.py")

if __name__ == "__main__":
    main()

