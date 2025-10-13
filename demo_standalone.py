# demo_standalone.py
# -*- coding: utf-8 -*-
"""
Standalone demo for Advanced Corrector (không cần chạy server)
"""

import os
import sys
from advanced_corrector import VietnamesePreprocessor, MultiDetector, AdvancedCorrector

def print_section(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def demo_preprocessor():
    """Demo preprocessing"""
    print_section("DEMO 1: PREPROCESSING")
    
    preprocessor = VietnamesePreprocessor(use_word_segmenter=False)
    
    test_text = "Xin chào! Email: test@example.com và website: https://example.com"
    
    print(f"\n📝 Input text:")
    print(f"   {test_text}")
    
    result = preprocessor.preprocess(test_text, protect_patterns=True)
    
    print(f"\n✅ Preprocessing results:")
    print(f"   Normalized:  {result['normalized']}")
    print(f"   Sentences:   {result['sentences']}")
    print(f"   Tokens:      {result['tokens']}")
    
    if result['protected_map']:
        print(f"\n🔒 Protected patterns:")
        for placeholder, original in result['protected_map'].items():
            print(f"   {placeholder} → {original}")

def demo_detector():
    """Demo multi-detector"""
    print_section("DEMO 2: MULTI-DETECTOR")
    
    # Check if models exist
    det_dir = os.environ.get("DET_DIR", "outputs/detector")
    lexicon_path = "data/vi_lexicon.txt"
    
    if not os.path.exists(det_dir):
        print(f"\n⚠️  Detector model not found at: {det_dir}")
        print(f"   Train detector first:")
        print(f"   python vi_spell_pipeline_plus.py --do_train_detector")
        return
    
    if not os.path.exists(lexicon_path):
        print(f"\n⚠️  Lexicon not found. Creating sample lexicon...")
        os.system("python prepare_data.py --create_sample")
    
    print(f"\n📦 Loading detector...")
    print(f"   Model: {det_dir}")
    print(f"   Lexicon: {lexicon_path}")
    
    detector = MultiDetector(
        token_classifier_dir=det_dir,
        lexicon_path=lexicon_path,
        weight_oov=0.4,
        weight_mlm=0.0,  # Disabled (slow)
        weight_classifier=0.6,
    )
    
    print(f"   ✓ Loaded (lexicon size: {len(detector.lexicon)})")
    
    # Test cases
    test_cases = [
        {
            'text': "Tôi đang học tiếng Việt",
            'expected': "No errors (correct sentence)"
        },
        {
            'text': "Tôii đangg họcc tiếng Việt",
            'expected': "Errors: tôii, đangg, họcc"
        },
        {
            'text': "Xin chào bạn nhé",
            'expected': "No errors"
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        text = case['text']
        expected = case['expected']
        
        print(f"\n📝 Test case {i}:")
        print(f"   Input:    {text}")
        print(f"   Expected: {expected}")
        
        # Tokenize
        from advanced_corrector import VietnamesePreprocessor
        preprocessor = VietnamesePreprocessor()
        tokens = preprocessor.split_syllables(text)
        
        # Detect
        detections = detector.detect(tokens, threshold=0.5, use_mlm=False)
        
        if detections:
            print(f"\n   ✓ Detected {len(detections)} error(s):")
            for det in detections:
                print(f"     • Position {det.position}: '{det.token}' (confidence={det.confidence:.2f})")
                print(f"       Scores: OOV={det.detector_scores['oov']:.2f}, "
                      f"Classifier={det.detector_scores['classifier']:.2f}")
        else:
            print(f"\n   ✓ No errors detected")

def demo_full_pipeline():
    """Demo full advanced corrector"""
    print_section("DEMO 3: FULL PIPELINE")
    
    det_dir = os.environ.get("DET_DIR", "outputs/detector")
    lexicon_path = "data/vi_lexicon.txt"
    
    if not os.path.exists(det_dir):
        print(f"\n⚠️  Detector model not found. Skipping demo.")
        return
    
    if not os.path.exists(lexicon_path):
        print(f"\n⚠️  Creating sample lexicon...")
        os.system("python prepare_data.py --create_sample")
    
    print(f"\n📦 Initializing Advanced Corrector...")
    
    corrector = AdvancedCorrector(
        detector_dir=det_dir,
        lexicon_path=lexicon_path,
        use_word_segmenter=False,
    )
    
    print(f"   ✓ Ready")
    
    test_cases = [
        "Tôii đangg họcc tiếng Việt.",
        "Email: test@example.com và website: https://example.com",
        "Hôm nay trời đẹpp quá!",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n📝 Test case {i}:")
        print(f"   Input: {text}")
        
        result = corrector.correct(
            text,
            detection_threshold=0.5,
            protect_patterns=True,
        )
        
        print(f"\n   Preprocessing:")
        print(f"     Tokens: {result['preprocessed']['tokens']}")
        
        if result['detections']:
            print(f"\n   Detection ({len(result['detections'])} error(s)):")
            for det in result['detections']:
                print(f"     • Position {det['position']}: '{det['token']}' (conf={det['confidence']:.2f})")
                scores = det['detector_scores']
                print(f"       OOV={scores['oov']:.2f}, MLM={scores['mlm']:.2f}, "
                      f"Classifier={scores['classifier']:.2f}")
        else:
            print(f"\n   Detection: No errors")
        
        if result['corrections']:
            print(f"\n   Corrections:")
            for corr in result['corrections']:
                print(f"     • Position {corr['position']}: '{corr['original']}' → '{corr['correction']}'")
        else:
            print(f"\n   Corrections: None (Generator not implemented yet - Phase 2)")
        
        print(f"\n   Final: {result['final']}")

def main():
    """Main demo"""
    print("\n" + "=" * 70)
    print("  ADVANCED VIETNAMESE SPELL CORRECTOR - STANDALONE DEMO")
    print("=" * 70)
    print("\n  Phase 1: Preprocessing + Multi-Detector")
    print("  Note: Candidate Generation & Ranking will be in Phase 2")
    
    # Demo 1: Preprocessor (always works)
    demo_preprocessor()
    
    # Demo 2: Detector (requires model)
    demo_detector()
    
    # Demo 3: Full pipeline (requires model)
    demo_full_pipeline()
    
    # Summary
    print_section("SUMMARY")
    print("\n✅ Phase 1 Features Demonstrated:")
    print("   • Unicode normalization (NFC)")
    print("   • Sentence splitting")
    print("   • Word segmentation (syllable-based)")
    print("   • Pattern protection (URLs, emails)")
    print("   • Multi-detector ensemble (OOV + Classifier)")
    print("   • Detailed detection scores")
    
    print("\n🚧 Phase 2 (Coming Soon):")
    print("   • Candidate Generation (SymSpell, Telex/VNI, keyboard)")
    print("   • Noisy-Channel Ranking (LM + P_err + freq)")
    print("   • KenLM 5-gram integration")
    
    print("\n📚 Next Steps:")
    print("   1. Read documentation: ADVANCED_CORRECTOR_README.md")
    print("   2. Read summary: PHASE1_SUMMARY.md")
    print("   3. Start API server: uvicorn app:app --port 8000")
    print("   4. Test API: curl -X POST http://localhost:8000/correct_v2 \\")
    print("                     -H 'Content-Type: application/json' \\")
    print("                     -d '{\"text\": \"Tôii đangg họcc\"}'")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

