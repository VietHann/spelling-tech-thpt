# advanced_corrector.py
# -*- coding: utf-8 -*-
"""
Advanced Vietnamese Spell Correction Pipeline (Multi-tier Architecture)

TIER 1 (Realtime):
  1. Preprocessing: Unicode NFC, sentence split, word segmentation
  2. Multi-Detector: OOV + masked-LM NLL spike + token-classifier ensemble
  3. Candidate Generator: SymSpell + Telex/VNI + keyboard adjacency + split/join
  4. Noisy-Channel Ranker: LM_masked + LM_5gram + P_err + freq + edit_dist + ortho
  5. Global Search: Viterbi beam search

TIER 2 (Heavy / Fix All):
  6. GEC Seq2Seq: T5/BART with constrained decoding
  7. Global Rescoring: External LM + penalty for large changes

TIER 3 (Post-processing & UX):
  8. Post-rules: punctuation, whitespace, capitalization, protect patterns
  9. UX: underline errors, suggestions, explain, user dictionary
"""

import os
import re
import json
import unicodedata
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import math

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMaskedLM

# =========================
# 1. PREPROCESSING
# =========================

class VietnamesePreprocessor:
    """
    Preprocessing pipeline for Vietnamese text:
    - Unicode normalization (NFC)
    - Sentence splitting
    - Word segmentation (syllable-based for Vietnamese)
    """
    
    # Sentence boundary markers
    SENT_END_REGEX = re.compile(r'([.!?…]+)\s+')
    
    # Punctuation handling
    PUNCT_REGEX = re.compile(r"([,.:;!?\"""'''()\[\]{}…])")
    
    # Patterns to protect (code, URLs, emails)
    PROTECT_PATTERNS = {
        'url': re.compile(r'https?://[^\s]+'),
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'code': re.compile(r'`[^`]+`'),
        'number': re.compile(r'\b\d+([.,]\d+)*\b'),
    }
    
    def __init__(self, use_word_segmenter: bool = False):
        """
        Args:
            use_word_segmenter: If True, use underthesea/pyvi for word segmentation
                               If False, use simple syllable splitting (faster)
        """
        self.use_word_segmenter = use_word_segmenter
        self.word_segmenter = None
        
        if use_word_segmenter:
            try:
                from underthesea import word_tokenize
                self.word_segmenter = word_tokenize
            except ImportError:
                try:
                    from pyvi import ViTokenizer
                    self.word_segmenter = ViTokenizer.tokenize
                except ImportError:
                    print("[Warning] Neither underthesea nor pyvi found. Falling back to syllable splitting.")
                    self.use_word_segmenter = False
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize to Unicode NFC (canonical composition)"""
        return unicodedata.normalize('NFC', text)
    
    def protect_special_patterns(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Replace special patterns (URLs, emails, code) with placeholders
        Returns: (protected_text, placeholder_map)
        """
        placeholder_map = {}
        protected = text
        
        for pattern_name, pattern in self.PROTECT_PATTERNS.items():
            matches = pattern.findall(protected)
            for i, match in enumerate(matches):
                placeholder = f"___{pattern_name.upper()}_{i}___"
                placeholder_map[placeholder] = match
                protected = protected.replace(match, placeholder, 1)
        
        return protected, placeholder_map
    
    def restore_special_patterns(self, text: str, placeholder_map: Dict[str, str]) -> str:
        """Restore protected patterns from placeholders"""
        restored = text
        for placeholder, original in placeholder_map.items():
            restored = restored.replace(placeholder, original)
        return restored
    
    def sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting by punctuation"""
        # Split by sentence-ending punctuation
        sentences = self.SENT_END_REGEX.split(text)
        
        # Merge punctuation back with sentences
        result = []
        i = 0
        while i < len(sentences):
            sent = sentences[i].strip()
            if sent:
                # Check if next item is punctuation
                if i + 1 < len(sentences) and sentences[i + 1].strip() in '.!?…':
                    sent += sentences[i + 1]
                    i += 2
                else:
                    i += 1
                result.append(sent)
            else:
                i += 1
        
        return result if result else [text]
    
    def split_syllables(self, text: str) -> List[str]:
        """
        Split Vietnamese text into syllables (basic tokenization)
        Handles punctuation separation
        """
        # Separate punctuation
        text = self.PUNCT_REGEX.sub(r' \1 ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split() if text else []
    
    def word_segment(self, text: str) -> List[str]:
        """
        Word segmentation for Vietnamese
        Uses underthesea/pyvi if available, otherwise falls back to syllable splitting
        """
        if self.use_word_segmenter and self.word_segmenter:
            try:
                # underthesea/pyvi returns space-separated words with underscores for multi-syllable words
                segmented = self.word_segmenter(text)
                # Split by space to get tokens
                return segmented.split() if isinstance(segmented, str) else segmented
            except Exception as e:
                print(f"[Warning] Word segmenter failed: {e}. Falling back to syllable splitting.")
                return self.split_syllables(text)
        else:
            return self.split_syllables(text)
    
    def preprocess(self, text: str, protect_patterns: bool = True) -> Dict:
        """
        Full preprocessing pipeline
        
        Returns:
            {
                'original': original text,
                'normalized': Unicode normalized,
                'sentences': list of sentences,
                'tokens': list of tokens (syllables/words),
                'protected_map': placeholder map (if protect_patterns=True)
            }
        """
        # Step 1: Unicode normalization
        normalized = self.normalize_unicode(text)
        
        # Step 2: Protect special patterns
        protected_map = {}
        if protect_patterns:
            normalized, protected_map = self.protect_special_patterns(normalized)
        
        # Step 3: Sentence splitting
        sentences = self.sentence_split(normalized)
        
        # Step 4: Word segmentation
        tokens = self.word_segment(normalized)
        
        return {
            'original': text,
            'normalized': normalized,
            'sentences': sentences,
            'tokens': tokens,
            'protected_map': protected_map
        }


# =========================
# 2. MULTI-DETECTOR
# =========================

@dataclass
class DetectionResult:
    """Result from error detection"""
    position: int           # Token position
    token: str             # The token
    confidence: float      # Detection confidence [0, 1]
    detector_scores: Dict[str, float] = field(default_factory=dict)  # Individual detector scores


class MultiDetector:
    """
    Multi-strategy error detector combining:
    1. OOV (Out-of-Vocabulary) detection
    2. Masked-LM NLL spike detection
    3. Token classifier (existing model)
    """
    
    def __init__(
        self,
        token_classifier_dir: Optional[str] = None,
        masked_lm_name: str = "vinai/phobert-base",
        lexicon_path: Optional[str] = None,
        device: Optional[str] = None,
        # Ensemble weights
        weight_oov: float = 0.3,
        weight_mlm: float = 0.3,
        weight_classifier: float = 0.4,
    ):
        """
        Args:
            token_classifier_dir: Path to trained token classification model
            masked_lm_name: Pretrained masked LM for NLL spike detection
            lexicon_path: Path to Vietnamese lexicon file (one word per line)
            device: torch device
            weight_*: Ensemble weights (should sum to 1.0)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensemble weights
        self.weight_oov = weight_oov
        self.weight_mlm = weight_mlm
        self.weight_classifier = weight_classifier
        
        # 1. Load lexicon for OOV detection
        self.lexicon = self._load_lexicon(lexicon_path) if lexicon_path else set()
        
        # 2. Load masked LM for NLL spike detection
        self.mlm_tokenizer = None
        self.mlm_model = None
        if masked_lm_name:
            try:
                self.mlm_tokenizer = AutoTokenizer.from_pretrained(masked_lm_name, use_fast=True)
                self.mlm_model = AutoModelForMaskedLM.from_pretrained(masked_lm_name).to(self.device).eval()
            except Exception as e:
                print(f"[Warning] Failed to load masked LM: {e}")
        
        # 3. Load token classifier
        self.classifier_tokenizer = None
        self.classifier_model = None
        if token_classifier_dir and os.path.exists(token_classifier_dir):
            try:
                self.classifier_tokenizer = AutoTokenizer.from_pretrained(token_classifier_dir, use_fast=True)
                self.classifier_model = AutoModelForTokenClassification.from_pretrained(
                    token_classifier_dir
                ).to(self.device).eval()
            except Exception as e:
                print(f"[Warning] Failed to load token classifier: {e}")
    
    def _load_lexicon(self, path: str) -> Set[str]:
        """Load lexicon from file (one word per line)"""
        lexicon = set()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        lexicon.add(word)
        return lexicon
    
    @torch.no_grad()
    def detect_oov(self, tokens: List[str]) -> List[float]:
        """
        OOV detection: returns confidence [0, 1] for each token
        1.0 = definitely OOV (error), 0.0 = in vocabulary (correct)
        """
        if not self.lexicon:
            return [0.0] * len(tokens)

        scores = []
        for token in tokens:
            # Normalize token
            token_lower = token.lower()
            # Check if in lexicon
            is_oov = token_lower not in self.lexicon
            # Simple binary score (can be made more sophisticated)
            scores.append(1.0 if is_oov else 0.0)

        return scores

    @torch.no_grad()
    def detect_mlm_spike(self, tokens: List[str], spike_threshold: float = 2.0) -> List[float]:
        """
        Masked-LM NLL spike detection
        For each token, mask it and compute NLL. High NLL = likely error.

        Args:
            tokens: List of tokens
            spike_threshold: Threshold for NLL spike (in standard deviations)

        Returns:
            List of confidence scores [0, 1]
        """
        if not self.mlm_model or not self.mlm_tokenizer:
            return [0.0] * len(tokens)

        if not tokens:
            return []

        # Join tokens to text
        text = " ".join(tokens)

        # Tokenize
        encoding = self.mlm_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(self.device)

        # Get word_ids mapping
        word_ids = encoding.word_ids(batch_index=0) if hasattr(encoding, 'word_ids') else None

        if word_ids is None:
            # Fallback: assume 1-to-1 mapping
            return [0.0] * len(tokens)

        # Compute NLL for each token position
        nlls = []
        input_ids = encoding['input_ids'][0]

        for token_idx in range(len(tokens)):
            # Find subword positions for this token
            subword_positions = [i for i, wid in enumerate(word_ids) if wid == token_idx]

            if not subword_positions:
                nlls.append(0.0)
                continue

            # Mask each subword and compute NLL
            token_nlls = []
            for pos in subword_positions:
                # Create masked input
                masked_input = input_ids.clone()
                original_id = masked_input[pos].item()
                masked_input[pos] = self.mlm_tokenizer.mask_token_id

                # Forward pass
                outputs = self.mlm_model(masked_input.unsqueeze(0))
                logits = outputs.logits[0, pos]

                # Compute NLL for original token
                log_probs = F.log_softmax(logits, dim=-1)
                nll = -log_probs[original_id].item()
                token_nlls.append(nll)

            # Average NLL for this token
            avg_nll = sum(token_nlls) / len(token_nlls) if token_nlls else 0.0
            nlls.append(avg_nll)

        # Normalize NLLs to [0, 1] using z-score
        if len(nlls) > 1:
            mean_nll = sum(nlls) / len(nlls)
            std_nll = math.sqrt(sum((x - mean_nll) ** 2 for x in nlls) / len(nlls))

            if std_nll > 0:
                scores = []
                for nll in nlls:
                    z_score = (nll - mean_nll) / std_nll
                    # Convert z-score to confidence [0, 1]
                    # z > spike_threshold => high confidence of error
                    conf = max(0.0, min(1.0, z_score / spike_threshold))
                    scores.append(conf)
                return scores

        # Fallback: no normalization possible
        return [0.0] * len(tokens)

    @torch.no_grad()
    def detect_token_classifier(self, tokens: List[str], threshold: float = 0.5) -> List[float]:
        """
        Token classification detection (existing model)

        Returns:
            List of confidence scores [0, 1]
        """
        if not self.classifier_model or not self.classifier_tokenizer:
            return [0.0] * len(tokens)

        if not tokens:
            return []

        # Tokenize
        encoding = self.classifier_tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(self.device)

        # Forward pass
        logits = self.classifier_model(**encoding).logits[0]
        probs = F.softmax(logits, dim=-1)[:, 1]  # class=1 is error

        # Map subword probabilities back to word positions
        word_ids = encoding.word_ids(batch_index=0) if hasattr(encoding, 'word_ids') else None

        if word_ids is None:
            # Fallback: use first N probabilities
            return [float(p) for p in probs[:len(tokens)]]

        # Aggregate probabilities by word
        word_probs = {}
        for i, wid in enumerate(word_ids):
            if wid is not None:
                if wid not in word_probs:
                    word_probs[wid] = []
                word_probs[wid].append(float(probs[i]))

        # Average probabilities for each word
        scores = []
        for token_idx in range(len(tokens)):
            if token_idx in word_probs:
                avg_prob = sum(word_probs[token_idx]) / len(word_probs[token_idx])
                scores.append(avg_prob)
            else:
                scores.append(0.0)

        return scores

    def detect(
        self,
        tokens: List[str],
        threshold: float = 0.5,
        use_oov: bool = True,
        use_mlm: bool = True,
        use_classifier: bool = True,
    ) -> List[DetectionResult]:
        """
        Multi-detector ensemble

        Args:
            tokens: List of tokens to check
            threshold: Final confidence threshold for flagging errors
            use_*: Enable/disable individual detectors

        Returns:
            List of DetectionResult for flagged positions
        """
        if not tokens:
            return []

        # Run individual detectors
        oov_scores = self.detect_oov(tokens) if use_oov else [0.0] * len(tokens)
        mlm_scores = self.detect_mlm_spike(tokens) if use_mlm else [0.0] * len(tokens)
        clf_scores = self.detect_token_classifier(tokens) if use_classifier else [0.0] * len(tokens)

        # Ensemble: weighted average
        results = []
        for i, token in enumerate(tokens):
            # Compute weighted score
            score = (
                self.weight_oov * oov_scores[i] +
                self.weight_mlm * mlm_scores[i] +
                self.weight_classifier * clf_scores[i]
            )

            # Flag if above threshold
            if score >= threshold:
                results.append(DetectionResult(
                    position=i,
                    token=token,
                    confidence=score,
                    detector_scores={
                        'oov': oov_scores[i],
                        'mlm': mlm_scores[i],
                        'classifier': clf_scores[i],
                    }
                ))

        return results


# =========================
# 3. CANDIDATE GENERATOR (Placeholder for Phase 2)
# =========================

class CandidateGenerator:
    """
    Generate correction candidates using multiple strategies:
    - SymSpell (with/without diacritics)
    - Telex/VNI conversion
    - Keyboard adjacency
    - Split/Join
    - Phonetic (optional)
    """

    def __init__(self):
        # TODO: Implement in Phase 2
        pass

    def generate(self, token: str, max_candidates: int = 10) -> List[str]:
        """Generate correction candidates for a token"""
        # Placeholder
        return [token]


# =========================
# 4. NOISY-CHANNEL RANKER (Placeholder for Phase 2)
# =========================

class NoisyChannelRanker:
    """
    Rank candidates using noisy-channel model with feature stack:
    score(c) = λ1*LM_masked + λ2*LM_5gram + λ3*log(P_err) + λ4*log(freq) - λ5*edit_dist + λ6*ortho
    """

    def __init__(self):
        # TODO: Implement in Phase 2
        pass

    def rank(self, candidates: List[str], context: str) -> List[Tuple[str, float]]:
        """Rank candidates by score"""
        # Placeholder
        return [(c, 0.0) for c in candidates]


# =========================
# MAIN ADVANCED CORRECTOR
# =========================

class AdvancedCorrector:
    """
    Main advanced correction pipeline
    Combines preprocessing, detection, generation, ranking
    """

    def __init__(
        self,
        detector_dir: Optional[str] = None,
        lexicon_path: Optional[str] = None,
        use_word_segmenter: bool = False,
        device: Optional[str] = None,
    ):
        self.preprocessor = VietnamesePreprocessor(use_word_segmenter=use_word_segmenter)
        self.detector = MultiDetector(
            token_classifier_dir=detector_dir,
            lexicon_path=lexicon_path,
            device=device,
        )
        self.generator = CandidateGenerator()
        self.ranker = NoisyChannelRanker()

    def correct(
        self,
        text: str,
        detection_threshold: float = 0.5,
        protect_patterns: bool = True,
    ) -> Dict:
        """
        Correct text using advanced pipeline

        Returns:
            {
                'input': original text,
                'preprocessed': preprocessing info,
                'detections': list of DetectionResult,
                'corrections': list of corrections,
                'final': corrected text,
            }
        """
        # Step 1: Preprocess
        preprocessed = self.preprocessor.preprocess(text, protect_patterns=protect_patterns)
        tokens = preprocessed['tokens']

        # Step 2: Detect errors
        detections = self.detector.detect(tokens, threshold=detection_threshold)

        # Step 3: Generate & rank candidates (placeholder for now)
        corrections = []
        for det in detections:
            candidates = self.generator.generate(det.token)
            ranked = self.ranker.rank(candidates, text)
            if ranked:
                best_candidate = ranked[0][0]
                corrections.append({
                    'position': det.position,
                    'original': det.token,
                    'correction': best_candidate,
                    'confidence': det.confidence,
                })

        # Step 4: Apply corrections
        corrected_tokens = tokens.copy()
        for corr in corrections:
            corrected_tokens[corr['position']] = corr['correction']

        final_text = " ".join(corrected_tokens)

        # Restore protected patterns
        if protect_patterns and preprocessed['protected_map']:
            final_text = self.preprocessor.restore_special_patterns(
                final_text,
                preprocessed['protected_map']
            )

        return {
            'input': text,
            'preprocessed': preprocessed,
            'detections': [
                {
                    'position': d.position,
                    'token': d.token,
                    'confidence': d.confidence,
                    'detector_scores': d.detector_scores,
                }
                for d in detections
            ],
            'corrections': corrections,
            'final': final_text,
        }

