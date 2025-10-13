# Phase 2 Implementation Plan

## 🎯 Mục tiêu

Implement **Candidate Generator** và **Noisy-Channel Ranker** để hoàn thiện pipeline sửa lỗi.

Hiện tại Phase 1 chỉ **detect** lỗi, Phase 2 sẽ **generate candidates** và **rank** để chọn correction tốt nhất.

---

## 📋 Tasks

### 2.1 Candidate Generator

#### A. SymSpell
- [ ] Implement SymSpell algorithm (edit distance 1-2)
- [ ] Build SymSpell dictionary từ lexicon
- [ ] Support cả có dấu và không dấu
- [ ] Optimize với prefix tree

**Files:**
- `candidate_generator.py` → `SymSpellGenerator`

**Data needed:**
- Lexicon với frequency counts
- Toneless → toned mapping

#### B. Telex/VNI Converter
- [ ] Build Telex conversion rules (s→ś, f→̀, aa→â, uw→ư, dd→đ)
- [ ] Build VNI conversion rules (1→́, 2→̀, 3→̉, 4→̃, 5→̣)
- [ ] Reverse conversion (detect Telex/VNI patterns)
- [ ] Generate candidates by applying rules

**Files:**
- `candidate_generator.py` → `TelexVNIGenerator`

**Data needed:**
- Telex/VNI mapping tables
- Common error patterns

#### C. Keyboard Adjacency
- [ ] Build Vietnamese keyboard layout map (QWERTY)
- [ ] Generate candidates by swapping adjacent keys
- [ ] Support Damerau-Levenshtein (swap, insert, delete, replace)

**Files:**
- `candidate_generator.py` → `KeyboardGenerator`

**Data needed:**
- Keyboard layout map (q↔w, a↔s, etc.)

#### D. Split/Join
- [ ] Detect merged words ("đểlàm" → "để làm")
- [ ] Detect split words ("để làm" → "đểlàm")
- [ ] Use lexicon to validate splits

**Files:**
- `candidate_generator.py` → `SplitJoinGenerator`

#### E. Toneless → Toned
- [ ] Build mapping từ không dấu → có dấu
- [ ] Handle ambiguous cases (e.g., "viet" → "việt", "viết", "viêt")
- [ ] Use frequency to rank

**Files:**
- `candidate_generator.py` → `DiacriticRestorer`

**Data needed:**
- Toneless → toned mapping with frequencies

---

### 2.2 Noisy-Channel Ranker

#### A. Feature Extractors

**Features:**
1. **LM_masked**: Masked-LM score (PhoBERT)
2. **LM_5gram**: 5-gram LM score (KenLM)
3. **P_err**: Error model probability
   - Telex/VNI probability
   - Keyboard adjacency probability
   - Edit operation probability
4. **freq**: Unigram frequency
5. **edit_dist**: Edit distance from observed
6. **ortho**: Orthography bonus (valid Vietnamese syllable structure)

**Files:**
- `ranker.py` → `FeatureExtractor`

#### B. Scoring Function

```python
score(c) = λ1*LM_masked(c, ctx) 
         + λ2*LM_5gram(c, ctx)
         + λ3*log(P_err(observed | c))
         + λ4*log(freq(c))
         - λ5*edit_dist(observed, c)
         + λ6*ortho_bonus(c)
```

**Files:**
- `ranker.py` → `NoisyChannelRanker`

#### C. Weight Tuning
- [ ] Implement grid search on dev set
- [ ] Optimize weights (λ1, λ2, ..., λ6)
- [ ] Save best weights to config

**Files:**
- `tune_ranker.py`

---

### 2.3 Data Preparation

#### A. KenLM 5-gram
- [ ] Download Vietnamese corpus (Wikipedia, news)
- [ ] Clean and tokenize corpus
- [ ] Train KenLM 5-gram model
- [ ] Prune model for speed

**Commands:**
```bash
# Install KenLM
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir build && cd build
cmake .. && make -j4

# Train model
./bin/lmplz -o 5 < corpus.txt > vi_5gram.arpa
./bin/build_binary vi_5gram.arpa vi_5gram.bin
```

**Files:**
- `data/vi_5gram.bin`

#### B. Frequency Dictionary
- [ ] Extract unigram/bigram frequencies from corpus
- [ ] Save to JSON

**Files:**
- `data/vi_freq.json`

**Format:**
```json
{
  "tôi": 123456,
  "là": 98765,
  "có": 87654,
  ...
}
```

#### C. Toneless → Toned Mapping
- [ ] Build mapping from lexicon
- [ ] Handle ambiguous cases with frequencies

**Files:**
- `data/toneless_to_toned.json`

**Format:**
```json
{
  "viet": [
    {"word": "việt", "freq": 10000},
    {"word": "viết", "freq": 5000}
  ],
  ...
}
```

#### D. Error Patterns
- [ ] Collect Telex/VNI error patterns
- [ ] Collect keyboard adjacency patterns
- [ ] Estimate P_err from corpus

**Files:**
- `data/error_patterns.json`

---

## 🔧 Implementation Order

### Week 1: Candidate Generator
1. **Day 1-2**: SymSpell
   - Implement algorithm
   - Build dictionary
   - Test on sample data

2. **Day 3-4**: Telex/VNI
   - Build conversion rules
   - Implement reverse detection
   - Test on common errors

3. **Day 5**: Keyboard + Split/Join
   - Keyboard adjacency map
   - Split/join logic
   - Integration test

### Week 2: Ranker + Data
1. **Day 1-2**: Feature Extractors
   - Implement all 6 features
   - Test individually

2. **Day 3-4**: Noisy-Channel Ranker
   - Implement scoring function
   - Initial weights (manual tuning)
   - Integration with generator

3. **Day 5**: Data Preparation
   - Train KenLM
   - Build frequency dict
   - Build mappings

### Week 3: Integration + Tuning
1. **Day 1-2**: End-to-end integration
   - Update `AdvancedCorrector`
   - Update API endpoint
   - Test full pipeline

2. **Day 3-4**: Weight Tuning
   - Grid search on dev set
   - Optimize weights
   - Evaluate on test set

3. **Day 5**: Documentation + Testing
   - Update README
   - Write tests
   - Performance benchmarks

---

## 📊 Expected Results

### Before Phase 2 (Current)
```
Input:  "Tôii đangg họcc tiếng Việt"
Output: {
  "detections": [
    {"position": 0, "token": "Tôii", "confidence": 0.85},
    {"position": 1, "token": "đangg", "confidence": 0.82},
    {"position": 2, "token": "họcc", "confidence": 0.79}
  ],
  "corrections": [],  # Empty!
  "final": "Tôii đangg họcc tiếng Việt"  # No change
}
```

### After Phase 2 (Expected)
```
Input:  "Tôii đangg họcc tiếng Việt"
Output: {
  "detections": [
    {"position": 0, "token": "Tôii", "confidence": 0.85},
    {"position": 1, "token": "đangg", "confidence": 0.82},
    {"position": 2, "token": "họcc", "confidence": 0.79}
  ],
  "corrections": [
    {"position": 0, "original": "Tôii", "correction": "Tôi", "score": 0.95},
    {"position": 1, "original": "đangg", "correction": "đang", "score": 0.93},
    {"position": 2, "original": "họcc", "correction": "học", "score": 0.91}
  ],
  "final": "Tôi đang học tiếng Việt"  # Corrected!
}
```

---

## 🧪 Testing Strategy

### Unit Tests
- [ ] Test each generator individually
- [ ] Test feature extractors
- [ ] Test ranker scoring

### Integration Tests
- [ ] Test generator → ranker pipeline
- [ ] Test full correction pipeline
- [ ] Test API endpoint

### Evaluation Metrics
- [ ] Exact Match (EM)
- [ ] Normalized Edit Distance (NED)
- [ ] Precision/Recall/F1 for detection
- [ ] Correction accuracy

### Benchmark Dataset
- Use VSEC test set
- Compare with v1 (original corrector)
- Target: EM > 0.7, NED < 0.1

---

## 📦 Dependencies

### Python packages
```bash
pip install symspellpy  # SymSpell
pip install kenlm       # KenLM Python bindings
```

### External tools
- KenLM (for training 5-gram model)
- Vietnamese corpus (Wikipedia dump)

---

## 🚀 Quick Start (After Phase 2)

```python
from advanced_corrector import AdvancedCorrector

corrector = AdvancedCorrector(
    detector_dir="outputs/detector",
    lexicon_path="data/vi_lexicon.txt",
    kenlm_path="data/vi_5gram.bin",
    freq_dict_path="data/vi_freq.json",
)

result = corrector.correct("Tôii đangg họcc tiếng Việt")
print(result['final'])  # "Tôi đang học tiếng Việt"
```

---

## 📝 Files to Create

### Phase 2 Files
```
T3_VietLe/
├── candidate_generator.py      # NEW
│   ├── SymSpellGenerator
│   ├── TelexVNIGenerator
│   ├── KeyboardGenerator
│   ├── SplitJoinGenerator
│   └── DiacriticRestorer
├── ranker.py                    # NEW
│   ├── FeatureExtractor
│   └── NoisyChannelRanker
├── tune_ranker.py               # NEW
├── prepare_kenlm.py             # NEW
├── data/
│   ├── vi_5gram.bin             # NEW
│   ├── vi_freq.json             # NEW
│   ├── toneless_to_toned.json   # NEW
│   └── error_patterns.json      # NEW
└── tests/
    ├── test_generator.py        # NEW
    └── test_ranker.py           # NEW
```

---

## 🎯 Success Criteria

Phase 2 is complete when:
- [x] All 5 generators implemented and tested
- [x] Noisy-channel ranker with 6 features
- [x] KenLM 5-gram trained and integrated
- [x] Frequency dict and mappings built
- [x] End-to-end correction works
- [x] EM > 0.7 on VSEC test set
- [x] API returns actual corrections (not empty)
- [x] Documentation updated
- [x] Tests pass

---
