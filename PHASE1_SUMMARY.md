# Phase 1 Implementation Summary

## ✅ Hoàn thành

### 📁 Files đã tạo

1. **`advanced_corrector.py`** (631 dòng)
   - `VietnamesePreprocessor`: Unicode NFC, sentence split, word segmentation, pattern protection
   - `MultiDetector`: Ensemble 3 detectors (OOV + masked-LM + token-classifier)
   - `CandidateGenerator`: Placeholder cho Phase 2
   - `NoisyChannelRanker`: Placeholder cho Phase 2
   - `AdvancedCorrector`: Main pipeline

2. **`app.py`** (đã cập nhật)
   - Import `AdvancedCorrector`
   - Khởi tạo `_advanced_corrector`
   - Endpoint mới: `POST /correct_v2`
   - Schema mới: `CorrectV2Request`, `CorrectV2Response`
   - Health check cập nhật với thông tin advanced corrector

3. **`prepare_data.py`** (280 dòng)
   - Build lexicon từ Hunspell
   - Build lexicon từ corpus
   - Build frequency dict
   - Create sample lexicon (cho testing)

4. **`test_advanced_corrector.py`** (230 dòng)
   - Test preprocessor
   - Test multi-detector
   - Test full pipeline
   - Test API integration

5. **`ADVANCED_CORRECTOR_README.md`**
   - Hướng dẫn cài đặt
   - Hướng dẫn sử dụng
   - API documentation
   - Roadmap Phase 2 & 3

6. **`demo_quick_start.ps1`** & **`demo_quick_start.sh`**
   - Script tự động setup và test

---

## 🎯 Tính năng đã triển khai

### 1. Preprocessing Pipeline

```python
preprocessor = VietnamesePreprocessor(use_word_segmenter=False)
result = preprocessor.preprocess(text, protect_patterns=True)
```

**Features:**
- ✅ Unicode normalization (NFC)
- ✅ Sentence splitting
- ✅ Word segmentation (syllable-based hoặc underthesea/pyvi)
- ✅ Pattern protection (URLs, emails, code, numbers)
- ✅ Placeholder mapping để restore sau khi sửa

**Example:**
```python
Input:  "Email: test@example.com và website: https://example.com"
Output: {
  'normalized': "Email: ___EMAIL_0___ và website: ___URL_0___",
  'tokens': ['Email', ':', '___EMAIL_0___', 'và', 'website', ':', '___URL_0___'],
  'protected_map': {
    '___EMAIL_0___': 'test@example.com',
    '___URL_0___': 'https://example.com'
  }
}
```

### 2. Multi-Detector Ensemble

```python
detector = MultiDetector(
    token_classifier_dir="outputs/detector",
    lexicon_path="data/vi_lexicon.txt",
    weight_oov=0.3,
    weight_mlm=0.3,
    weight_classifier=0.4,
)
```

**3 Detectors:**

#### a) OOV Detector
- Kiểm tra từ có trong lexicon không
- Binary score: 1.0 (OOV) hoặc 0.0 (in vocab)
- Nhanh, không cần GPU

#### b) Masked-LM NLL Spike Detector
- Mask từng token, tính NLL từ PhoBERT
- High NLL = likely error
- Normalize bằng z-score
- **Chậm** (masked-LM inference cho mỗi token)

#### c) Token Classifier
- Sử dụng model đã train (PhoBERT token classification)
- Softmax probability cho class "error"
- Nhanh với GPU

**Ensemble:**
```
final_score = λ1*OOV + λ2*MLM + λ3*Classifier
```

**Example:**
```python
tokens = ["tôii", "đangg", "họcc", "tiếng", "việt"]
detections = detector.detect(tokens, threshold=0.5)

# Output:
[
  DetectionResult(position=0, token="tôii", confidence=0.85,
                  detector_scores={'oov': 1.0, 'mlm': 0.0, 'classifier': 0.7}),
  DetectionResult(position=1, token="đangg", confidence=0.82, ...),
  DetectionResult(position=2, token="họcc", confidence=0.79, ...)
]
```

### 3. API Endpoint `/correct_v2`

**Request:**
```json
{
  "text": "Tôii đangg họcc tiếng Việt",
  "detection_threshold": 0.5,
  "protect_patterns": true,
  "use_oov": true,
  "use_mlm": false,
  "use_classifier": true
}
```

**Response:**
```json
{
  "input": "Tôii đangg họcc tiếng Việt",
  "preprocessed": {
    "normalized": "Tôii đangg họcc tiếng Việt",
    "tokens": ["Tôii", "đangg", "họcc", "tiếng", "Việt"],
    "sentences": ["Tôii đangg họcc tiếng Việt"],
    "protected_map": {}
  },
  "detections": [
    {
      "position": 0,
      "token": "Tôii",
      "confidence": 0.85,
      "detector_scores": {
        "oov": 1.0,
        "mlm": 0.0,
        "classifier": 0.7
      }
    }
  ],
  "corrections": [],
  "final": "Tôii đangg họcc tiếng Việt"
}
```

**Note:** `corrections` trống vì Generator/Ranker chưa implement (Phase 2)

---

## 🚀 Cách sử dụng

### Quick Start (Windows)

```powershell
# 1. Tạo lexicon mẫu
python prepare_data.py --create_sample

# 2. Chạy tests
python test_advanced_corrector.py

# 3. Start server
uvicorn app:app --host 0.0.0.0 --port 8000

# 4. Test API (terminal khác)
curl -X POST http://localhost:8000/correct_v2 `
  -H "Content-Type: application/json" `
  -d '{"text": "Tôii đangg họcc tiếng Việt"}'
```

### Hoặc dùng script tự động

```powershell
.\demo_quick_start.ps1
```

---

## 📊 So sánh v1 vs v2

| Feature | v1 (Original) | v2 (Advanced - Phase 1) |
|---------|---------------|-------------------------|
| **Preprocessing** | Simple syllable split | ✅ Unicode NFC + sentence split + word segment |
| **Detection** | Token classifier only | ✅ Multi-detector ensemble (OOV + MLM + classifier) |
| **Pattern protection** | ❌ | ✅ URLs, emails, code |
| **Detailed scores** | ❌ | ✅ Per-detector confidence |
| **Candidate generation** | Seq2seq beam search | ❌ (Phase 2) |
| **Ranking** | Seq2seq score + NED | ❌ (Phase 2) |
| **Speed** | Fast | Fast (if MLM disabled) |

---

## 🔧 Configuration

### Detector weights

Trong `app.py`:
```python
_advanced_corrector = AdvancedCorrector(
    detector_dir=DET_DIR,
    lexicon_path=LEXICON_PATH,
    use_word_segmenter=False,  # Set True nếu có underthesea/pyvi
)

# Weights mặc định
detector.weight_oov = 0.3
detector.weight_mlm = 0.3
detector.weight_classifier = 0.4
```

### API request

```python
# Disable MLM (khuyến nghị cho realtime)
{
  "text": "...",
  "use_mlm": false,  # Tắt MLM detector (chậm)
  "use_oov": true,
  "use_classifier": true
}
```

---

## 🐛 Known Issues & Limitations

### Phase 1 Limitations

1. **Chưa có Candidate Generator**
   - Hiện tại chỉ detect lỗi, chưa generate candidates
   - `corrections` array luôn trống
   - Cần implement Phase 2

2. **Chưa có Ranker**
   - Chưa có noisy-channel ranking
   - Chưa có feature stack (LM + P_err + freq)
   - Cần implement Phase 2

3. **MLM Detector chậm**
   - Masked-LM inference cho mỗi token
   - Khuyến nghị: `use_mlm=false` cho realtime
   - Có thể dùng cho "Fix All" mode

4. **Lexicon mẫu nhỏ**
   - Sample lexicon chỉ ~200 từ
   - Cần build từ Hunspell hoặc corpus cho production

### Warnings (không ảnh hưởng)

- `pyvi` import warning: Optional dependency
- Unused imports: Chuẩn bị cho Phase 2
- Unused parameters: Placeholder functions

---

## 📝 Next Steps (Phase 2)

### 2.1 Candidate Generator

```python
class CandidateGenerator:
    def generate(self, token: str) -> List[str]:
        candidates = []
        
        # 1. SymSpell (edit distance 1-2)
        candidates += self.symspell_candidates(token)
        
        # 2. Telex/VNI conversion
        candidates += self.telex_vni_candidates(token)
        
        # 3. Keyboard adjacency
        candidates += self.keyboard_candidates(token)
        
        # 4. Split/Join
        candidates += self.split_join_candidates(token)
        
        # 5. Toneless → toned
        candidates += self.restore_diacritics(token)
        
        return candidates[:max_candidates]
```

### 2.2 Noisy-Channel Ranker

```python
class NoisyChannelRanker:
    def rank(self, candidates: List[str], context: str) -> List[Tuple[str, float]]:
        scores = []
        for c in candidates:
            score = (
                1.7 * self.lm_masked_score(c, context) +
                1.2 * self.lm_5gram_score(c, context) +
                0.9 * math.log(self.p_err(observed, c)) +
                0.6 * math.log(self.freq(c)) -
                0.3 * self.edit_distance(observed, c) +
                0.2 * self.orthography_bonus(c)
            )
            scores.append((c, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
```

### 2.3 Data Preparation

- [ ] Train KenLM 5-gram từ corpus
- [ ] Build unigram/bigram frequency dict
- [ ] Build toneless → toned mapping
- [ ] Collect Telex/VNI error patterns
- [ ] Build keyboard adjacency map

---

## 📚 Files Structure

```
T3_VietLe/
├── advanced_corrector.py          # Phase 1 implementation
├── app.py                          # API server (updated)
├── prepare_data.py                 # Data preparation scripts
├── test_advanced_corrector.py     # Test suite
├── ADVANCED_CORRECTOR_README.md   # User documentation
├── PHASE1_SUMMARY.md              # This file
├── demo_quick_start.ps1           # Windows quick start
├── demo_quick_start.sh            # Linux/Mac quick start
├── vi_spell_pipeline_plus.py      # Original training pipeline
├── data/
│   └── vi_lexicon.txt             # Vietnamese lexicon
├── outputs/
│   ├── detector/                  # Trained detector model
│   └── corrector/                 # Trained corrector model
└── static/
    ├── index.html
    ├── script.js
    └── style.css
```

---

## ✅ Checklist

### Phase 1 (Completed)
- [x] VietnamesePreprocessor
- [x] MultiDetector (OOV + MLM + Classifier)
- [x] API endpoint `/correct_v2`
- [x] Data preparation scripts
- [x] Test suite
- [x] Documentation
- [x] Quick start scripts

### Phase 2 (Next)
- [ ] CandidateGenerator
  - [ ] SymSpell
  - [ ] Telex/VNI
  - [ ] Keyboard adjacency
  - [ ] Split/Join
- [ ] NoisyChannelRanker
  - [ ] KenLM 5-gram
  - [ ] Feature stack
- [ ] Data artifacts
  - [ ] Train KenLM
  - [ ] Frequency dict
  - [ ] Error patterns

### Phase 3 (Future)
- [ ] Global Search (Viterbi)
- [ ] GEC Seq2Seq
- [ ] Post-processing
- [ ] UX features

---

