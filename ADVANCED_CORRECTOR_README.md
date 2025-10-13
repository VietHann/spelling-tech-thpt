# Advanced Vietnamese Spell Corrector

## 📋 Tổng quan

Hệ thống sửa lỗi chính tả tiếng Việt nâng cao với kiến trúc 3 tầng:

### **TIER 1: Realtime (Đã triển khai Phase 1)**
- ✅ **Preprocessing**: Unicode NFC, sentence split, word segmentation
- ✅ **Multi-Detector**: OOV + masked-LM NLL spike + token-classifier ensemble
- 🚧 **Candidate Generator**: SymSpell + Telex/VNI + keyboard adjacency (Phase 2)
- 🚧 **Noisy-Channel Ranker**: Feature stack với LM + P_err + freq (Phase 2)
- 🚧 **Global Search**: Viterbi beam search (Phase 3)

### **TIER 2: Heavy / Fix All (Phase 3)**
- 🚧 **GEC Seq2Seq**: T5/BART với constrained decoding
- 🚧 **Global Rescoring**: External LM + penalty

### **TIER 3: Post-processing & UX (Phase 3)**
- 🚧 **Post-rules**: Punctuation, whitespace, capitalization
- 🚧 **UX**: Underline errors, suggestions, explain

---

## 🚀 Cài đặt

### 1. Dependencies

```bash
# Core dependencies (đã có)
pip install torch transformers fastapi uvicorn

# Optional: Word segmentation (khuyến nghị)
pip install underthesea
# hoặc
pip install pyvi

# Optional: Testing
pip install requests
```

### 2. Chuẩn bị dữ liệu

#### Tạo lexicon mẫu (cho testing):
```bash
python prepare_data.py --create_sample
```

Tạo file `data/vi_lexicon.txt` với ~200 từ phổ biến.

#### Tạo lexicon từ Hunspell (production):
```bash
# Download Hunspell Vietnamese dictionary
wget https://raw.githubusercontent.com/LibreOffice/dictionaries/master/vi/vi_VN.dic

# Build lexicon
python prepare_data.py --hunspell_dic vi_VN.dic --output_dir data
```

#### Tạo lexicon từ corpus:
```bash
# Nếu bạn có corpus tiếng Việt (plain text)
python prepare_data.py --corpus path/to/corpus.txt --output_dir data
```

### 3. Train models (nếu chưa có)

```bash
# Train detector
python vi_spell_pipeline_plus.py --do_train_detector \
    --det_out outputs/detector \
    --det_epochs 3

# Train corrector
python vi_spell_pipeline_plus.py --do_train_corrector \
    --corr_out outputs/corrector \
    --corr_epochs 3
```

---

## 📖 Sử dụng

### 1. Chạy API server

```bash
# Set environment variables (optional)
export DET_DIR=outputs/detector
export CORR_DIR=outputs/corrector
export LEXICON_PATH=data/vi_lexicon.txt

# Start server
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 2. Test endpoints

#### Health check:
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "device": "cuda",
  "det_dir": "outputs/detector",
  "corr_dir": "outputs/corrector",
  "advanced_corrector": {
    "enabled": true,
    "has_lexicon": true,
    "lexicon_size": 234
  }
}
```

#### Original corrector (v1):
```bash
curl -X POST http://localhost:8000/correct \
  -H "Content-Type: application/json" \
  -d '{"text": "Tôii đangg họcc tiếng Việt"}'
```

#### Advanced corrector (v2):
```bash
curl -X POST http://localhost:8000/correct_v2 \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Tôii đangg họcc tiếng Việt",
    "detection_threshold": 0.5,
    "use_oov": true,
    "use_mlm": false,
    "use_classifier": true,
    "protect_patterns": true
  }'
```

Response:
```json
{
  "input": "Tôii đangg họcc tiếng Việt",
  "preprocessed": {
    "normalized": "Tôii đangg họcc tiếng Việt",
    "tokens": ["Tôii", "đangg", "họcc", "tiếng", "Việt"],
    "sentences": ["Tôii đangg họcc tiếng Việt"]
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

### 3. Python API

```python
from advanced_corrector import AdvancedCorrector

# Initialize
corrector = AdvancedCorrector(
    detector_dir="outputs/detector",
    lexicon_path="data/vi_lexicon.txt",
    use_word_segmenter=False,  # Set True if underthesea installed
)

# Correct text
result = corrector.correct(
    text="Tôii đangg họcc tiếng Việt",
    detection_threshold=0.5,
    protect_patterns=True,
)

print(f"Input: {result['input']}")
print(f"Detections: {len(result['detections'])}")
print(f"Final: {result['final']}")
```

### 4. Run tests

```bash
python test_advanced_corrector.py
```

---

## 🔧 Cấu hình

### Multi-Detector weights

Trong `app.py` hoặc khi khởi tạo `MultiDetector`:

```python
detector = MultiDetector(
    token_classifier_dir="outputs/detector",
    lexicon_path="data/vi_lexicon.txt",
    # Ensemble weights (should sum to 1.0)
    weight_oov=0.3,          # OOV detection
    weight_mlm=0.3,          # Masked-LM spike (slow!)
    weight_classifier=0.4,   # Token classifier
)
```

**Lưu ý**: 
- `use_mlm=True` rất chậm (masked-LM inference cho mỗi token)
- Khuyến nghị: `use_mlm=False` cho realtime, `use_mlm=True` cho "Fix All"

### API parameters

#### `/correct_v2` endpoint:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to correct |
| `detection_threshold` | float | 0.5 | Confidence threshold [0, 1] |
| `protect_patterns` | bool | true | Protect URLs, emails, code |
| `use_oov` | bool | true | Enable OOV detector |
| `use_mlm` | bool | false | Enable masked-LM detector (slow!) |
| `use_classifier` | bool | true | Enable token classifier |

---

## 📊 So sánh v1 vs v2

| Feature | v1 (Original) | v2 (Advanced) |
|---------|---------------|---------------|
| **Preprocessing** | Simple syllable split | Unicode NFC + sentence split + word segment |
| **Detection** | Token classifier only | Multi-detector ensemble (OOV + MLM + classifier) |
| **Pattern protection** | ❌ | ✅ URLs, emails, code |
| **Detailed scores** | ❌ | ✅ Per-detector confidence |
| **Speed** | Fast | Fast (if MLM disabled) |
| **Accuracy** | Good | Better (with lexicon) |

---

## 🛣️ Roadmap

### ✅ Phase 1 (Completed)
- [x] Preprocessing pipeline
- [x] Multi-detector (OOV + MLM + classifier)
- [x] API endpoint `/correct_v2`
- [x] Test suite
- [x] Documentation

### 🚧 Phase 2 (Next)
- [ ] Candidate Generator
  - [ ] SymSpell (with/without diacritics)
  - [ ] Telex/VNI conversion
  - [ ] Keyboard adjacency
  - [ ] Split/Join
- [ ] Noisy-Channel Ranker
  - [ ] KenLM 5-gram
  - [ ] Feature stack (LM + P_err + freq + edit + ortho)
- [ ] Data preparation
  - [ ] Train KenLM from corpus
  - [ ] Build frequency dict
  - [ ] Toneless-to-toned mapping

### 🚧 Phase 3 (Future)
- [ ] Global Search (Viterbi beam)
- [ ] GEC Seq2Seq (T5/BART)
- [ ] Global Rescoring
- [ ] Post-processing rules
- [ ] UX features (underline, explain, dictionary)

---

## 🐛 Troubleshooting

### 1. "Detector model not found"
```bash
# Train detector first
python vi_spell_pipeline_plus.py --do_train_detector
```

### 2. "Lexicon not found"
```bash
# Create sample lexicon
python prepare_data.py --create_sample
```

### 3. "MLM detection is slow"
```bash
# Disable MLM in API request
curl -X POST http://localhost:8000/correct_v2 \
  -d '{"text": "...", "use_mlm": false}'
```

### 4. "underthesea not found"
```bash
# Install word segmenter (optional)
pip install underthesea

# Or use simple syllable splitting (default)
# Set use_word_segmenter=False in AdvancedCorrector
```

---

## 📝 Examples

### Example 1: Basic correction
```python
corrector.correct("Tôii đangg họcc tiếng Việt")
# Detects: Tôii, đangg, họcc (OOV + classifier)
```

### Example 2: Pattern protection
```python
corrector.correct(
    "Email: test@example.com và website: https://example.com",
    protect_patterns=True
)
# Protects: test@example.com, https://example.com
```

### Example 3: Custom threshold
```python
corrector.correct(
    "Hôm nay trời đẹpp quá",
    detection_threshold=0.7  # Higher threshold = fewer detections
)
```

---

## 📚 References

- **Hunspell Vietnamese**: https://github.com/LibreOffice/dictionaries/tree/master/vi
- **underthesea**: https://github.com/undertheseanlp/underthesea
- **pyvi**: https://github.com/trungtv/pyvi
- **SymSpell**: https://github.com/wolfgarbe/SymSpell
- **KenLM**: https://github.com/kpu/kenlm

