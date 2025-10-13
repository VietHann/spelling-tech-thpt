# 🚀 Advanced Vietnamese Spell Corrector - Quick Guide

## ⚡ Quick Start (3 bước)

### 1️⃣ Chuẩn bị lexicon
```bash
python prepare_data.py --create_sample
```

### 2️⃣ Test (không cần server)
```bash
python demo_standalone.py
```

### 3️⃣ Chạy API server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 📡 API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Original Corrector (v1)
```bash
curl -X POST http://localhost:8000/correct \
  -H "Content-Type: application/json" \
  -d '{"text": "Tôii đangg họcc tiếng Việt"}'
```

### Advanced Corrector (v2) ⭐ NEW
```bash
curl -X POST http://localhost:8000/correct_v2 \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Tôii đangg họcc tiếng Việt",
    "detection_threshold": 0.5,
    "use_oov": true,
    "use_mlm": false,
    "use_classifier": true
  }'
```

---

## 🎯 Phase 1 Features (✅ Hoàn thành)

| Feature | Status | Description |
|---------|--------|-------------|
| **Preprocessing** | ✅ | Unicode NFC, sentence split, word segment |
| **Multi-Detector** | ✅ | OOV + Masked-LM + Token Classifier |
| **Pattern Protection** | ✅ | URLs, emails, code |
| **Detailed Scores** | ✅ | Per-detector confidence |
| **API v2** | ✅ | `/correct_v2` endpoint |

---

## 🚧 Phase 2 (Đang lên kế hoạch)

| Feature | Status | Description |
|---------|--------|-------------|
| **Candidate Generator** | 🚧 | SymSpell, Telex/VNI, keyboard |
| **Noisy-Channel Ranker** | 🚧 | LM + P_err + freq + edit |
| **KenLM 5-gram** | 🚧 | Language model |
| **Corrections** | 🚧 | Actually fix errors! |

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| `README_QUICK.md` | ⭐ This file (quick reference) |
| `ADVANCED_CORRECTOR_README.md` | Full user guide |
| `PHASE1_SUMMARY.md` | Implementation details |
| `PHASE2_PLAN.md` | Phase 2 roadmap |
| `IMPLEMENTATION_COMPLETE.md` | Phase 1 completion summary |

---

## 🧪 Testing

### Standalone demo (no server needed)
```bash
python demo_standalone.py
```

### Full test suite
```bash
python test_advanced_corrector.py
```

### Quick start script
**Windows:**
```powershell
.\demo_quick_start.ps1
```

**Linux/Mac:**
```bash
./demo_quick_start.sh
```

---

## 🔧 Configuration

### Detector weights (in code)
```python
detector = MultiDetector(
    weight_oov=0.3,          # OOV detection
    weight_mlm=0.3,          # Masked-LM (slow!)
    weight_classifier=0.4,   # Token classifier
)
```

### API parameters
```json
{
  "detection_threshold": 0.5,  // Confidence threshold [0, 1]
  "protect_patterns": true,    // Protect URLs, emails
  "use_oov": true,             // Enable OOV detector
  "use_mlm": false,            // Enable MLM (slow!)
  "use_classifier": true       // Enable token classifier
}
```

---

## ⚠️ Important Notes

### 1. MLM Detector is SLOW
- Masked-LM inference for each token
- **Recommendation**: `use_mlm=false` for realtime
- Use `use_mlm=true` only for "Fix All" mode

### 2. Corrections are Empty (Phase 1)
- Phase 1 only **detects** errors
- Phase 2 will **generate & rank** corrections
- Current output: `"corrections": []`

### 3. Sample Lexicon is Small
- Only ~200 common words
- For production: use Hunspell or corpus
```bash
python prepare_data.py --hunspell_dic vi_VN.dic
```

---

## 🐛 Troubleshooting

### "Detector model not found"
```bash
python vi_spell_pipeline_plus.py --do_train_detector
```

### "Lexicon not found"
```bash
python prepare_data.py --create_sample
```

### "Server not running"
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 📊 Example Output

### Input
```
"Tôii đangg họcc tiếng Việt"
```

### Output (Phase 1)
```json
{
  "detections": [
    {"position": 0, "token": "Tôii", "confidence": 0.85},
    {"position": 1, "token": "đangg", "confidence": 0.82},
    {"position": 2, "token": "họcc", "confidence": 0.79}
  ],
  "corrections": [],
  "final": "Tôii đangg họcc tiếng Việt"
}
```

### Expected Output (After Phase 2)
```json
{
  "detections": [...],
  "corrections": [
    {"position": 0, "original": "Tôii", "correction": "Tôi"},
    {"position": 1, "original": "đangg", "correction": "đang"},
    {"position": 2, "original": "họcc", "correction": "học"}
  ],
  "final": "Tôi đang học tiếng Việt"
}
```

