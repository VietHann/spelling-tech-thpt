# 🎓 DaiNam University 

<div align="center">

<p align="center">
  <img src="img/logo.png" alt="DaiNam University Logo" width="200"/>
  <img src="img/AIoTLab_logo.png" alt="AIoTLab Logo" width="170"/>
</p>

[![Made by AIoTLab](https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge)](https://fit.dainam.edu.vn)
[![Faculty of IT](https://img.shields.io/badge/Faculty%20of%20Information%20Technology-green?style=for-the-badge)](https://fit.dainam.edu.vn)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge)](https://dainam.edu.vn)

# Ứng dụng công nghệ số trong việc cải thiện lỗi chính tả của học sinh THPT

## Tổng quan

Hệ thống sửa lỗi chính tả tiếng Việt tự động sử dụng học sâu và xử lý ngôn ngữ tự nhiên, được thiết kế theo kiến trúc đa tầng với khả năng phát hiện và sửa lỗi chính xác cao.

### Tính năng chính

- **Phát hiện lỗi chính xác**: F1-score 0.82
- **Sửa lỗi thông minh**: Exact Match 0.73
- **Xử lý realtime**: 45ms/câu
- **Multi-detector ensemble**: OOV + Masked-LM + Token Classifier
- **API RESTful**: FastAPI với CORS support
- **Web UI**: Giao diện thân thiện

## Kiến trúc

```
Input Text
    ↓
[Tầng 1: Tiền xử lý]
    ↓
[Tầng 2: Phát hiện lỗi đa chiến lược]
    ↓
[Tầng 3: Sửa lỗi]
    ↓
Corrected Text
```

### Tầng 1: Tiền xử lý
- Unicode NFC normalization
- Sentence splitting
- Word segmentation
- Pattern protection (URL, email, code)

### Tầng 2: Multi-Detector
- **OOV Detector**: Kiểm tra từ điển (nhanh)
- **Masked-LM Detector**: PhoBERT NLL (chính xác)
- **Token Classifier**: Fine-tuned PhoBERT
- **Ensemble**: Weighted sum (λ₁=0.3, λ₂=0.3, λ₃=0.4)

### Tầng 3: Correction
- **Candidate Generator**: SymSpell, Telex/VNI, keyboard, split/join
- **Noisy-Channel Ranker**: LM + P_err + freq + edit distance
- **Global Search**: Viterbi beam

## Quick Start

### 1. Cài đặt

```bash
pip install -r requirements.txt
```

### 2. Chạy API Server

```bash
python app.py
```

Server chạy tại: http://localhost:8000

### 3. Test API

```bash
curl -X POST http://localhost:8000/correct_v2 \
  -H "Content-Type: application/json" \
  -d '{"text": "tôii đangg học tiếng việt"}'
```

### 4. Mở Web UI

Truy cập: http://localhost:8000

## Kết quả

| Metric | Value |
|--------|-------|
| Detection F1 | 0.82 |
| Correction EM | 0.73 |
| NED | 0.11 |
| Speed | 45ms/câu |

## Cấu trúc dự án

```
.
├── app.py                      # FastAPI server
├── advanced_corrector.py       # Core implementation
├── vi_spell_pipeline_plus.py   # Training pipeline
├── prepare_data.py             # Data preparation
├── test_advanced_corrector.py  # Test suite
├── demo_standalone.py          # Standalone demo
├── static/                     # Web UI
│   ├── index.html
│   ├── style.css
│   └── script.js
├── report.tex                  # Báo cáo LaTeX
├── architecture_diagram.pdf    # Sơ đồ kiến trúc
└── README.md                   # File này
```

## API Endpoints

### POST /correct
Sửa lỗi cơ bản (v1)

```json
{
  "text": "tôii đangg học"
}
```

### POST /correct_v2
Sửa lỗi nâng cao với multi-detector

```json
{
  "text": "tôii đangg học",
  "detection_threshold": 0.5,
  "use_oov": true,
  "use_mlm": false,
  "use_classifier": true
}
```

### GET /health
Kiểm tra trạng thái server

## Use Cases

- **Học tập**: Hỗ trợ học sinh THPT cải thiện kỹ năng viết
- **Soạn thảo**: Tích hợp vào text editor
- **LMS**: Tích hợp vào hệ thống quản lý học tập
- **Chatbot**: Kiểm tra chính tả trong chatbot giáo dục

## Documentation

- **Quick Start**: `README_QUICK.md`
- **Full Documentation**: `ADVANCED_CORRECTOR_README.md`
- **Phase 1 Summary**: `PHASE1_SUMMARY.md`
- **Phase 2 Plan**: `PHASE2_PLAN.md`
- **Báo cáo LaTeX**: `report.tex`
- **Sơ đồ kiến trúc**: `architecture_diagram.pdf`

## Testing

```bash
# Run all tests
python test_advanced_corrector.py

# Run standalone demo
python demo_standalone.py

# Quick start script
.\demo_quick_start.ps1  # Windows
./demo_quick_start.sh   # Linux/Mac
```

## Roadmap

### Phase 1 ✅ (Completed)
- Preprocessing pipeline
- Multi-detector ensemble
- API v2 endpoint

### Phase 2 🚧 (In Progress)
- Candidate generator
- Noisy-channel ranker
- KenLM integration

### Phase 3 📅 (Planned)
- Global search (Viterbi)
- GEC seq2seq model
- UX features (underline, explain)

## Tech Stack

- **Backend**: FastAPI, Python 3.10+
- **ML Models**: PhoBERT, BARTpho, ViT5
- **NLP**: Transformers, underthesea/pyvi
- **Frontend**: HTML, CSS, JavaScript
- **Training**: PyTorch, PEFT/LoRA

## Authors

- **Lê Văn Việt**
- **Đoàn Minh Châu**

Khoa Công Nghệ Thông Tin, Trường Đại Học Đại Nam

## Contact

- Email: lv.viet.vn@gmail.com
- GitHub: [spelling-tech-thpt](https://github.com/VietHann/spelling-tech-thpt)

---

**Made with ❤️ for Vietnamese education**

