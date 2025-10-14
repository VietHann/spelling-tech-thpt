# app.py
# FastAPI service cho Vietnamese Spell Correction + OCR (Gemini)
# Chạy: uvicorn app:app --host 0.0.0.0 --port 8000
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("ACCELERATE_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("PYTORCH_JIT", "0")
import io
import re
import json
import base64
from typing import List, Optional

import torch
from torch.nn import functional as F
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------- Hugging Face imports ----------
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
)

# ---------- Google Gemini (OCR) ----------
# pip install google-generativeai
import google.generativeai as genai

# ---------- Advanced Corrector ----------
from advanced_corrector import AdvancedCorrector

# =========================
# CẤU HÌNH MÔ HÌNH & GEMINI
# =========================
DET_DIR = os.environ.get("DET_DIR", "outputs/detector")
CORR_DIR = os.environ.get("CORR_DIR", "outputs/corr_lora_fast")

# Gán cứng API key vào code như bạn yêu cầu:
# (Bạn có thể đổi chuỗi này thành khóa thật của bạn)
GEMINI_API_KEY = ""
GEMINI_MODEL_NAME = "gemini-2.5-pro"  # nhanh & rẻ; có thể đổi "gemini-1.5-pro"

# Thư mục static để phục vụ giao diện
STATIC_DIR = os.environ.get("STATIC_DIR", "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# =========================
# TIỆN ÍCH TÁCH TIẾNG/NED
# =========================
PUNCT_REGEX = re.compile(r"([,.:;!?\"“”'‘’()\[\]{}…])")

def split_vi_syllables(s: str) -> List[str]:
    s = PUNCT_REGEX.sub(r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [] if not s else s.split(" ")

def levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            cost = 0 if ai == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[n][m]

def normalized_edit_distance(ref: str, hyp: str) -> float:
    ref_toks = split_vi_syllables(ref)
    hyp_toks = split_vi_syllables(hyp)
    if not ref_toks and not hyp_toks:
        return 0.0
    return levenshtein(ref_toks, hyp_toks) / max(1, len(ref_toks))

# =========================
# LOADER CHO CORRECTOR (PEFT/LoRA HỖ TRỢ)
# =========================
def _load_corrector_tokenizer_and_model(corr_dir: str):
    """
    Tự nhận biết thư mục corrector là LoRA adapter hay full model.
    Nếu là adapter (PEFT), resize embedding base theo vocab đã thêm (<err>, </err>) trước khi load.
    """
    adapter_cfg = os.path.join(corr_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base_name = cfg.get("base_model_name_or_path")
        if base_name is None:
            raise ValueError("Không tìm thấy base_model_name_or_path trong adapter_config.json")
        tok = AutoTokenizer.from_pretrained(base_name, use_fast=True)
        added = tok.add_tokens(["<err>", "</err>"], special_tokens=True)
        base = AutoModelForSeq2SeqLM.from_pretrained(base_name)
        if added and added > 0:
            base.resize_token_embeddings(len(tok))
        from peft import PeftModel  # pip install peft
        model = PeftModel.from_pretrained(base, corr_dir)
        return tok, model
    else:
        tok = AutoTokenizer.from_pretrained(corr_dir, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(corr_dir)
        return tok, model

# =========================
# TWO-STAGE CORRECTOR (INFER)
# =========================
class TwoStageCorrector:
    def __init__(self, det_dir: str, corr_dir: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Detector
        self.det_tok = AutoTokenizer.from_pretrained(det_dir, use_fast=True)
        self.det_model = AutoModelForTokenClassification.from_pretrained(det_dir).to(self.device).eval()
        # Corrector (support LoRA)
        self.corr_tok, self.corr_model = _load_corrector_tokenizer_and_model(corr_dir)
        self.corr_model.to(self.device).eval()

    @torch.no_grad()
    def detect_positions(self, text: str, threshold: float = 0.5) -> List[int]:
        tokens = split_vi_syllables(text)
        if not tokens:
            return []
        enc = self.det_tok(tokens, is_split_into_words=True,
                           return_tensors="pt", truncation=True, max_length=256)
        for k in enc:
            enc[k] = enc[k].to(self.device)
        logits = self.det_model(**enc).logits[0]
        probs = F.softmax(logits, dim=-1)[:, 1]  # class=1 là lỗi

        pos_probs = {}
        if getattr(self.det_tok, "is_fast", False):
            try:
                word_ids = enc.word_ids(batch_index=0)
                for i, wid in enumerate(word_ids):
                    if wid is not None and (i == 0 or wid != word_ids[i - 1]):
                        pos_probs[wid] = float(probs[i].detach().cpu())
            except Exception:
                n_words = len(tokens)
                seq_len = min(len(probs), n_words)
                for i in range(seq_len):
                    pos_probs[i] = float(probs[i].detach().cpu())
        else:
            n_words = len(tokens)
            seq_len = min(len(probs), n_words)
            for i in range(seq_len):
                pos_probs[i] = float(probs[i].detach().cpu())

        flagged = [i for i, p in pos_probs.items() if p >= threshold]
        return flagged

    def _build_tagged_src(self, text: str, flagged_positions: List[int], window: int = 0) -> str:
        toks = split_vi_syllables(text)
        if not flagged_positions:
            return text
        spans = []
        flagged_positions = sorted(flagged_positions)
        start = flagged_positions[0]
        prev = start
        for idx in flagged_positions[1:]:
            if idx - prev <= 1 + 2*window:
                prev = idx
            else:
                spans.append((max(0, start - window), min(len(toks), prev + 1 + window)))
                start = idx
                prev = idx
        spans.append((max(0, start - window), min(len(toks), prev + 1 + window)))
        out = []
        i = 0
        for l, r in spans:
            out.extend(toks[i:l]); out.append("<err>"); out.extend(toks[l:r]); out.append("</err>"); i = r
        out.extend(toks[i:])
        return " ".join(out)

    @torch.no_grad()
    def correct(self, text: str, det_thres: float = 0.5,
                beam_size: int = 6, max_new_tokens: int = 64, rerank_lambda: float = 0.6):
        flagged = self.detect_positions(text, threshold=det_thres)
        if not flagged:
            return {
                "input": text,
                "flagged_positions": [],
                "raw_correction": text,
                "final": text,
            }
        tagged = self._build_tagged_src(text, flagged, window=0)
        enc = self.corr_tok(tagged, return_tensors="pt", truncation=True, max_length=192).to(self.device)
        out = self.corr_model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            length_penalty=1.0,
            early_stopping=True,
        )
        cands = [self.corr_tok.decode(seq, skip_special_tokens=True) for seq in out.sequences]
        scores = out.sequences_scores.detach().cpu().tolist()

        # Rerank (ưu tiên ít chỉnh sửa so với input)
        best_idx, best_val = 0, -1e18
        for i, (cand, sc) in enumerate(zip(cands, scores)):
            ned = normalized_edit_distance(text, cand)
            val = sc - rerank_lambda * ned
            if val > best_val:
                best_val, best_idx = val, i
        final = cands[best_idx]
        return {
            "input": text,
            "flagged_positions": flagged,
            "raw_correction": final,
            "final": final,
        }

# =========================
# KHỞI TẠO APP + MODEL
# =========================
app = FastAPI(title="Vietnamese Spell Correction API", version="1.0.0")

# CORS mở cho tiện test
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Khởi tạo Gemini client
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# Khởi tạo corrector (nạp sớm để dùng lại)
_corrector = TwoStageCorrector(DET_DIR, CORR_DIR)

# Khởi tạo advanced corrector (v2)
# Lexicon path - tạo file này trong Phase 2
LEXICON_PATH = os.environ.get("LEXICON_PATH", "data/vi_lexicon.txt")
_advanced_corrector = AdvancedCorrector(
    detector_dir=DET_DIR,
    lexicon_path=LEXICON_PATH if os.path.exists(LEXICON_PATH) else None,
    use_word_segmenter=False,  # Set True if underthesea/pyvi installed
)

# Mount static (UI)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# =========================
# SCHEMAS
# =========================
class CorrectRequest(BaseModel):
    text: str
    det_thres: Optional[float] = 0.5
    beam_size: Optional[int] = 6
    max_new_tokens: Optional[int] = 64
    rerank_lambda: Optional[float] = 0.6

class CorrectResponse(BaseModel):
    input: str
    flagged_positions: List[int]
    raw_correction: str
    final: str

class OCRCorrectResponse(BaseModel):
    ocr_text: str
    corrected: CorrectResponse

class CorrectV2Request(BaseModel):
    text: str
    detection_threshold: Optional[float] = 0.5
    protect_patterns: Optional[bool] = True
    use_oov: Optional[bool] = True
    use_mlm: Optional[bool] = False  # Disabled by default (slow)
    use_classifier: Optional[bool] = True

class CorrectV2Response(BaseModel):
    input: str
    preprocessed: Dict
    detections: List[Dict]
    corrections: List[Dict]
    final: str

# =========================
# ENDPOINTS
# =========================
@app.get("/health")
def health():
    device = _corrector.device
    has_lexicon = _advanced_corrector.detector.lexicon is not None and len(_advanced_corrector.detector.lexicon) > 0
    return {
        "status": "ok",
        "device": device,
        "det_dir": DET_DIR,
        "corr_dir": CORR_DIR,
        "advanced_corrector": {
            "enabled": True,
            "has_lexicon": has_lexicon,
            "lexicon_size": len(_advanced_corrector.detector.lexicon) if has_lexicon else 0,
        }
    }

@app.post("/correct", response_model=CorrectResponse)
def correct(req: CorrectRequest):
    """Original two-stage corrector (v1)"""
    try:
        res = _corrector.correct(
            req.text,
            det_thres=req.det_thres or 0.5,
            beam_size=req.beam_size or 6,
            max_new_tokens=req.max_new_tokens or 64,
            rerank_lambda=req.rerank_lambda or 0.6,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Correction error: {e}")

@app.post("/correct_v2", response_model=CorrectV2Response)
def correct_v2(req: CorrectV2Request):
    """
    Advanced corrector (v2) with multi-detector and enhanced pipeline

    Features:
    - Unicode normalization
    - Multi-detector ensemble (OOV + masked-LM + token-classifier)
    - Pattern protection (URLs, emails, code)
    - Detailed detection scores

    Note: MLM detection is slow, disabled by default
    """
    try:
        # Temporarily override detector settings
        original_weights = (
            _advanced_corrector.detector.weight_oov,
            _advanced_corrector.detector.weight_mlm,
            _advanced_corrector.detector.weight_classifier,
        )

        # Adjust weights based on enabled detectors
        enabled_count = sum([req.use_oov, req.use_mlm, req.use_classifier])
        if enabled_count == 0:
            raise HTTPException(status_code=400, detail="At least one detector must be enabled")

        weight_per_detector = 1.0 / enabled_count
        _advanced_corrector.detector.weight_oov = weight_per_detector if req.use_oov else 0.0
        _advanced_corrector.detector.weight_mlm = weight_per_detector if req.use_mlm else 0.0
        _advanced_corrector.detector.weight_classifier = weight_per_detector if req.use_classifier else 0.0

        # Run correction
        res = _advanced_corrector.correct(
            req.text,
            detection_threshold=req.detection_threshold or 0.5,
            protect_patterns=req.protect_patterns,
        )

        # Restore original weights
        (_advanced_corrector.detector.weight_oov,
         _advanced_corrector.detector.weight_mlm,
         _advanced_corrector.detector.weight_classifier) = original_weights

        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced correction error: {e}")

@app.post("/ocr-correct", response_model=OCRCorrectResponse)
async def ocr_correct(
    image: UploadFile = File(...),
    det_thres: float = Form(0.5),
    beam_size: int = Form(6),
    max_new_tokens: int = Form(64),
    rerank_lambda: float = Form(0.6),
):
    # Đọc file ảnh
    try:
        content = await image.read()
        mime = image.content_type or "image/png"

        # Gọi Gemini để OCR (trả văn bản thuần)
        prompt = "Extract ALL readable text (OCR). Return clean plain text without extra commentary."
        result = gemini_model.generate_content([
            prompt,
            {"mime_type": mime, "data": content}
        ])
        # Lấy text từ response
        ocr_text = (result.text or "").strip()
        if not ocr_text:
            # fallback: thử thêm hướng dẫn
            prompt2 = "Perform OCR and output only the exact text content from the image."
            result2 = gemini_model.generate_content([
                prompt2,
                {"mime_type": mime, "data": content}
            ])
            ocr_text = (result2.text or "").strip()
        if not ocr_text:
            raise RuntimeError("Gemini OCR trả về rỗng.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OCR (Gemini) error: {e}")

    # Chạy sửa chính tả
    try:
        corr = _corrector.correct(
            ocr_text,
            det_thres=det_thres,
            beam_size=beam_size,
            max_new_tokens=max_new_tokens,
            rerank_lambda=rerank_lambda,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Correction error: {e}")

    return {"ocr_text": ocr_text, "corrected": corr}

# Route "/" để load UI
@app.get("/")
def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": f"UI chưa có. Hãy đặt index.html trong thư mục '{STATIC_DIR}/' hoặc đổi STATIC_DIR env."}
