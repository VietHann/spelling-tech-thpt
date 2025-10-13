# vi_spell_pipeline_plus.py
# -*- coding: utf-8 -*-
"""
Pipeline phát hiện & sửa lỗi chính tả tiếng Việt (2-stage, bản nâng cấp):
- Stage 1 (Detection): PhoBERT token-classification + FocalLoss + class weighting + threshold calibration
- Stage 2 (Correction): BARTpho/ViT5 tag-guided seq2seq (chèn <err>...</err>) + label smoothing + beam rerank
- Tùy chọn: LoRA (PEFT) cho corrector, freeze encoder để train nhanh
- End-to-end: decode có TAG + merge bảo thủ quanh vị trí flag
- Đánh giá: EM & NED trên VSEC test, auto-calibrate ngưỡng theo dev

Dữ liệu:
  * Correction train: ShynBui/Vietnamese_spelling_error (error_text -> text)
  * Detection train/val/test + End-to-end test: VSEC (text -> corrected_text, + nhãn syllable-level)
"""

import os

# --- TẮT HOÀN TOÀN Dynamo/compile NGAY TỪ ĐẦU TRƯỚC KHI IMPORT TORCH ---
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("ACCELERATE_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("PYTORCH_JIT", "0")
os.environ.setdefault("TORCH_LOGS", "")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
if "TORCH_LOGS" in os.environ:
    del os.environ["TORCH_LOGS"]
# ------------------------------------------------------------------------

import re
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from datasets import load_dataset, DatasetDict

import torch
from torch.nn import functional as F

# Tắt thêm ở runtime để chắn mọi đường
try:
    import torch._dynamo as dynamo
    dynamo.disable()
    dynamo.config.suppress_errors = True
except Exception:
    pass

from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, Trainer, TrainingArguments,
    AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from inspect import signature

# -----------------------------
# Tiện ích chung
# -----------------------------

SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

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
    if len(ref_toks) == 0 and len(hyp_toks) == 0:
        return 0.0
    return levenshtein(ref_toks, hyp_toks) / max(1, len(ref_toks))

# -----------------------------
# VSEC splits (Detection + E2E)
# -----------------------------

def load_vsec_detection_splits(test_size=0.1, val_size=0.1, seed=SEED) -> DatasetDict:
    vsec = load_dataset("nguyenthanhasia/vsec-vietnamese-spell-correction")["train"]

    def to_tokens_labels(ex):
        anns = ex["syllable_annotations"]
        tokens = [a["syllable"] for a in anns]
        labels = [0 if a["is_correct"] else 1 for a in anns]
        return {"tokens": tokens, "labels": labels,
                "raw_text": ex["text"], "corrected_text": ex["corrected_text"]}

    vsec2 = vsec.map(to_tokens_labels, remove_columns=vsec.column_names)
    tmp = vsec2.train_test_split(test_size=test_size, seed=seed)
    train_val = tmp["train"].train_test_split(test_size=val_size/(1-test_size), seed=seed)
    return DatasetDict(train=train_val["train"], validation=train_val["test"], test=tmp["test"])

def prepare_detection_encodings(ds: DatasetDict, tok_name="vinai/phobert-base", max_len=256):
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    label_all_tokens = False

    def tokenize_batch_fast(batch):
        enc = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=max_len)
        all_labels = []
        for i, labels in enumerate(batch["labels"]):
            word_ids = enc.word_ids(batch_index=i)
            prev_wid = None
            label_ids = []
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                elif wid != prev_wid:
                    label_ids.append(labels[wid])
                else:
                    label_ids.append(labels[wid] if label_all_tokens else -100)
                prev_wid = wid
            all_labels.append(label_ids)
        enc["labels"] = all_labels
        return enc

    def tokenize_batch_slow(batch):
        texts = [" ".join(toks) for toks in batch["tokens"]]
        enc = tokenizer(texts, truncation=True, max_length=max_len)
        all_labels = []
        for toks, labs, input_ids in zip(batch["tokens"], batch["labels"], enc["input_ids"]):
            sub_lens = [len(tokenizer.tokenize(w)) for w in toks]
            flat = []
            for lab, k in zip(labs, sub_lens):
                flat.append(lab)
                if k > 1:
                    flat.extend(([lab] if label_all_tokens else [-100]) * (k-1))
            max_inner = max_len - 2
            inner = flat[:max_inner]
            label_ids = [-100] + inner + [-100]
            if len(label_ids) < len(input_ids):
                label_ids += [-100] * (len(input_ids) - len(label_ids))
            else:
                label_ids = label_ids[:len(input_ids)]
            all_labels.append(label_ids)
        enc["labels"] = all_labels
        return enc

    mapper = tokenize_batch_fast if getattr(tokenizer, "is_fast", False) else tokenize_batch_slow
    if not getattr(tokenizer, "is_fast", False):
        print("[prepare_detection_encodings] Warning: tokenizer is SLOW; fallback manual alignment.")

    cols_to_remove = ["tokens", "labels", "raw_text", "corrected_text"]
    enc_train = ds["train"].map(mapper, batched=True, remove_columns=cols_to_remove)
    enc_val = ds["validation"].map(mapper, batched=True, remove_columns=cols_to_remove)
    enc_test = ds["test"].map(mapper, batched=True, remove_columns=cols_to_remove)
    return tokenizer, DatasetDict(train=enc_train, validation=enc_val, test=enc_test)

# -----------------------------
# Correction data (ShynBui)
# -----------------------------

def load_correction_data(max_train_samples: int = None) -> DatasetDict:
    ds = load_dataset("ShynBui/Vietnamese_spelling_error")["train"]
    def to_pair(ex): return {"src": ex["error_text"], "tgt": ex["text"]}
    ds2 = ds.map(to_pair, remove_columns=ds.column_names)
    if max_train_samples:
        ds2 = ds2.select(range(min(max_train_samples, len(ds2))))
    return DatasetDict(train=ds2)

# -----------------------------
# Diff & tag span lỗi cho seq2seq
# -----------------------------

def diff_error_spans(src: str, tgt: str, window_merge: int = 0) -> List[Tuple[int, int]]:
    import difflib
    s_toks = split_vi_syllables(src)
    t_toks = split_vi_syllables(tgt)
    sm = difflib.SequenceMatcher(a=s_toks, b=t_toks)
    spans = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal" and i1 < i2:
            spans.append((i1, i2))
    if window_merge > 0 and spans:
        merged, cur_l, cur_r = [], spans[0][0], spans[0][1]
        for l, r in spans[1:]:
            if l - cur_r <= window_merge:
                cur_r = r
            else:
                merged.append((cur_l, cur_r))
                cur_l, cur_r = l, r
        merged.append((cur_l, cur_r))
        return merged
    return spans

def insert_err_tags_tokens(tokens: List[str], spans: List[Tuple[int, int]],
                           tag_open="<err>", tag_close="</err>") -> List[str]:
    if not spans: return tokens
    out, i = [], 0
    for l, r in sorted(spans):
        out.extend(tokens[i:l]); out.append(tag_open); out.extend(tokens[l:r]); out.append(tag_close); i = r
    out.extend(tokens[i:])
    return out

def tag_source_by_alignment(src: str, tgt: str) -> str:
    s_toks = split_vi_syllables(src)
    spans = diff_error_spans(src, tgt)
    tagged = insert_err_tags_tokens(s_toks, spans)
    return " ".join(tagged)

# -----------------------------
# Chuẩn bị encodings cho corrector (tag-guided)
# -----------------------------

def _encode_split_with_tags(ds_split, tok, max_src_len, max_tgt_len, tag_ratio):
    def make_src(batch):
        srcs, tgts = [], []
        for s, t in zip(batch["src"], batch["tgt"]):
            srcs.append(tag_source_by_alignment(s, t) if random.random() < tag_ratio else s)
            tgts.append(t)
        model_inputs = tok(srcs, truncation=True, max_length=max_src_len)
        try:
            with tok.as_target_tokenizer():
                labels = tok(tgts, truncation=True, max_length=max_tgt_len)
        except AttributeError:
            labels = tok(text_target=tgts, truncation=True, max_length=max_tgt_len)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return ds_split.map(make_src, batched=True, remove_columns=ds_split.column_names)

def prepare_correction_encodings_tagged(ds: DatasetDict,
                                        model_name="vinai/bartpho-syllable",
                                        max_src_len=192, max_tgt_len=192,
                                        tag_ratio=0.7,
                                        special_tags=("<err>", "</err>")):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    added = tok.add_tokens(list(special_tags), special_tokens=True)
    enc = {"train": _encode_split_with_tags(ds["train"], tok, max_src_len, max_tgt_len, tag_ratio)}
    if "validation" in ds:
        enc["validation"] = _encode_split_with_tags(ds["validation"], tok, max_src_len, max_tgt_len, tag_ratio)
    return tok, DatasetDict(**enc), added

# -----------------------------
# Focal loss Trainer cho detector
# -----------------------------

class FocalTokenTrainer(Trainer):
    def __init__(self, *args, class_weights=None, gamma=2.0, alpha_pos=0.75, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha_pos = alpha_pos
        if class_weights is None:
            self.class_weights = None
        else:
            import torch
            if isinstance(class_weights, (list, tuple, np.ndarray)):
                class_weights = torch.tensor(class_weights, dtype=torch.float)
            self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        import torch
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        C = logits.size(-1)
        loss_ce = F.cross_entropy(
            logits.view(-1, C), labels.view(-1),
            reduction='none',
            ignore_index=-100,
            weight=(self.class_weights.to(logits.device) if self.class_weights is not None else None)
        )
        pt = torch.exp(-loss_ce)
        with torch.no_grad():
            y = labels.view(-1)
            alpha = torch.ones_like(pt)
            alpha[y == 1] = self.alpha_pos
        loss = (alpha * ((1 - pt) ** self.gamma) * loss_ce)
        mask_valid = labels.view(-1) != -100
        loss = loss[mask_valid].mean()
        return (loss, outputs) if return_outputs else loss

# -----------------------------
# Train DETECTOR (PhoBERT + Focal)
# -----------------------------

def compute_class_weights_from_ds(ds_word_level: DatasetDict) -> Tuple[float, float]:
    pos = 0; neg = 0
    for labs in ds_word_level["train"]["labels"]:
        for y in labs:
            if y == 1: pos += 1
            elif y == 0: neg += 1
    if pos == 0: return (1.0, 1.0)
    w_neg = 1.0
    w_pos = max(1.0, neg / pos)
    return (w_neg, float(w_pos))

def train_detector(output_dir="outputs/detector",
                   base_model="vinai/phobert-base",
                   epochs=3, lr=2e-5, bsz=16, max_len=256,
                   focal_gamma=2.0, alpha_pos=0.75,
                   fp16=False, bf16=True,
                   save_strategy="epoch", save_total_limit=1,
                   group_by_length=False):
    os.makedirs(output_dir, exist_ok=True)
    ds_word = load_vsec_detection_splits()
    det_tokenizer, enc = prepare_detection_encodings(ds_word, tok_name=base_model, max_len=max_len)
    model = AutoModelForTokenClassification.from_pretrained(base_model, num_labels=2)
    class_weights = compute_class_weights_from_ds(ds_word)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        labels = p.label_ids
        y_true, y_pred = [], []
        for yt, yp in zip(labels, preds):
            for t, p_ in zip(yt, yp):
                if t != -100:
                    y_true.append(int(t)); y_pred.append(int(p_))
        tp = sum(1 for t, p_ in zip(y_true, y_pred) if t==1 and p_==1)
        fp = sum(1 for t, p_ in zip(y_true, y_pred) if t==0 and p_==1)
        fn = sum(1 for t, p_ in zip(y_true, y_pred) if t==1 and p_==0)
        prec = tp / (tp+fp+1e-12); rec = tp / (tp+fn+1e-12)
        f1 = 2*prec*rec/(prec+rec+1e-12)
        acc = sum(1 for t,p_ in zip(y_true,y_pred) if t==p_) / max(1,len(y_true))
        return {"precision_err": prec, "recall_err": rec, "f1_err": f1, "acc_all": acc}

    # Tạo args (eval theo epoch cho detector)
    sig_tr = signature(TrainingArguments.__init__)
    kwargs = dict(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=bsz,
        per_device_eval_batch_size=bsz,
        num_train_epochs=epochs,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_err",
        greater_is_better=True,
        fp16=fp16, bf16=bf16,
        gradient_accumulation_steps=2,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
        group_by_length=group_by_length,
    )
    # tên tham số eval cho bản cũ/mới
    if "eval_strategy" in sig_tr.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    args = TrainingArguments(**kwargs)

    data_collator = DataCollatorForTokenClassification(det_tokenizer)
    trainer = FocalTokenTrainer(
        model=model,
        args=args,
        train_dataset=enc["train"],
        eval_dataset=enc["validation"],
        tokenizer=det_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        gamma=focal_gamma,
        alpha_pos=alpha_pos
    )
    trainer.train()
    trainer.save_model(output_dir)
    det_tokenizer.save_pretrained(output_dir)

    test_metrics = trainer.evaluate(enc["test"])
    print("== Detection Test Metrics ==", test_metrics)
    return output_dir

# -----------------------------
# Train CORRECTOR (tag-guided seq2seq, LoRA optional) — EVAL THEO STEPS
# -----------------------------

def train_corrector(output_dir="outputs/corrector",
                    base_model="vinai/bartpho-syllable",
                    epochs=3, lr=2e-5, bsz=8,
                    max_src_len=192, max_tgt_len=192,
                    tag_ratio=0.7,
                    label_smoothing=0.1,
                    max_train_samples=None,
                    fp16=False, bf16=True,
                    # speed/IO
                    save_strategy="steps", save_total_limit=2, save_steps=1000,
                    group_by_length=True,
                    # eval theo steps (CLI: --eval_strategy/--eval_steps)
                    eval_strategy="steps", eval_steps=1000, load_best=False, corr_val_ratio=0.01,
                    # speed options
                    freeze_encoder=False,
                    # LoRA options
                    use_lora=False, lora_r=8, lora_alpha=32, lora_dropout=0.05,
                    lora_target="q_proj,k_proj,v_proj,o_proj,fc1,fc2",
                    lora_merge=False,
                    # grad ckpt
                    grad_ckpt=False):

    os.makedirs(output_dir, exist_ok=True)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # 1) Dữ liệu + split val nếu cần đánh giá
    ds_full = load_correction_data(max_train_samples=max_train_samples)
    if eval_strategy != "no":
        split = ds_full["train"].train_test_split(test_size=corr_val_ratio, seed=SEED)
        ds = DatasetDict(train=split["train"], validation=split["test"])
    else:
        ds = DatasetDict(train=ds_full["train"])

    # 2) Tokenizer + encode (có TAG cho cả train & val)
    tok, enc, added_tokens = prepare_correction_encodings_tagged(
        ds, model_name=base_model, max_src_len=max_src_len, max_tgt_len=max_tgt_len, tag_ratio=tag_ratio
    )

    # 3) Model
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    if added_tokens and added_tokens > 0:
        model.resize_token_embeddings(len(tok))

    # Freeze encoder (tùy chọn)
    if freeze_encoder:
        enc_mod = None
        if hasattr(model, "get_encoder"):
            enc_mod = model.get_encoder()
        else:
            enc_mod = getattr(getattr(model, "model", model), "encoder", None)
        if enc_mod is not None:
            for p in enc_mod.parameters():
                p.requires_grad = False
            print("[Corrector] Encoder is frozen.")

    # LoRA
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("PEFT chưa cài. Vui lòng: pip install peft bitsandbytes")
        target = [x.strip() for x in lora_target.split(",") if x.strip()]
        peft_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias="none", task_type="SEQ_2_SEQ_LM", target_modules=target
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    # Gradient checkpointing
    if grad_ckpt:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        print("[Corrector] Gradient checkpointing ENABLED.")
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        print("[Corrector] Gradient checkpointing DISABLED.")

    # 4) Seq2SeqTrainingArguments (tự chọn tên tham số eval_* tương thích)
    sig_s2s = signature(Seq2SeqTrainingArguments.__init__)
    kwargs = dict(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=bsz,
        num_train_epochs=epochs,
        # save theo steps
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best,
        metric_for_best_model="loss",
        greater_is_better=False,
        # common
        logging_steps=200,
        bf16=bf16, fp16=fp16,
        gradient_accumulation_steps=2,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        predict_with_generate=False,
        torch_compile=False,
        report_to="none",
        label_smoothing_factor=label_smoothing,
        group_by_length=group_by_length,
    )
    # tên tham số eval theo version
    if "eval_strategy" in sig_s2s.parameters:
        kwargs["eval_strategy"] = eval_strategy
        kwargs["eval_steps"] = eval_steps
    else:
        kwargs["evaluation_strategy"] = eval_strategy
        kwargs["eval_steps"] = eval_steps

    args = Seq2SeqTrainingArguments(**kwargs)

    # 5) Train
    data_collator = DataCollatorForSeq2Seq(tok, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=enc["train"],
        eval_dataset=enc.get("validation"),
        tokenizer=tok,
        data_collator=data_collator,
    )
    trainer.train()

    # 6) Lưu
    if use_lora and lora_merge:
        try:
            merged = model.merge_and_unload()
            merged.save_pretrained(output_dir)
            tok.save_pretrained(output_dir)
            print("== Correction model (LoRA merged) saved to", output_dir)
        except Exception as e:
            print("[Warn] merge_and_unload thất bại, lưu adapter như PEFT. Err:", e)
            trainer.save_model(output_dir); tok.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir); tok.save_pretrained(output_dir)

    print("== Correction model saved to", output_dir)
    return output_dir

# -----------------------------
# Suy diễn end-to-end
# -----------------------------
def _load_corrector_tokenizer_and_model(corr_dir: str):
    """Tự nhận biết thư mục corrector là LoRA adapter hay full model.
    Nếu là adapter (PEFT), đảm bảo resize embedding base theo số token đã thêm khi fine-tune."""
    adapter_cfg = os.path.join(corr_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        # PEFT adapter: cần base model + resize vocab cho khớp
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base_name = cfg.get("base_model_name_or_path")
        if base_name is None:
            raise ValueError("Không tìm thấy base_model_name_or_path trong adapter_config.json")

        # 1) Tokenizer base + thêm lại special tags như lúc train
        tok = AutoTokenizer.from_pretrained(base_name, use_fast=True)
        added = tok.add_tokens(["<err>", "</err>"], special_tokens=True)

        # 2) Nạp base model và resize embedding khớp vocab mới
        base = AutoModelForSeq2SeqLM.from_pretrained(base_name)
        if added and added > 0:
            base.resize_token_embeddings(len(tok))

        # 3) Nạp adapter LoRA
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, corr_dir)
        return tok, model
    else:
        tok = AutoTokenizer.from_pretrained(corr_dir, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(corr_dir)
        return tok, model

@dataclass
class TwoStageCorrector:
    det_dir: str
    corr_dir: str
    use_err_tags: bool = True
    beam_size: int = 6
    max_new_tokens: int = 64
    rerank_lambda: float = 0.6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.det_tok = AutoTokenizer.from_pretrained(self.det_dir, use_fast=True)
        self.det_model = AutoModelForTokenClassification.from_pretrained(self.det_dir).to(self.device).eval()
        self.corr_tok, self.corr_model = _load_corrector_tokenizer_and_model(self.corr_dir)
        self.corr_model.to(self.device).eval()

    @torch.no_grad()
    def detect_positions(self, text: str, threshold: float = 0.5) -> List[int]:
        tokens = split_vi_syllables(text)
        if not tokens: return []
        enc = self.det_tok(tokens, is_split_into_words=True,
                           return_tensors="pt", truncation=True, max_length=256)
        for k in enc: enc[k] = enc[k].to(self.device)
        logits = self.det_model(**enc).logits[0]
        probs = F.softmax(logits, dim=-1)[:, 1]
        pos_probs = {}
        if getattr(self.det_tok, "is_fast", False):
            try:
                word_ids = enc.word_ids(batch_index=0)
                for i, wid in enumerate(word_ids):
                    if wid is not None and (i == 0 or wid != word_ids[i - 1]):
                        pos_probs[wid] = float(probs[i].detach().cpu())
            except Exception:
                n_words = len(tokens); seq_len = min(len(probs), n_words)
                for i in range(seq_len): pos_probs[i] = float(probs[i].detach().cpu())
        else:
            n_words = len(tokens); seq_len = min(len(probs), n_words)
            for i in range(seq_len): pos_probs[i] = float(probs[i].detach().cpu())
        return [i for i, p in pos_probs.items() if p >= threshold]

    def _build_tagged_src(self, text: str, flagged_positions: List[int], window: int = 0) -> str:
        toks = split_vi_syllables(text)
        if not flagged_positions: return text
        spans = []
        flagged_positions = sorted(flagged_positions)
        start = flagged_positions[0]; prev = start
        for idx in flagged_positions[1:]:
            if idx - prev <= 1 + 2*window:
                prev = idx
            else:
                spans.append((max(0, start - window), min(len(toks), prev + 1 + window)))
                start = idx; prev = idx
        spans.append((max(0, start - window), min(len(toks), prev + 1 + window)))
        tagged_toks = insert_err_tags_tokens(toks, spans)
        return " ".join(tagged_toks)

    @torch.no_grad()
    def correct_with_rerank(self, original_text: str, det_positions: List[int]) -> Tuple[str, str]:
        if self.use_err_tags and det_positions:
            inp = self._build_tagged_src(original_text, det_positions, window=0)
        else:
            inp = original_text
        enc = self.corr_tok(inp, return_tensors="pt", truncation=True, max_length=192).to(self.device)
        gen_out = self.corr_model.generate(
            **enc, max_new_tokens=self.max_new_tokens, num_beams=self.beam_size,
            num_return_sequences=self.beam_size, output_scores=True, return_dict_in_generate=True,
            do_sample=False, length_penalty=1.0, early_stopping=True,
        )
        cands = [self.corr_tok.decode(seq, skip_special_tokens=True) for seq in gen_out.sequences]
        scores = gen_out.sequences_scores.detach().cpu().numpy().tolist()
        best_idx, best_val = 0, -1e18
        for i, (cand, sc) in enumerate(zip(cands, scores)):
            ned = normalized_edit_distance(original_text, cand)
            val = sc - self.rerank_lambda * ned
            if val > best_val:
                best_val, best_idx = val, i
        return cands[best_idx], inp

    def conservative_merge(self, original: str, corrected: str,
                           flagged_positions: List[int], window: int = 1) -> str:
        orig = split_vi_syllables(original)
        corr = split_vi_syllables(corrected)
        if not flagged_positions: return original
        allowed = set()
        for p in flagged_positions:
            for q in range(max(0, p-window), min(len(orig), p+window+1)):
                allowed.add(q)
        import difflib
        sm = difflib.SequenceMatcher(a=orig, b=corr)
        result = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                result.extend(orig[i1:i2])
            elif tag in ("replace", "delete", "insert"):
                if any(idx in allowed for idx in range(i1, i2)):
                    result.extend(corr[j1:j2])
                else:
                    result.extend(orig[i1:i2])
        out = " ".join(result)
        out = (out.replace(" ,", ",").replace(" .", ".").replace(" !", "!")
                   .replace(" ?", "?").replace(" ;", ";").replace(" :", ":"))
        return out

    def predict(self, text: str, det_thres: float = 0.5) -> Dict:
        flagged = self.detect_positions(text, threshold=det_thres)
        if not flagged:
            return {"input": text, "flagged_positions": [], "raw_correction": text, "final": text}
        best_corr, _inp = self.correct_with_rerank(text, flagged)
        final = self.conservative_merge(text, best_corr, flagged, window=1)
        return {"input": text, "flagged_positions": flagged, "raw_correction": best_corr, "final": final}

# -----------------------------
# Hiệu chuẩn & Đánh giá E2E
# -----------------------------

def calibrate_threshold_end2end(det_dir: str, corr_dir: str,
                                grid = (0.3, 0.4, 0.5, 0.6, 0.7)) -> float:
    ds = load_vsec_detection_splits()
    dev = ds["validation"]
    corrector = TwoStageCorrector(det_dir, corr_dir)
    best_th, best_em, best_ned = 0.5, -1.0, 1e9
    for th in grid:
        em, ned_sum = 0, 0.0
        for ex in dev:
            inp = ex["raw_text"]; tgt = ex["corrected_text"]
            pred = corrector.predict(inp, det_thres=th)["final"]
            if pred.strip() == tgt.strip(): em += 1
            ned_sum += normalized_edit_distance(tgt, pred)
        em_rate = em / max(1, len(dev)); ned_avg = ned_sum / max(1, len(dev))
        if em_rate > best_em or (abs(em_rate - best_em) < 1e-9 and ned_avg < best_ned):
            best_em, best_ned, best_th = em_rate, ned_avg, th
        print(f"[Calib] th={th:.2f} -> EM={em_rate:.4f}, NED={ned_avg:.4f}")
    print(f"[Calib] Best threshold={best_th:.2f} (EM={best_em:.4f}, NED={best_ned:.4f})")
    return best_th

def evaluate_end2end(det_dir="outputs/detector", corr_dir="outputs/corrector",
                     det_thres=0.5):
    ds = load_vsec_detection_splits()
    test = ds["test"]
    corrector = TwoStageCorrector(det_dir, corr_dir)
    total, em, ned_sum = len(test), 0, 0.0
    for ex in test:
        inp = ex["raw_text"]; tgt = ex["corrected_text"]
        pred = corrector.predict(inp, det_thres=det_thres)["final"]
        if pred.strip() == tgt.strip(): em += 1
        ned_sum += normalized_edit_distance(tgt, pred)
    print(f"[End2End] Exact match: {em/total:.4f} ({em}/{total})")
    print(f"[End2End] Avg NED (lower better): {ned_sum/total:.4f}")

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train_detector", action="store_true")
    parser.add_argument("--do_train_corrector", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_calibrate", action="store_true")

    parser.add_argument("--det_model", type=str, default="vinai/phobert-base")
    parser.add_argument("--corr_model", type=str, default="vinai/bartpho-syllable")
    parser.add_argument("--det_out", type=str, default="outputs/detector")
    parser.add_argument("--corr_out", type=str, default="outputs/corrector")
    parser.add_argument("--det_epochs", type=int, default=3)
    parser.add_argument("--corr_epochs", type=int, default=3)
    parser.add_argument("--det_bsz", type=int, default=16)
    parser.add_argument("--corr_bsz", type=int, default=8)
    parser.add_argument("--corr_max_train_samples", type=int, default=None)
    parser.add_argument("--det_thres", type=float, default=0.5)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--alpha_pos", type=float, default=0.75)
    parser.add_argument("--tag_ratio", type=float, default=0.7)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--beam_size", type=int, default=6)
    parser.add_argument("--rerank_lambda", type=float, default=0.6)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    # speed/IO
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["no", "epoch", "steps"])
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--group_by_length", action="store_true")

    # Corrector eval theo steps (dùng 'eval_strategy' ở CLI)
    parser.add_argument("--eval_strategy", type=str, default="steps", choices=["no", "epoch", "steps"])
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--load_best", action="store_true")
    parser.add_argument("--corr_val_ratio", type=float, default=0.01)

    # speed
    parser.add_argument("--freeze_encoder", action="store_true")

    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target", type=str, default="q_proj,k_proj,v_proj,o_proj,fc1,fc2")
    parser.add_argument("--lora_merge", action="store_true")

    # learning rate & grad ckpt
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad_ckpt", action="store_true")

    args = parser.parse_args()

    # Train DETECTOR
    if args.do_train_detector:
        train_detector(output_dir=args.det_out, base_model=args.det_model,
                       epochs=args.det_epochs, bsz=args.det_bsz,
                       focal_gamma=args.focal_gamma, alpha_pos=args.alpha_pos,
                       fp16=args.fp16, bf16=args.bf16,
                       save_strategy=args.save_strategy,
                       save_total_limit=args.save_total_limit,
                       group_by_length=args.group_by_length)

    # Train CORRECTOR (eval theo steps)
    if args.do_train_corrector:
        lr_use = args.lr if args.lr is not None else 2e-5
        train_corrector(output_dir=args.corr_out, base_model=args.corr_model,
                        epochs=args.corr_epochs, bsz=args.corr_bsz,
                        max_train_samples=args.corr_max_train_samples,
                        tag_ratio=args.tag_ratio, label_smoothing=args.label_smoothing,
                        fp16=args.fp16, bf16=args.bf16,
                        save_strategy=args.save_strategy, save_total_limit=args.save_total_limit,
                        save_steps=args.save_steps, group_by_length=args.group_by_length,
                        eval_strategy=args.eval_strategy, eval_steps=args.eval_steps,
                        load_best=args.load_best, corr_val_ratio=args.corr_val_ratio,
                        freeze_encoder=args.freeze_encoder,
                        use_lora=args.use_lora, lora_r=args.lora_r,
                        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        lora_target=args.lora_target, lora_merge=args.lora_merge,
                        lr=lr_use, grad_ckpt=args.grad_ckpt)

    # Calibrate threshold
    if args.do_calibrate:
        best_th = calibrate_threshold_end2end(args.det_out, args.corr_out,
                                              grid=(0.3,0.4,0.5,0.6,0.7))
        print(f"[Main] Suggested det_thres = {best_th:.2f}")

    # Evaluate E2E
    if args.do_eval:
        corrector = TwoStageCorrector(args.det_out, args.corr_out,
                                      use_err_tags=True, beam_size=args.beam_size,
                                      rerank_lambda=args.rerank_lambda)
        evaluate_end2end(det_dir=args.det_out, corr_dir=args.corr_out, det_thres=args.det_thres)

if __name__ == "__main__":
    main()
