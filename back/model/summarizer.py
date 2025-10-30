# summarizer.py
# 依赖: torch, transformers, numpy
# 用途: 传入长文本(字符串)或句子列表，返回抽取式摘要（Top-k 句）

import re
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

_CN_SPLIT_RE = re.compile(r'(?<=[。！？!?；;])')  # 简单中文分句

def split_cn(text: str) -> List[str]:
    if not text:
        return []
    sents = _CN_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sents if s and s.strip()]

def strip_spaces(s: str) -> str:
    return s.replace(" ", "").replace("\u3000", "").strip()

class _Clf(nn.Module):
    """Encoder + [CLS] 线性分类头，输出每句的重要性得分（logit）"""
    def __init__(self, name: str):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.drop = nn.Dropout(0.2)
        self.clf = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, **enc):
        out = self.enc(**enc).last_hidden_state[:, 0, :]  # [CLS]
        return self.clf(self.drop(out)).squeeze(-1)       # (B,)

class Summarizer:
    """
    用法:
        sm = Summarizer(ckpt_path="extractor_tiny_simple.pt")
        summary, (sents, scores) = sm.summarize_text(long_text, topk=5)
    """
    def __init__(
        self,
        model_name: str = "uer/roberta-tiny-wwm-chinese-cluecorpussmall",
        ckpt_path: Optional[str] = "model/sum_model/extractor_tiny_simple.pt",
        max_len: int = 128,
        device: Optional[str] = None,      # "cuda" | "cpu" | "mps"
        use_fast_tokenizer: bool = True,
        amp_dtype: Optional[torch.dtype] = None,  # torch.float16 / bfloat16 (仅在cuda上有效)
    ):
        # 设备
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        self.amp_dtype = amp_dtype if (self.device.type == "cuda") else None
        self.max_len = max_len

        # 组件
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)
        self.model = _Clf(model_name)
        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict(state, strict=False)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def _score_sents(self, sents: List[str], batch_size: int = 64) -> np.ndarray:
        if not sents:
            return np.zeros((0,), dtype=np.float32)

        out = np.zeros(len(sents), dtype=np.float32)
        i = 0
        while i < len(sents):
            batch = sents[i:i + batch_size]
            enc = self.tokenizer(
                batch, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            if self.device.type == "cuda" and self.amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    logits = self.model(**enc)
            else:
                logits = self.model(**enc)

            probs = torch.sigmoid(logits).float().cpu().numpy()
            out[i:i + len(batch)] = probs
            i += len(batch)
        return out

    def summarize_text(
        self,
        text: str,
        topk: int = 5,
        batch_size: int = 64,
        min_sentence_chars: int = 6,
        joiner: str = " ",
    ) -> Tuple[str, Tuple[List[str], List[float]]]:
        """输入原始长文本，返回(摘要文本, (选中句子列表, 概率列表))"""
        sents = split_cn(text)
        return self.summarize_list(
            sents,
            topk=topk,
            batch_size=batch_size,
            min_sentence_chars=min_sentence_chars,
            joiner=joiner,
        )

    def summarize_list(
        self,
        article_sents: List[str],
        topk: int = 5,
        batch_size: int = 64,
        min_sentence_chars: int = 6,
        joiner: str = " ",
    ) -> Tuple[str, Tuple[List[str], List[float]]]:
        """输入句子列表，返回(摘要文本, (选中句子列表, 概率列表))"""
        sents = [strip_spaces(s) for s in article_sents if s and strip_spaces(s)]
        sents = [s for s in sents if len(s) >= min_sentence_chars]
        if not sents:
            return "", ([], [])

        scores = self._score_sents(sents, batch_size=batch_size)
        k = min(max(1, topk), len(sents))
        top_idx = np.argsort(-scores)[:k]
        top_idx = sorted(top_idx.tolist())  # 保持原文顺序

        picked_sents = [sents[i] for i in top_idx]
        picked_scores = [float(scores[i]) for i in top_idx]
        summary_text = joiner.join(picked_sents)
        return summary_text, (picked_sents, picked_scores)
