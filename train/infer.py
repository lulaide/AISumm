import re, torch, numpy as np, json
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

CKPT  = "extractor_tiny_simple.pt"
MODEL = "uer/roberta-tiny-wwm-chinese-cluecorpussmall"
MAX_LEN = 128

def strip_spaces(s): return s.replace(" ", "")
def split_cn(text):
    sents = re.split(r'(?<=[。！？!?；;])', text.strip())
    return [s.strip() for s in sents if s.strip()]

class Clf(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.drop = nn.Dropout(0.2)
        self.clf = nn.Linear(self.enc.config.hidden_size, 1)
    def forward(self, **enc):
        out = self.enc(**enc).last_hidden_state[:,0,:]
        return self.clf(self.drop(out)).squeeze(-1)

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = Clf(MODEL)
state = torch.load(CKPT, map_location="cpu")
model.load_state_dict(state); model.eval()

def score_sents(sents):
    sents = [strip_spaces(s) for s in sents if s.strip()]
    enc = tokenizer(sents, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    with torch.no_grad():
        p = torch.sigmoid(model(**enc)).squeeze(-1).cpu().numpy()
    return p.tolist()

def extract_summary_from_text(text, topk=3):
    sents = split_cn(text)
    if not sents: return [], []
    probs = np.array(score_sents(sents))
    idx = np.argsort(-probs)[:topk]
    idx = sorted(idx.tolist())
    return [sents[i] for i in idx], [float(probs[i]) for i in idx]

def extract_from_article_list(article_list, topk=3):
    probs = np.array(score_sents(article_list))
    idx = np.argsort(-probs)[:topk]
    idx = sorted(idx.tolist())
    return [article_list[i] for i in idx], [float(probs[i]) for i in idx]

if __name__ == "__main__":
    # 示例：用你贴的这条样本
    article = ["13岁 的 小敏 离开 家乡 怒江 福贡 来到 昆明 打工 , 经 人 介绍 , 她 来到 在 西山区 一 足疗 店 工作 。",
               "表面 上 , 她 是 帮 人 做 足疗 , 实际上 则 从事 着 卖淫 的 事 。",
               "5日 晚 11点 多 , 西山 公安 分局 东风 派出所 治安 行动 组 吴 组长 接到 匿名 举报 , 将 这 一 卖淫 窝点 查获 。",
               "被 查 时 , 她 刚 “ 服务 ” 完 一名 36 岁 的 客人 。"]
    sents, scores = extract_from_article_list(article, topk=2)
    print(list(zip(sents, scores)))
