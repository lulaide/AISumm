import json, jsonlines, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm

MODEL = "uer/roberta-tiny-wwm-chinese-cluecorpussmall"   # 轻量中文RoBERTa
MAX_LEN, BATCH, EPOCHS, LR = 128, 64, 3, 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def strip_spaces(s: str) -> str:
    # simple 格式里中文被空格分词了；去掉多余空格更贴近原句
    # 也可选择保留，Tokenizer 也能处理；这里做个轻清洗
    return s.replace(" ", "")

class SentDataset(Dataset):
    """
    把 simple.jsonl 的 (article: [sent...], label:[idx...]) 展开成逐句样本 (sent_text, y)
    """
    def __init__(self, jsonl_path):
        self.rows = []
        with jsonlines.open(jsonl_path) as reader:
            for obj in reader:
                sents = [strip_spaces(s) for s in obj["article"]]
                pos = set(obj.get("label", []))  # 0-based indices
                for i, sent in enumerate(sents):
                    y = 1 if i in pos else 0
                    # 丢弃特别短的句子，减少噪声（可选）
                    if len(sent) >= 4:
                        self.rows.append((sent, y))
        print(f"[LOAD] {jsonl_path}: {len(self.rows)} sentence samples")

    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

class Model(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(backbone_name)
        self.drop = nn.Dropout(0.2)
        self.clf  = nn.Linear(self.enc.config.hidden_size, 1)
    def forward(self, **enc):
        out = self.enc(**enc).last_hidden_state[:,0,:]  # [CLS]
        logit = self.clf(self.drop(out)).squeeze(-1)
        return logit

def make_loader(path, tokenizer, shuffle):
    ds = SentDataset(path)
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt")
        return {k:v for k,v in enc.items()}, torch.tensor(labels, dtype=torch.float32)
    return DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0)



def main():
    data_dir = Path("data/cnewsum")
    train_path = data_dir/"train.simple.label.jsonl"
    dev_path   = data_dir/"dev.simple.label.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = Model(MODEL).to(DEVICE)

    dl_tr = make_loader(train_path, tokenizer, shuffle=True)
    dl_va = make_loader(dev_path, tokenizer, shuffle=False)

    # 轻度处理类不平衡
    pos_weight = 1.5  # 可调：>1 提升召回
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(dl_tr)*EPOCHS
    sched = get_linear_schedule_with_warmup(optim, int(0.1*total_steps), total_steps)

    best_f1 = 0.0
    for ep in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"Epoch {ep}")
        for enc, y in pbar:
            enc = {k:v.to(DEVICE) for k,v in enc.items()}
            y = y.to(DEVICE)
            logit = model(**enc)
            loss = criterion(logit, y)
            optim.zero_grad(); loss.backward(); optim.step(); sched.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # eval
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for enc, y in dl_va:
                enc = {k:v.to(DEVICE) for k,v in enc.items()}
                p = torch.sigmoid(model(**enc)).cpu().numpy()
                preds.extend((p>0.5).astype(int).tolist())
                gts.extend(y.numpy().astype(int).tolist())
        f1 = f1_score(gts, preds, average="macro")
        print(f"[Val] macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "extractor_tiny_simple.pt")
            print("Saved extractor_tiny_simple.pt")

if __name__ == "__main__":
    main()
