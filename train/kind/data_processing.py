import os
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, BERT_MODEL_NAME


def process_data():
    """完整数据处理流程"""
    # 1. 使用配置文件中的路径
    train_path = TRAIN_DATA_PATH
    test_path = TEST_DATA_PATH

    # 检查文件是否存在
    for path in [train_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件：{path}，请检查路径！")

    # 2. 读取训练集和测试集
    train_df = pd.read_csv(
        train_path,
        sep="\t",
        header=None,
        names=["text", "label"],
        encoding="utf-8"
    )

    test_df = pd.read_csv(
        test_path,
        sep="\t",
        header=None,
        names=["text", "label"],
        encoding="utf-8"
    )

    # 3. 拆分验证集
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df["label"]
    )

    # 4. 转换为DatasetDict
    raw_ds = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })

    # 5. 加载分词器（从Hugging Face下载）
    print(f"正在从Hugging Face下载模型: {BERT_MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 6. 分词函数
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length"
        )

    # 7. 多线程分词
    tokenized_ds = raw_ds.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )

    # 8. 转换为PyTorch格式
    tokenized_ds = tokenized_ds.with_format(
        "torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    # 9. 调试打印
    print(f"训练集样本数：{len(tokenized_ds['train'])}")
    print(f"验证集样本数：{len(tokenized_ds['validation'])}")
    print(f"测试集样本数：{len(tokenized_ds['test'])}")
    print("训练集标签示例：", tokenized_ds["train"]["label"][:5])

    return tokenized_ds, tokenizer
