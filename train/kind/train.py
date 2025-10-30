import torch
import os
from transformers import (
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from data_processing import process_data  # 导入你的数据处理函数
from config import MODEL_SAVE_DIR, BERT_MODEL_NAME  # 从配置文件导入模型保存路径


def compute_metrics(eval_pred):
    """手动计算准确率，不依赖外部库"""
    logits, labels = eval_pred  # logits是模型输出的预测分数，labels是真实标签
    predictions = logits.argmax(axis=-1)  # 取分数最高的类别作为预测结果
    correct = (predictions == labels).sum().item()  # 计算正确的样本数
    total = labels.shape[0]  # 总样本数
    accuracy = correct / total  # 准确率 = 正确数 / 总数
    return {"accuracy": accuracy}  # 返回字典格式，与Trainer要求一致

def main():
    # 1. 数据处理：加载并分词数据
    tokenized_ds, tokenizer = process_data()
    print("数据准备完成，开始加载模型...")

    # 2. 加载模型（10个类别）- 从Hugging Face下载
    num_labels = 10  # 你的任务有10个类别（教育、科技等）
    print(f"正在从Hugging Face下载模型: {BERT_MODEL_NAME}")
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,  # 从Hugging Face下载
        num_labels=num_labels  # 你的任务类别数（比如THUCNews是10类）
    )

    # 3. 自动检测设备（优先GPU，无则用CPU）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)  # 将模型移动到对应设备
    print(f"使用设备：{device}（若为cuda，说明GPU已启用）")

    # 4. 配置训练参数（解决警告，适配GPU）
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_DIR,  # 模型保存路径（从config.py导入）
        eval_strategy="steps",  # 按步数评估，更快看到进度
        eval_steps=500,  # 每500步评估一次
        save_strategy="steps",  # 按步数保存
        save_steps=500,  # 每500步保存一次
        learning_rate=2e-5,  # 学习率
        per_device_train_batch_size=64,  # 增大批次大小加快训练（如GPU显存不足可改回32）
        per_device_eval_batch_size=64,  # 增大验证批次
        gradient_accumulation_steps=1,  # 去掉梯度累积，加快训练
        num_train_epochs=3,  # 训练轮数
        weight_decay=0.01,  # 权重衰减（防止过拟合）
        load_best_model_at_end=True,  # 训练结束后加载最优模型
        metric_for_best_model="accuracy",  # 用准确率选择最佳模型
        logging_dir="./logs",  # 日志保存路径
        logging_steps=50,  # 每50步打印一次日志，更频繁的反馈
        fp16=torch.cuda.is_available(),  # 如果有GPU，使用混合精度训练加速
        dataloader_num_workers=4,  # 多线程加载数据
        save_total_limit=2,  # 只保留最近2个检查点，节省磁盘
    )

    # 5. 数据整理器（自动处理padding，适配批次训练）
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 6. 初始化Trainer（解决tokenizer警告，改用processing_class）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        compute_metrics=compute_metrics,  # 评估指标
        data_collator=data_collator,
        processing_class=tokenizer  # 替代 deprecated 的 tokenizer 参数
    )

    # 7. 开始训练
    print("开始训练...")
    trainer.train()

    # 8. 训练结束后，在测试集上评估
    print("在测试集上评估模型...")
    test_results = trainer.evaluate(tokenized_ds["test"])
    print("测试集评估结果：", test_results)

if __name__ == "__main__":
    print("CUDA 可用：", torch.cuda.is_available())
    print("GPU 数量：", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("当前 GPU：", torch.cuda.current_device())
        print("GPU 名称：", torch.cuda.get_device_name(0))
    
    main()