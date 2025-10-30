import torch
from transformers import BertTokenizer, BertForSequenceClassification
from config import SAVED_MODEL_DIR, MAX_LENGTH

# 类别ID到名称的映射
LABEL_MAP = {
    0: "财经",
    1: "房产",
    2: "股票",
    3: "教育",
    4: "科技",
    5: "社会",
    6: "政治",
    7: "体育",
    8: "游戏",
    9: "娱乐"
}


# --------------------------
# 2. 加载模型（自动适配GPU/CPU）
# --------------------------
def load_inference_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(SAVED_MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(
        SAVED_MODEL_DIR, 
        use_safetensors=True
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


# --------------------------
# 3. 预测函数（返回类别名称和概率）
# --------------------------
def predict_news(text, tokenizer, model, device):
    # 预处理文本（与训练时保持一致）
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"  # 返回PyTorch张量
    )
    # 将输入数据移动到与模型相同的设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 推理（关闭梯度计算，加速+省内存）
    with torch.no_grad():
        outputs = model(**inputs)

    # 计算预测结果
    logits = outputs.logits  # 模型原始输出
    probabilities = torch.softmax(logits, dim=1)  # 转换为概率（各类别概率和为1）
    pred_label_id = torch.argmax(probabilities, dim=1).item()  # 预测类别ID
    pred_label_name = LABEL_MAP.get(pred_label_id, f"未知类别（ID：{pred_label_id}）")  # 映射为名称
    pred_prob = probabilities[0][pred_label_id].item()  # 预测类别的概率（置信度）

    return {
        "文本": text,
        "预测类别ID": pred_label_id,
        "预测类别名称": pred_label_name,
        "置信度": round(pred_prob, 4)  # 保留4位小数
    }


# --------------------------
# 4. 外部调用接口
# --------------------------

# 全局单例（避免重复加载模型）
_model_cache = None

def classify_news(text: str) -> str:
    """
    外部调用函数：对新闻进行分类
    
    参数:
        text: 新闻标题或内容（字符串）
    
    返回:
        str: 类别名称（如 "财经"、"体育"、"娱乐" 等）
    
    示例:
        >>> from kind.predict import classify_news
        >>> category = classify_news("周迅走进课堂当老师")
        >>> print(category)  # "娱乐"
    """
    global _model_cache
    if _model_cache is None:
        _model_cache = load_inference_model()
    
    tokenizer, model, device = _model_cache
    result = predict_news(text, tokenizer, model, device)
    return result["预测类别名称"]


# --------------------------
# 5. 测试预测
# --------------------------
if __name__ == "__main__":
    # 测试文本
    test_texts = [
        "周迅走进课堂当老师 变身邻家大姐姐",
        "稀土永磁全线下跌 包钢稀土跌4.3",
    ]
    
    # 使用外部接口测试
    for text in test_texts:
        category = classify_news(text)
        print(f"文本：{text}")
        print(f"类别：{category}")
        print("-" * 50)