# classifier.py
# 用途: 对新闻文本进行分类，返回类别名称
# 依赖: torch, transformers

from typing import Optional
import torch
from transformers import BertTokenizer, BertForSequenceClassification

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

# 模型配置
CLASSIFIER_MODEL_DIR = "model/classifier_model"  # 模型路径(相对于后端项目根目录)
MAX_LENGTH = 128  # 文本最大长度


class NewsClassifier:
    """
    新闻分类器
    用法:
        classifier = NewsClassifier()
        category = classifier.classify("周迅走进课堂当老师")
        print(category)  # "娱乐"
    """
    def __init__(
        self,
        model_dir: str = CLASSIFIER_MODEL_DIR,
        max_length: int = MAX_LENGTH,
        device: Optional[str] = None,
    ):
        """
        初始化分类器
        
        参数:
            model_dir: 模型文件夹路径
            max_length: 文本最大长度
            device: 设备类型 ("cuda" | "cpu" | "mps" | None=自动检测)
        """
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
        
        self.max_length = max_length
        
        # 加载tokenizer和模型
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(
            model_dir,
            use_safetensors=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def classify(self, text: str) -> str:
        """
        对新闻进行分类
        
        参数:
            text: 新闻标题或内容（字符串）
        
        返回:
            str: 类别名称（如 "财经"、"体育"、"娱乐" 等）
        """
        # 预处理文本
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # 将输入数据移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        outputs = self.model(**inputs)
        
        # 计算预测结果
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        pred_label_id = torch.argmax(probabilities, dim=1).item()
        pred_label_name = LABEL_MAP.get(pred_label_id, f"未知类别（ID：{pred_label_id}）")
        
        return pred_label_name
    
    @torch.no_grad()
    def classify_with_confidence(self, text: str) -> dict:
        """
        对新闻进行分类（带置信度）
        
        参数:
            text: 新闻标题或内容（字符串）
        
        返回:
            dict: {
                "category": 类别名称,
                "confidence": 置信度(0-1),
                "label_id": 类别ID
            }
        """
        # 预处理文本
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # 将输入数据移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        outputs = self.model(**inputs)
        
        # 计算预测结果
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        pred_label_id = torch.argmax(probabilities, dim=1).item()
        pred_label_name = LABEL_MAP.get(pred_label_id, f"未知类别（ID：{pred_label_id}）")
        pred_prob = probabilities[0][pred_label_id].item()
        
        return {
            "category": pred_label_name,
            "confidence": round(pred_prob, 4),
            "label_id": pred_label_id
        }


# --------------------------
# 全局单例（避免重复加载模型）
# --------------------------
_classifier_instance: Optional[NewsClassifier] = None

def get_classifier() -> NewsClassifier:
    """获取分类器单例"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = NewsClassifier()
    return _classifier_instance


def classify_news(text: str) -> str:
    """
    外部调用函数：对新闻进行分类
    
    参数:
        text: 新闻标题或内容（字符串）
    
    返回:
        str: 类别名称（如 "财经"、"体育"、"娱乐" 等）
    
    示例:
        >>> from model.classifier import classify_news
        >>> category = classify_news("周迅走进课堂当老师")
        >>> print(category)  # "娱乐"
    """
    classifier = get_classifier()
    return classifier.classify(text)


def classify_news_with_confidence(text: str) -> dict:
    """
    外部调用函数：对新闻进行分类（带置信度）
    
    参数:
        text: 新闻标题或内容（字符串）
    
    返回:
        dict: {
            "category": 类别名称,
            "confidence": 置信度(0-1),
            "label_id": 类别ID
        }
    
    示例:
        >>> from model.classifier import classify_news_with_confidence
        >>> result = classify_news_with_confidence("周迅走进课堂当老师")
        >>> print(result)  # {"category": "娱乐", "confidence": 0.9876, "label_id": 9}
    """
    classifier = get_classifier()
    return classifier.classify_with_confidence(text)


# --------------------------
# 测试代码
# --------------------------
if __name__ == "__main__":
    # 测试文本
    test_texts = [
        "周迅走进课堂当老师 变身邻家大姐姐",
        "稀土永磁全线下跌 包钢稀土跌4.3",
        "NBA季后赛首轮G3：火箭vs开拓者",
        "央行宣布降准0.5个百分点",
    ]
    
    print("=" * 60)
    print("新闻分类测试")
    print("=" * 60)
    
    # 测试基础分类
    print("\n【基础分类】")
    for text in test_texts:
        category = classify_news(text)
        print(f"文本：{text}")
        print(f"类别：{category}")
        print("-" * 60)
    
    # 测试带置信度的分类
    print("\n【带置信度的分类】")
    for text in test_texts:
        result = classify_news_with_confidence(text)
        print(f"文本：{text}")
        print(f"类别：{result['category']}")
        print(f"置信度：{result['confidence']}")
        print(f"类别ID：{result['label_id']}")
        print("-" * 60)
