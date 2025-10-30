from transformers import BertForSequenceClassification
from config import LOCAL_BERT_PATH

# 加载分类模型
def load_bert_classifier(num_labels):
    model = BertForSequenceClassification.from_pretrained(
        LOCAL_BERT_PATH,
        num_labels=num_labels  # 类别数量
    )
    return model