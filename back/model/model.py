import dao.database as database

from model.summarizer import Summarizer
from model.classifier import classify_news, classify_news_with_confidence
from newspaper import Article

def fetch_article(url: str):
    article = Article(url, language='zh')
    article.download()
    article.parse()
    return article

def get_article_summary(content: str, topk: int = 5) -> str:
    summarizer = Summarizer(
        ckpt_path="model/sum_model/extractor_tiny_simple.pt",
        model_name="uer/roberta-tiny-wwm-chinese-cluecorpussmall",
        max_len=128,
        device=None,
    )
    summary, (sents, scores) = summarizer.summarize_text(content, topk=topk)
    print(summary)
    return summary


def get_article_kind(article_summary: str) -> str:
    """
    对文章摘要进行分类
    
    参数:
        article_summary: 文章摘要文本
    
    返回:
        str: 类别名称（如 "财经"、"体育"、"娱乐" 等）
    """
    return classify_news(article_summary)


def get_article_kind_with_confidence(article_summary: str) -> dict:
    """
    对文章摘要进行分类（带置信度）
    
    参数:
        article_summary: 文章摘要文本
    
    返回:
        dict: {
            "category": 类别名称,
            "confidence": 置信度(0-1),
            "label_id": 类别ID
        }
    """
    return classify_news_with_confidence(article_summary)