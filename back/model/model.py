import dao.database as database

from model.summarizer import Summarizer
from newspaper import Article

def fetch_article(url: str):
    article = Article(url, language='zh')
    article.download()
    article.parse()
    return article

def get_article_summary(content: str, topk: int = 5) -> str:
    summarizer = Summarizer(
        ckpt_path="model/extractor_tiny_simple.pt",
        model_name="uer/roberta-tiny-wwm-chinese-cluecorpussmall",
        max_len=128,
        device=None,
    )
    summary, (sents, scores) = summarizer.summarize_text(content, topk=topk)
    print(summary)
    return summary


def get_article_kind(article_summary: str) -> str:
    return ""