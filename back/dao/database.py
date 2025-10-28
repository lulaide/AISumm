import os
import sqlite3
from typing import Dict, List, Optional


BASE_DIR = os.path.dirname(__file__)
DB_FILE = os.path.join(BASE_DIR, "articles.db")


def get_db_connection():
	"""获取数据库连接（row_factory 为 sqlite3.Row）。"""
	conn = sqlite3.connect(DB_FILE)
	conn.row_factory = sqlite3.Row
	return conn


def init_db():
	"""初始化数据库，创建 articles 表（如果不存在）。

	字段：id, url, title, content, summary, kind, created_at
	created_at 使用 SQLite 的 CURRENT_TIMESTAMP 默认值。
	"""
	conn = get_db_connection()
	cursor = conn.cursor()
	cursor.execute(
		"""
		CREATE TABLE IF NOT EXISTS articles (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			url TEXT,
			title TEXT,
			content TEXT,
			summary TEXT,
			kind TEXT,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)
		"""
	)
	conn.commit()
	conn.close()


def add_article(
	url: str,
	title: str,
	content: str,
	summary: Optional[str] = None,
	kind: Optional[str] = None,
):
	"""向数据库中添加新文章。

	返回插入后的整行字典（通过 get_article_by_id），如果失败返回 None。
	"""
	conn = get_db_connection()
	cursor = conn.cursor()
	try:
		cursor.execute(
			"INSERT INTO articles (url, title, content, summary, kind) VALUES (?, ?, ?, ?, ?)",
			(url, title, content, summary, kind)
		)
		conn.commit()
		new_id = cursor.lastrowid
		return get_article_by_id(new_id) if new_id is not None else None
	except sqlite3.IntegrityError:
		# 可捕获唯一性冲突或其他完整性错误
		return None
	finally:
		conn.close()


def get_article_by_id(article_id: int) -> Optional[Dict]:
	"""通过 ID 获取单篇文章（字典或 None）。"""
	conn = get_db_connection()
	row = conn.execute(
		"SELECT id, url, title, content, summary, kind, created_at FROM articles WHERE id = ?",
		(article_id,),
	).fetchone()
	conn.close()
	return dict(row) if row else None

def delete_article_by_id(article_id: int) -> bool:
	"""通过 ID 删除单篇文章，返回是否成功删除。"""
	conn = get_db_connection()
	cursor = conn.cursor()
	cursor.execute(
		"DELETE FROM articles WHERE id = ?",
		(article_id,),
	)
	conn.commit()
	deleted = cursor.rowcount > 0
	conn.close()
	return deleted

def get_articles_by_kind(kind: Optional[str] = None) -> List[Dict]:
	"""按 kind 获取文章，返回包含 id,title,created_at 的字典列表。

	如果 kind 为 None，则返回所有文章的简要信息。
	"""
	conn = get_db_connection()
	if kind is None:
		rows = conn.execute("SELECT id, title, created_at FROM articles ORDER BY created_at DESC").fetchall()
	else:
		rows = conn.execute(
			"SELECT id, title, created_at FROM articles WHERE kind = ? ORDER BY created_at DESC",
			(kind,),
		).fetchall()
	conn.close()
	return [dict(r) for r in rows]
