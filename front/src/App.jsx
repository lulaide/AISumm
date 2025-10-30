import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

const CATEGORIES = [
  { id: 'all', name: '全部', apiValue: null },
  { id: 'finance', name: '财经', apiValue: '财经' },
  { id: 'estate', name: '房产', apiValue: '房产' },
  { id: 'stock', name: '股票', apiValue: '股票' },
  { id: 'edu', name: '教育', apiValue: '教育' },
  { id: 'tech', name: '科技', apiValue: '科技' },
  { id: 'society', name: '社会', apiValue: '社会' },
  { id: 'politics', name: '政治', apiValue: '政治' },
  { id: 'sports', name: '体育', apiValue: '体育' },
  { id: 'game', name: '游戏', apiValue: '游戏' },
  { id: 'entertainment', name: '娱乐', apiValue: '娱乐' }
]

function App() {
  const [url, setUrl] = useState('')
  const [articles, setArticles] = useState([])
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedArticle, setSelectedArticle] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  // 获取文章列表
  const fetchArticles = async (categoryId) => {
    setLoading(true)
    setError(null)
    try {
      const category = CATEGORIES.find(c => c.id === categoryId)
      const params = category?.apiValue ? { kind: category.apiValue } : {}
      const response = await axios.get('/api/articles', { params })
      setArticles(response.data)
    } catch (err) {
      setError('获取文章失败')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  // 切换分类
  const handleCategoryChange = (categoryId) => {
    setSelectedCategory(categoryId)
    fetchArticles(categoryId)
  }

  // 添加文章
  const handleSubmit = async () => {
    if (!url.trim()) return
    
    setSubmitting(true)
    setError(null)
    try {
      await axios.post('/api/articles', { url })
      setUrl('')
      fetchArticles(selectedCategory)
    } catch (err) {
      setError(err.response?.data?.error || '添加文章失败')
      console.error(err)
    } finally {
      setSubmitting(false)
    }
  }

  // 粘贴
  const handlePaste = async () => {
    try {
      const text = await navigator.clipboard.readText()
      setUrl(text)
    } catch (err) {
      console.error('粘贴失败:', err)
    }
  }

  // 删除文章
  const handleDelete = async (id, e) => {
    e.stopPropagation()
    if (!confirm('确定删除这篇文章吗?')) return
    
    try {
      await axios.delete(`/api/articles/${id}`)
      fetchArticles(selectedCategory)
      if (selectedArticle?.id === id) {
        setSelectedArticle(null)
      }
    } catch (err) {
      setError('删除文章失败')
      console.error(err)
    }
  }

  // 获取文章详情
  const handleArticleClick = async (article) => {
    try {
      const response = await axios.get(`/api/articles/${article.id}`)
      setSelectedArticle(response.data)
    } catch (err) {
      setError('获取文章详情失败')
      console.error(err)
    }
  }

  // 初始加载
  useEffect(() => {
    fetchArticles('all')
  }, [])

  // 键盘事件
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !submitting) {
      handleSubmit()
    }
  }

  return (
    <div className="app">
      {/* 分类导航 */}
      <nav className="category-nav">
        <div className="category-list">
          {CATEGORIES.map(cat => (
            <button
              key={cat.id}
              className={`category-btn ${selectedCategory === cat.id ? 'active' : ''}`}
              onClick={() => handleCategoryChange(cat.id)}
            >
              {cat.name}
            </button>
          ))}
        </div>
      </nav>

      {/* 主内容 */}
      <main className="main-content">
        {/* 输入框 */}
        <div className="input-section">
          {error && <div className="error-message">{error}</div>}
          <div className="input-container">
            <input
              type="text"
              className="url-input"
              placeholder="输入文章链接..."
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={submitting}
            />
            <div className="input-actions">
              <button 
                className="paste-btn" 
                onClick={handlePaste}
                title="粘贴"
              >
                粘贴
              </button>
              <button
                className="submit-btn"
                onClick={handleSubmit}
                disabled={submitting || !url.trim()}
                title="添加文章"
              >
                →
              </button>
            </div>
          </div>
        </div>

        {/* 文章列表 */}
        {loading ? (
          <div className="loading">加载中...</div>
        ) : articles.length === 0 ? (
          <div className="empty-state">
            <p>📝</p>
            <p>暂无文章，添加一个开始吧</p>
          </div>
        ) : (
          <div className="articles-list">
            {articles.map(article => (
              <div
                key={article.id}
                className="article-card"
                onClick={() => handleArticleClick(article)}
              >
                <div className="article-header">
                  <div className="article-meta">
                    {article.kind && (
                      <span className="article-kind">
                        {CATEGORIES.find(c => c.apiValue === article.kind)?.name || article.kind}
                      </span>
                    )}
                    <span className="article-time">{article.created_at}</span>
                  </div>
                  <button
                    className="delete-btn"
                    onClick={(e) => handleDelete(article.id, e)}
                  >
                    删除
                  </button>
                </div>
                <h3 className="article-title">{article.title}</h3>
                {article.summary && (
                  <p className="article-summary">{article.summary}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </main>

      {/* 文章详情模态框 */}
      {selectedArticle && (
        <div className="modal-overlay" onClick={() => setSelectedArticle(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">{selectedArticle.title}</h2>
              <button
                className="close-btn"
                onClick={() => setSelectedArticle(null)}
              >
                ×
              </button>
            </div>
            <div className="modal-body">
              {/* 左侧摘要区域 - 小红书风格 */}
              <div className="summary-section">
                <div className="summary-content">
                  <div className="summary-header">
                    <div className="summary-meta">
                      {selectedArticle.kind && (
                        <span className="summary-tag">
                          {CATEGORIES.find(c => c.apiValue === selectedArticle.kind)?.name || selectedArticle.kind}
                        </span>
                      )}
                    </div>
                  </div>
                  
                  {selectedArticle.summary ? (
                    <div className="summary-text">
                      {selectedArticle.summary}
                    </div>
                  ) : (
                    <div className="summary-text">
                      暂无摘要
                    </div>
                  )}
                </div>
              </div>

              {/* 右侧内容区域 */}
              <div className="content-section">
                <div className="content-wrapper">
                  <div className="content-header">
                    {selectedArticle.created_at && (
                      <span className="content-time">{selectedArticle.created_at}</span>
                    )}
                    {selectedArticle.url && (
                      <a 
                        href={selectedArticle.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="source-link-btn"
                      >
                        🔗 查看原文
                      </a>
                    )}
                  </div>
                  
                  {selectedArticle.content ? (
                    <div className="content-text">
                      {selectedArticle.content}
                    </div>
                  ) : (
                    <div className="content-text">
                      暂无内容
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
