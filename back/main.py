import model.model as model
import dao.database as database
from flask import Flask, request, jsonify
from newspaper import Article

database.init_db()

app = Flask(__name__)

@app.route('/api/articles', methods=['GET'])
def get_articles_by_kind():
    kind = request.args.get('kind')
    articles = database.get_articles_by_kind(kind)
    return jsonify(articles)

@app.route('/api/articles/<int:article_id>', methods=['GET'])
def get_article_by_id(article_id: int):
    article = database.get_article_by_id(article_id)
    if article:
        return jsonify(article)
    return jsonify({'error': 'Article not found'}), 404

@app.route('/api/articles', methods=['POST'])
def add_article():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    article : Article = model.fetch_article(url)
    content = article.text
    summary = model.get_article_summary(content)
    kind = model.get_article_kind(summary)
    database.add_article(url, article.title, content, summary, kind)
    if article:
        return jsonify({'message': 'Article added successfully'}), 200
    return jsonify({'error': 'Failed to add article'}), 500

@app.route('/api/articles/<int:article_id>', methods=['DELETE'])
def delete_article_by_id(article_id: int):
    success = database.delete_article_by_id(article_id)
    if success:
        return jsonify({'message': 'Article deleted successfully'}), 200
    return jsonify({'error': 'Failed to delete article'}), 500

if __name__ == '__main__':
    app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)