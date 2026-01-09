from flask import Flask, request, jsonify
from rag_core import ZeekRAGService
import time

app = Flask(__name__)

# 在 App 启动时就加载模型到 GPU，避免每次请求都重新加载
print("🚀 [Server] 正在启动 RAG 服务，请稍候...")
rag_service = ZeekRAGService()
print("✨ [Server] 服务启动就绪！访问 http://localhost:5000/chat")

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    API 端点
    输入: {"query": "如何检测 SSH 爆破?"}
    输出: {"answer": "...", "references": [...], "cost_seconds": 0.5}
    """
    start_time = time.time()

    # 1. 获取参数
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "缺少 query 参数"}), 400

    query = data['query']

    # 2. 调用核心业务逻辑
    result = rag_service.ask(query)

    # 3. 补充耗时信息
    result['cost_seconds'] = round(time.time() - start_time, 2)

    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "device": rag_service.device})

if __name__ == '__main__':
    # host='0.0.0.0' 允许局域网访问
    app.run(host='0.0.0.0', port=5000, debug=False)