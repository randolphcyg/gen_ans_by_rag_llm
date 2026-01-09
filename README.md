# Zeek RAG Assistant (High-Performance)

> 基于父子索引 (Parent-Child Indexing) 与 BGE-M3/Reranker 的高性能 Zeek 技术文档问答助手。

## 📖 项目简介

本项目实现了一个垂直领域的 RAG（检索增强生成）系统，专精于 **Zeek 网络安全监控框架** 的技术问答。系统通过 Flask 封装为 API，后端集成 Langfuse 链路追踪，并利用 GPU 加速重排（Rerank）过程，实现毫秒级响应。

### 🌟 核心特性

* **父子索引 (Parent-Child Indexing)**：解决上下文碎片化问题，在代码生成和复杂概念解释上表现优异。
* **GPU 加速推理**：强制使用 CUDA 加速 BGE-Reranker 模型，检索耗时从 20s+ 降至 1s 内。
* **智能 Prompt 路由**：根据用户意图（写代码/排错/解释）自动切换 System Prompt。
* **全链路监控**：集成 Langfuse，记录检索、重排、生成的完整 Trace 和耗时。
* **混合检索**：Milvus 向量检索 + Cross-Encoder 精排序。

---

## 🛠️ 技术栈

* **LLM 运行时**: Ollama (Qwen2.5-Coder:3b)
* **向量数据库**: Milvus (Standalone)
* **Embedding**: BAAI/bge-m3
* **Reranker**: BAAI/bge-reranker-base
* **Observability**: Langfuse
* **Web 框架**: Flask

---

## 📂 项目结构

```text
gen_rag_by_zeek_doc/
├── models/                  # 本地模型权重目录 (可选)
│   └── bge-reranker-base/   # 预下载的重排模型
├── app.py                   # [入口] Flask API 服务，负责模型预加载与 HTTP 接口
├── rag_core.py              # [核心] 封装检索、重排、生成逻辑及 Langfuse 埋点
├── test_api.py              # [测试] 自动化测试脚本，包含典型测试用例
├── requirements.txt         # 项目依赖清单
├── download_model.py        # 工具：模型下载脚本
├── test_rag_index.py        # 工具：早期索引效果对比测试
└── test_langfuse.py         # 工具：Langfuse 连接性测试
```

---

## 🚀 快速开始

### 1. 环境准备

确保本机已安装以下服务并运行：

* **Milvus**: 端口 `19530`
* **Ollama**: 端口 `11434` (需拉取 `qwen2.5-coder:3b`)
* **Langfuse**: 端口 `3100`

确保显卡驱动支持 **CUDA 12.1+** (推荐 RTX 4070 及以上)。

### 2. 安装依赖

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. 安装 Python 依赖 (确保安装 CUDA 版 PyTorch)
pip install -r requirements.txt

```

### 3. 配置

目前配置项位于 `rag_core.py` 头部，关键配置如下：

```python
# rag_core.py
MILVUS_URI = "http://localhost:19530"
COLLECTION_TARGET = "Vector_index_xxx" # 你的父子索引 Collection ID
LANGFUSE_PUBLIC_KEY = "pk-lf-xxx"
LANGFUSE_SECRET_KEY = "sk-lf-xxx"

```

### 4. 运行服务

```bash
python app.py

```

*成功启动后将显示：*

> `🖥️ [Core] 计算设备: CUDA`
> `✅ [Core] 模型加载完毕`
> `Running on http://0.0.0.0:5000`

### 5. 运行测试

另起终端运行测试脚本，验证 API 是否正常：

```bash
python test_api.py

```

---

## 🔌 API 文档

### 1. 问答接口

* **URL**: `/chat`
* **Method**: `POST`
* **Content-Type**: `application/json`

**请求参数:**

| 参数名 | 类型 | 必选 | 描述 |
| --- | --- | --- | --- |
| `query` | string | 是 | 用户的问题，例如 "Write a zeek script for ssh" |

**响应示例:**

```json
{
    "query": "Write a script...",
    "answer": "Certainly! Here is the script...",
    "references": [
        {
            "score": 0.98,
            "content": "event ssh_auth_successful..."
        }
    ],
    "cost_seconds": 8.52
}

```

### 2. 健康检查

* **URL**: `/health`
* **Method**: `GET`

---

## 📝 TODO & 路线图

* [ ] **Prompt 工程强化**: 将所有 System Prompt 迁移至 Langfuse 远程管理，支持版本回滚。
* [ ] **配置解耦**: 引入 `.env` 文件管理敏感 Key 和 URL。
* [ ] **前端 UI**: 开发一个简单的 Vue/React 聊天界面。
* [ ] **多轮对话**: 增加 Session ID 支持，实现带有上下文的对话。
* [ ] **Docker 化**: 编写 Dockerfile，将应用打包为容器。

---

## ⚠️ 常见问题

**Q: 启动时提示 "未检测到 GPU"？**
A: 请检查 PyTorch 版本。请务必使用以下命令重新安装：
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

**Q: 第一次提问非常慢？**
A: 这是正常的。第一次运行需要初始化 CUDA 上下文，耗时约 3-5 秒，后续请求将变快。