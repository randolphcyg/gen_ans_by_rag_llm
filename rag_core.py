import logging
import os
import requests
import torch
import hashlib
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from pymilvus import MilvusClient, DataType
from langfuse import Langfuse, observe

# --- 配置保持不变 ---
LANGFUSE_SECRET_KEY = "sk-lf-93542e4b-15ef-4a50-8719-0a12fbc42a8b"
LANGFUSE_PUBLIC_KEY = "pk-lf-f7f639cf-2585-4578-9404-26dec6b91626"
LANGFUSE_BASE_URL = "http://localhost:3100"
MILVUS_URI = "http://localhost:19530"
OLLAMA_HOST = "http://localhost:11434"

COLLECTION_TARGET = "Vector_index_0804549e_ed61_4f22_9f94_16176bb0cede_Node"

EMBED_MODEL = "bge-m3:latest"
LLM_MODEL = "qwen2.5-coder:3b"
RERANKER_MODEL_NAME = 'BAAI/bge-reranker-base'
RETRIEVE_TOP_K = 40
RERANK_TOP_K = 8
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_RERANKER_PATH = os.path.join(BASE_DIR, "models", "bge-reranker-base")

# 初始化 Langfuse
langfuse = Langfuse(
    secret_key=LANGFUSE_SECRET_KEY,
    public_key=LANGFUSE_PUBLIC_KEY,
    host=LANGFUSE_BASE_URL,
    debug=False,
    timeout=3
)

class ZeekRAGService:
    def __init__(self):
        self.session = requests.Session()
        self.schema = {}

        print("⏳ [Core] 正在初始化 AI 引擎 (Loading PyTorch & CUDA)...")
        try:
            self.milvus_client = MilvusClient(uri=MILVUS_URI)
            # 注册父子索引 Schema
            self._register_schema()

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🖥️ [Core] 计算设备: {self.device.upper()}")

            model_path = LOCAL_RERANKER_PATH if os.path.exists(LOCAL_RERANKER_PATH) else RERANKER_MODEL_NAME
            self.reranker = CrossEncoder(
                model_path,
                device=self.device,
                model_kwargs={"dtype": "auto"}
            )
            print("✅ [Core] 模型加载完毕")

        except Exception as e:
            print(f"❌ [Core] 初始化失败: {e}")
            raise

    def _register_schema(self):
        res = self.milvus_client.describe_collection(COLLECTION_TARGET)
        self.schema = {"vector": "vector", "text": "text", "meta": "meta"}
        for field in res.get('fields', []):
            if field['type'] == DataType.FLOAT_VECTOR: self.schema["vector"] = field['name']
            if field['type'] == DataType.VARCHAR and field['name'] in ['text', 'page_content']: self.schema["text"] = field['name']
            if field['type'] == DataType.JSON: self.schema["meta"] = field['name']

    def _route_prompt(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["script", "code", "write", "function", "脚本", "代码"]): return "zeek_coder"
        if any(k in q for k in ["error", "fail", "empty", "why", "debug", "报错", "排错"]): return "zeek_debugger"
        return "zeek_explainer"

    @observe(name="RAG-API-请求")
    def ask(self, query: str) -> Dict[str, Any]:
        """对外暴露的核心方法，返回结构化数据"""
        try:
            # 1. 检索
            hits = self._retrieve(query)
            # 2. 重排
            top_hits = self._rerank(query, hits)
            # 3. 生成
            answer = self._generate_llm(query, top_hits)

            # 构造返回给前端的数据
            return {
                "query": query,
                "answer": answer.strip(),
                "references": [
                    {"score": h["final_score"], "content": h["raw_content"][:200] + "..."}
                    for h in top_hits[:3]
                ]
            }
        except Exception as e:
            return {"error": str(e)}

    @observe(name="RAG-检索阶段")
    def _retrieve(self, query: str):
        # ... (此处代码与原 retrieve 相同，只需将 print 改为 logging 或删除) ...
        # 为节省篇幅，假设此处逻辑与上一版一致
        try:
            resp = self.session.post(f"{OLLAMA_HOST}/api/embed", json={"model": EMBED_MODEL, "input": [query]}, timeout=30)
            vec = resp.json().get("embeddings", [])
            if not vec: return []

            res = self.milvus_client.search(
                collection_name=COLLECTION_TARGET,
                data=vec,
                anns_field=self.schema["vector"],
                limit=RETRIEVE_TOP_K,
                output_fields=[self.schema["text"]],
                search_params={"metric_type": "IP", "params": {"nprobe": 10}}
            )

            v_hits = []
            for hits in res:
                for hit in hits:
                    v_hits.append({
                        "score": hit["distance"],
                        "raw_content": hit["entity"].get(self.schema["text"], "")
                    })
            return v_hits
        except Exception:
            return []

    @observe(name="RAG-重排阶段")
    def _rerank(self, query: str, hits: List[Dict]):
        if not hits: return []
        pairs = [[query[:512], h["raw_content"][:1024]] for h in hits]
        try:
            scores = self.reranker.predict(pairs, batch_size=16)
            for i, h in enumerate(hits): h["final_score"] = float(scores[i])
            return sorted(hits, key=lambda x: x["final_score"], reverse=True)[:RERANK_TOP_K]
        except:
            return hits[:3]

    @observe(name="LLM-生成回答")
    def _generate_llm(self, query: str, chunks: List[Dict]):
        if not chunks: return "未找到相关文档。"
        context = "\n".join([f"### Ref [{i+1}]:\n{c['raw_content'].strip()}" for i, c in enumerate(chunks)])

        prompt_name = self._route_prompt(query)
        final_prompt = ""

        try:
            # 尝试从 Langfuse 获取
            lf_prompt = langfuse.get_prompt(prompt_name)
            compiled = lf_prompt.compile(context=context, query=query)
            for msg in compiled:
                prefix = "SYSTEM" if msg['role'] == 'system' else "USER"
                final_prompt += f"{prefix}: {msg['content']}\n\n"
        except:
            # 兜底
            final_prompt = f"SYSTEM: You are a Zeek Expert.\n\nUSER: Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        try:
            r = self.session.post(f"{OLLAMA_HOST}/api/generate", json={
                "model": LLM_MODEL, "prompt": final_prompt, "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 4096}
            })
            return r.json().get("response", "生成失败")
        except:
            return "LLM 服务不可用"