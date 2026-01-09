import logging
import os
import requests
import torch
import json
import time
import hashlib
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from pymilvus import MilvusClient, DataType
from langfuse import Langfuse
from langfuse import observe

# ===================== 1. 全局配置 =====================

# --- Langfuse 配置 ---
LANGFUSE_SECRET_KEY = "sk-lf-93542e4b-15ef-4a50-8719-0a12fbc42a8b"
LANGFUSE_PUBLIC_KEY = "pk-lf-f7f639cf-2585-4578-9404-26dec6b91626"
LANGFUSE_BASE_URL = "http://localhost:3100"

# --- 核心服务地址 ---
MILVUS_URI = "http://localhost:19530"
OLLAMA_HOST = "http://localhost:11434"

# --- 对比配置 ---
COLLECTION_HIERARCHICAL = "Vector_index_0804549e_ed61_4f22_9f94_16176bb0cede_Node"
COLLECTION_GENERAL      = "Vector_index_19191596_0e1f_492c_ab31_15e11501cec4_Node"

# --- 模型配置 ---
EMBED_MODEL = "bge-m3:latest"
LLM_MODEL = "qwen2.5-coder:3b"
RERANKER_MODEL_NAME = 'BAAI/bge-reranker-base'

# --- 算法参数 ---
RETRIEVE_TOP_K = 30
RERANK_TOP_K = 8
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_RERANKER_PATH = os.path.join(BASE_DIR, "models", "bge-reranker-base")

# ===================== 2. 初始化服务 =====================
# 提前配置日志，确保用户能看到启动过程
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
print("⏳ 正在初始化 AI 引擎 (加载 PyTorch & CUDA)...")

langfuse = Langfuse(
    secret_key=LANGFUSE_SECRET_KEY,
    public_key=LANGFUSE_PUBLIC_KEY,
    host=LANGFUSE_BASE_URL,
    debug=False,
    timeout=3
)

class ZeekRAGComparisonAssistant:
    def __init__(self):
        self.session = requests.Session()
        self.schemas = {}

        try:
            # 1. 初始化 Milvus
            self.milvus_client = MilvusClient(uri=MILVUS_URI)
            self._register_collection(COLLECTION_HIERARCHICAL, "父子索引")
            self._register_collection(COLLECTION_GENERAL, "通用索引")

            # 2. 🚀 强制 GPU 检查
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🖥️ 计算设备检测: {self.device.upper()}")

            if self.device == 'cpu':
                print("⚠️ [严重警告] 未检测到 GPU！建议安装 CUDA 版 PyTorch。")

            model_path = LOCAL_RERANKER_PATH if os.path.exists(LOCAL_RERANKER_PATH) else RERANKER_MODEL_NAME

            # 3. 加载 Reranker
            self.reranker = CrossEncoder(
                model_path,
                device=self.device,
                # 修复 Warning: 使用 model_kwargs 替代 automodel_args
                model_kwargs={"dtype": "auto"}
            )
            print(f"✅ Reranker 模型加载完成")

        except Exception as e:
            logging.error(f"❌ 初始化严重失败: {e}")
            raise

    def _register_collection(self, collection_name, label):
        """注册并探测集合 Schema"""
        try:
            res = self.milvus_client.describe_collection(collection_name)
        except Exception as e:
            print(f"❌ 无法连接集合 {collection_name}: {e}")
            return

        schema_info = {"vector_field": "vector", "text_field": "text", "meta_field": "meta", "label": label}

        for field in res.get('fields', []):
            if field['type'] == DataType.FLOAT_VECTOR: schema_info["vector_field"] = field['name']
            if field['type'] == DataType.VARCHAR and field['name'] in ['text', 'page_content', 'raw_content']: schema_info["text_field"] = field['name']
            if field['type'] == DataType.JSON: schema_info["meta_field"] = field['name']

        self.schemas[collection_name] = schema_info
        print(f"🔍 [{label}] Schema: Vector='{schema_info['vector_field']}', Text='{schema_info['text_field']}'")

    def _route_prompt(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["script", "code", "write", "function", "generate", "脚本", "代码", "编写"]):
            return "zeek_coder"
        if any(k in q for k in ["error", "fail", "empty", "why", "debug", "fix", "报错", "为空", "排错"]):
            return "zeek_debugger"
        return "zeek_explainer"

    @observe(name="RAG-检索阶段")
    def retrieve(self, query: str, collection_name: str):
        schema = self.schemas.get(collection_name)
        if not schema: return []

        # Embedding
        try:
            resp = self.session.post(
                f"{OLLAMA_HOST}/api/embed",
                json={"model": EMBED_MODEL, "input": [query]},
                timeout=30
            )
            query_vecs = resp.json().get("embeddings", [])
        except Exception as e:
            logging.error(f"Embedding 失败: {e}")
            return []

        # Milvus Search
        v_hits = []
        if query_vecs:
            try:
                res = self.milvus_client.search(
                    collection_name=collection_name,
                    data=query_vecs,
                    anns_field=schema["vector_field"],
                    limit=RETRIEVE_TOP_K,
                    output_fields=[schema["text_field"], schema["meta_field"]],
                    search_params={"metric_type": "IP", "params": {"nprobe": 10}}
                )

                for hits in res:
                    for hit in hits:
                        entity = hit["entity"]
                        adapted_hit = {
                            "score": hit["distance"],
                            "raw_content": entity.get(schema["text_field"], "")
                        }
                        v_hits.append(adapted_hit)
            except Exception as e:
                logging.error(f"Milvus 搜索失败: {e}")

        return v_hits

    @observe(name="RAG-重排阶段")
    def rerank(self, query: str, hits: List[Dict]):
        if not hits: return []

        pairs = [[query[:512], h["raw_content"][:1024]] for h in hits]
        try:
            bge_scores = self.reranker.predict(pairs, batch_size=16)
        except Exception as e:
            return hits[:3]

        for i, hit in enumerate(hits):
            hit["final_score"] = float(bge_scores[i])

        sorted_hits = sorted(hits, key=lambda x: x["final_score"], reverse=True)

        unique_hits = []
        seen_content = set()

        for h in sorted_hits:
            content_sig = hashlib.md5(h["raw_content"][:100].encode('utf-8')).hexdigest()
            if content_sig not in seen_content:
                unique_hits.append(h)
                seen_content.add(content_sig)
            if len(unique_hits) >= RERANK_TOP_K: break

        return unique_hits

    @observe(name="LLM-生成回答")
    def ask_llm(self, query: str, chunks: List[Dict]):
        if not chunks: return "未找到相关 Zeek 文档，无法回答。"

        context_str = ""
        for i, c in enumerate(chunks):
            clean_text = c['raw_content'].strip()
            context_str += f"### Reference [{i+1}]:\n{clean_text}\n\n"

        prompt_name = self._route_prompt(query)
        final_prompt = ""

        try:
            lf_prompt = langfuse.get_prompt(prompt_name)
            compiled = lf_prompt.compile(context=context_str, query=query)
            for msg in compiled:
                role_prefix = "SYSTEM" if msg['role'] == 'system' else "USER"
                final_prompt += f"{role_prefix}: {msg['content']}\n\n"

        except Exception as e:
            # 兜底 Prompt
            if prompt_name == "zeek_coder":
                sys_msg = "You are a Zeek Scripting Expert. Write code based STRICTLY on the context. Use modern Zeek syntax."
            else:
                sys_msg = "You are a Zeek Expert. Answer based on the context."

            final_prompt = f"SYSTEM: {sys_msg}\n\nUSER: Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"

        try:
            r = self.session.post(f"{OLLAMA_HOST}/api/generate", json={
                "model": LLM_MODEL,
                "prompt": final_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 4096
                }
            })
            return r.json().get("response", "LLM 生成为空")
        except Exception as e:
            return f"LLM 调用错误: {e}"

    def run_comparison(self, query: str):
        print(f"\n{'='*20} 🟢 测试问题: {query} {'='*20}")
        prompt_type = self._route_prompt(query)
        print(f"🎨 路由 Prompt 类型: {prompt_type}")

        targets = [
            (COLLECTION_HIERARCHICAL, "👨‍👦 父子索引"),
            (COLLECTION_GENERAL,      "📄 通用索引")
        ]

        for col_id, label in targets:
            print(f"\n>>> 正在测试: {label} ...")
            start_t = time.time()

            try:
                hits = self.retrieve(query, col_id)
                top_hits = self.rerank(query, hits)
                ans = self.ask_llm(query, top_hits)
                cost = time.time() - start_t

                print(f"⏱️ 总耗时: {cost:.2f}s")
                print(f"🤖 回答:\n{ans.strip()}")
                print(f"\n📚 引用片段 (Top 3):")
                for i, h in enumerate(top_hits[:3]):
                    preview = h['raw_content'][:100].replace('\n', ' ')
                    print(f"   [{i+1}] Score: {h['final_score']:.4f} | {preview}...")

            except Exception as e:
                print(f"❌ 错误: {e}")

            print("-" * 50)

if __name__ == "__main__":
    assistant = ZeekRAGComparisonAssistant()

    test_queries = [
        "what is zeek?",
        "Explain the meaning of the `history` field string 'ShADadFf' in `conn.log`.",
        "Write a script to handle `ssh_auth_successful` event. Do I need to load any module?",
        "Why is my `notice.log` empty even though I see attacks in `conn.log`?"
    ]

    print(f"🚀 开始执行 {len(test_queries)} 个测试用例...")

    for query in test_queries:
        assistant.run_comparison(query)
        langfuse.flush()
        time.sleep(1)