import logging
import requests
import torch
import jieba
from typing import List, Dict
from sentence_transformers import CrossEncoder
from pymilvus import MilvusClient
from stopwords import StopWordsManager

# ===================== 核心配置 =====================
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "zeek_rag_v8_0_4"
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "bge-m3:latest"
LLM_MODEL = "qwen2.5-coder:3b"

# 检索与重排配置
RETRIEVE_TOP_K = 50
RERANK_TOP_K = 3
RRF_K = 60  # RRF算法参数

class ZeekRAGAssistant:
    def __init__(self):
        self.stop_words_manager = StopWordsManager()
        jieba.initialize()  # 显式初始化

        try:
            self.milvus_client = MilvusClient(uri=MILVUS_URI)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # 强制 Reranker 使用 CPU 以节省显存给 LLM
            self.reranker = CrossEncoder('BAAI/bge-reranker-base', device=device)
            logging.info(f"✅ 助手初始化完成 | Reranker: CPU | LLM: {LLM_MODEL}")
        except Exception as e:
            logging.error(f"初始化失败: {e}")
            raise

    def get_embedding(self, text: str):
        """获取文本的向量表示（增加异常处理）"""
        try:
            resp = requests.post(f"{OLLAMA_HOST}/api/embed",
                                 json={"model": EMBED_MODEL, "input": [text]}, timeout=10)
            return resp.json()["embeddings"][0]
        except: return None

    def retrieve(self, query: str):
        # 1. 语义检索
        query_vec = self.get_embedding(query)
        v_hits = []
        if query_vec:
            res = self.milvus_client.search(
                collection_name=COLLECTION_NAME, data=[query_vec],
                limit=RETRIEVE_TOP_K, output_fields=["doc_title", "section_title", "raw_content", "content_type"]
            )
            v_hits = [hit["entity"] for hit in res[0]]

        # 2. 增强关键词检索
        k_hits = []
        # 这里的关键词提取现在能保留 zeek_init
        keywords = self.stop_words_manager.filter_stop_words(query)

        # 兜底策略：如果分词没分出结果，直接拿整个 query 做模糊匹配
        if not keywords: keywords = [query.strip()]

        filter_exprs = []
        for kw in keywords[:5]: # 限制词数提高性能
            filter_exprs.append(f'section_title like "%{kw}%"')
            filter_exprs.append(f'raw_content like "%{kw}%"')

        if filter_exprs:
            k_res = self.milvus_client.query(
                collection_name=COLLECTION_NAME,
                filter=" or ".join(filter_exprs),
                limit=RETRIEVE_TOP_K,
                output_fields=["doc_title", "section_title", "raw_content", "content_type"]
            )
            k_hits = k_res

        return self.reciprocal_rank_fusion(v_hits, k_hits)

    def reciprocal_rank_fusion(self, v_hits, k_hits):
        scores = {}
        doc_map = {}
        for rank, h in enumerate(v_hits):
            key = h["raw_content"][:200]
            scores[key] = scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)
            doc_map[key] = h
        for rank, h in enumerate(k_hits):
            key = h["raw_content"][:200]
            scores[key] = scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)
            doc_map[key] = h

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[k] for k, s in sorted_docs[:30]]

    def rerank_with_bge(self, query: str, hits: List[Dict]):
        if not hits: return []

        # 评分前过滤，确保包含具体函数名/字段名的切片排在前面
        query_lower = query.lower()
        pairs = [[query, h["raw_content"][:800]] for h in hits]
        bge_scores = self.reranker.predict(pairs)

        for i, hit in enumerate(hits):
            base = float(bge_scores[i])
            boost = 0.0
            title = hit.get("section_title", "").lower()

            # 策略 A: 标题关键词密度
            q_words = set(self.stop_words_manager.filter_stop_words(query, 1))
            t_words = set(self.stop_words_manager.filter_stop_words(title, 1))
            if q_words.intersection(t_words):
                boost += 25.0

            # 策略 B: 结构化奖励
            if ">" in title: boost += 10.0

            # 策略 C: 技术意图奖励 (针对 client_cert, zeek_init 等)
            if any(term in query_lower for term in ["cert", "init", "done", "log", "字段"]):
                if any(term in title for term in ["base", "script", "reference", "manual"]):
                    boost += 15.0

            hit["base_score"] = base
            hit["boost"] = boost
            hit["final_score"] = base + boost

        return sorted(hits, key=lambda x: x["final_score"], reverse=True)[:RERANK_TOP_K]

    def ask_llm(self, query: str, chunks: List[Dict]):
        if not chunks: return "未找到相关参考资料。"

        context = ""
        for i, c in enumerate(chunks):
            context += f"资料[{i+1}] (来源: {c['section_title']}):\n{c['raw_content']}\n\n"

        prompt = f"你是 Zeek 专家。请根据资料回答问题。资料按相关性排序。\n资料：\n{context}\n问题：{query}\n回答："

        try:
            r = requests.post(f"{OLLAMA_HOST}/api/generate",
                              json={"model": LLM_MODEL, "prompt": prompt, "stream": False}, timeout=60)
            return r.json().get("response", "生成失败")
        except: return "LLM 服务超时"

    def chat(self, query: str):
        print(f"\n问: {query}")
        hits = self.retrieve(query)
        top = self.rerank_with_bge(query, hits)
        ans = self.ask_llm(query, top)

        print(f"🤖 Zeek AI:\n{ans}")
        print("-" * 30)
        for i, c in enumerate(top):
            print(f"Top {i+1}: {c['section_title']} (Score: {c['final_score']:.2f})")


# ===================== 测试执行 =====================
if __name__ == "__main__":
    assistant = ZeekRAGAssistant()

    # 测试用例
    # test_queries = [
    #     "当前zeek版本号",
    #     "what is zeek?",
    #     "Zeek 是什么？",
    #     "介绍一下 Zeek 的核心功能",
    #     "Zeek 和 Suricata 的区别是什么？",
    #     "zeek_init 和 zeek_done 函数的区别",
    #     "如何用 Zeek 脚本提取 HTTP 请求的 URL？",
    #     "Zeek SSL 日志中的 client_cert 字段含义",
    #     "Zeek 支持 Python 3.10 吗？",
    # ]

    test_queries = [
        "写一个zeek8.0.4版本分析pcap文件中ddos攻击的zeek脚本",
    ]

    # 执行测试
    for query in test_queries:
        try:
            assistant.chat(query)
        except Exception as e:
            logging.error(f"处理查询失败 '{query}': {str(e)}")
            print(f"\n❌ 处理失败: {str(e)}\n")