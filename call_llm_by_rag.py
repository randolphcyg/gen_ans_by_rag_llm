import logging
import os
import requests
import torch
import jieba
from typing import List, Dict
from sentence_transformers import CrossEncoder
from pymilvus import MilvusClient
from stopwords import StopWordsManager
from langfuse import Langfuse
from langfuse import observe

# ===================== 1. 配置管理 =====================
# langfuse配置
LANGFUSE_SECRET_KEY = "sk-lf-ab1f9c14-4b8e-4d76-8533-52a2985fb4e3"
LANGFUSE_PUBLIC_KEY = "pk-lf-76c0c3e9-90c6-45af-b564-b9ed7052daf4"
LANGFUSE_BASE_URL = "http://localhost:3100"

# 核心服务地址
MILVUS_URI = "http://localhost:19530"
OLLAMA_HOST = "http://localhost:11434"

# 模型配置
COLLECTION_NAME = "zeek_rag_v8_0_4"
EMBED_MODEL = "bge-m3:latest"
LLM_MODEL = "qwen2.5-coder:3b"
RERANKER_MODEL_NAME = 'BAAI/bge-reranker-base'

# 算法参数
RETRIEVE_TOP_K = 50
RERANK_TOP_K = 3
RRF_K = 60
MAX_CONTEXT_CHARS = 3500
SCORE_THRESHOLD = -8.0

# 自动推导本地模型路径 (假设模型在当前脚本同级的 models 目录下)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_RERANKER_PATH = os.path.join(BASE_DIR, "models", "bge-reranker-base")

# ===================== 2. 初始化服务 =====================
langfuse = Langfuse(
    secret_key=LANGFUSE_SECRET_KEY,
    public_key=LANGFUSE_PUBLIC_KEY,
    host=LANGFUSE_BASE_URL,
    debug=False
)

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ZeekRAGAssistant:
    def __init__(self):
        self.session = requests.Session()

        self.stop_words_manager = StopWordsManager()
        jieba.initialize()  # 显式初始化

        try:
            self.milvus_client = MilvusClient(uri=MILVUS_URI)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_path = RERANKER_MODEL_NAME

            # 检查本地模型是否存在
            if os.path.exists(LOCAL_RERANKER_PATH):
                logging.info(f"📂 发现本地模型，启用离线模式: {LOCAL_RERANKER_PATH}")
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_HUB_OFFLINE"] = "1"
                model_path = LOCAL_RERANKER_PATH
            else:
                logging.info(f"🌐 本地模型不存在，将尝试在线下载: {RERANKER_MODEL_NAME}")

            # 加载重排模型
            self.reranker = CrossEncoder(model_path, device=device)
            logging.info(f"✅ 助手初始化完成 | Reranker: {device} | LLM: {LLM_MODEL}")

            # 预热 Ollama 模型
            logging.info("🔥 正在预热 Ollama 模型...")
            self.session.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": LLM_MODEL, "prompt": "hi", "stream": False},
                timeout=1 # 这里超时无所谓，只要触发加载就行
            )

        except Exception as e:
            logging.error(f"❌ 初始化严重失败: {e}")
            raise

    def get_embedding(self, text: str):
        """获取文本的向量表示（增加异常处理）"""
        try:
            resp = self.session.post(
                f"{OLLAMA_HOST}/api/embed",
                json={"model": EMBED_MODEL, "input": [text]},
                timeout=10
            )
            return resp.json()["embeddings"][0]
        except: return None

    def generate_multi_queries(self, original_query: str) -> List[str]:
        """
        最佳实践：利用 LLM 生成 3 个不同维度的搜索词
        1. 原始词
        2. 专家术语词
        3. 假设性代码片段 (HyDE 思想)
        """
        # 简单问题不消耗 Token
        if len(original_query) < 5: return [original_query]

        try:
            # 从 Langfuse 加载 Prompt
            langfuse_prompt = langfuse.get_prompt("zeek_query_expansion")
            compiled_prompt = langfuse_prompt.compile(query=original_query)

            # 这里可以用更小的模型 (如 qwen:0.5b 或 1.5b) 来降低延迟
            resp = self.session.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": compiled_prompt,
                    "stream": False,
                    "options": {"temperature": 0.5} # 稍微有点创造力
                },
                timeout=20
            )
            text = resp.json().get("response", "").strip()

            # 解析 LLM 输出的 3 行文本
            queries = [line.strip() for line in text.split('\n') if line.strip()]

            # 兜底：如果 LLM 格式乱了，至少保底原始问题
            if not queries: queries = [original_query]

            logging.info(f"🚀 多路查询生成: {queries}")
            return queries

        except Exception as e:
            logging.warning(f"查询扩展失败，回退到原始查询: {e}")
            return [original_query]

    @observe(name="RAG-检索阶段")
    def retrieve(self, query: str):
        # 1. --- 核心升级：获取多路查询 ---
        search_queries = self.generate_multi_queries(query)

        # 2. --- 核心升级：批量向量化 (Batch Embedding) ---
        # Ollama 支持 batch input，一次网络请求拿回 3 个向量，比循环快
        query_vecs = []
        try:
            resp = self.session.post(
                f"{OLLAMA_HOST}/api/embed",
                json={"model": EMBED_MODEL, "input": search_queries},
                timeout=10
            )
            query_vecs = resp.json().get("embeddings", [])
        except Exception as e:
            logging.error(f"Embedding 失败: {e}")

        # 3. --- 核心升级：Milvus 批量检索 ---
        v_hits = []
        if query_vecs:
            try:
                # search_requests 可以并行搜索
                res = self.milvus_client.search(
                    collection_name=COLLECTION_NAME,
                    data=query_vecs,  # 传入多个向量
                    limit=RETRIEVE_TOP_K // 2, # 每个向量少取点，反正要合并
                    output_fields=["doc_title", "section_title", "raw_content", "content_type"]
                )
                # 展平结果 (res 是一个二维列表: [ [query1_hits], [query2_hits] ])
                for hits in res:
                    for hit in hits:
                        v_hits.append(hit["entity"])
            except Exception as e:
                logging.error(f"Milvus 搜索失败: {e}")

        # 4. 关键词检索 (对所有扩展词都做一遍关键词匹配)
        k_hits = []
        # 将所有扩展词合并成一个大的关键词池，去重
        all_keywords = set()
        for q in search_queries:
            kws = self.stop_words_manager.filter_stop_words(q)
            all_keywords.update(kws if kws else [q])

        # 构造复杂的 OR 查询
        filter_exprs = []
        for kw in list(all_keywords)[:8]: # 限制数量防止 URL 过长
            safe_kw = kw.replace('"', '').replace("'", "")
            filter_exprs.append(f'section_title like "%{safe_kw}%"')
            filter_exprs.append(f'raw_content like "%{safe_kw}%"')

        if filter_exprs:
            try:
                k_res = self.milvus_client.query(
                    collection_name=COLLECTION_NAME,
                    filter=" or ".join(filter_exprs),
                    limit=RETRIEVE_TOP_K,
                    output_fields=["doc_title", "section_title", "raw_content", "content_type"]
                )
                k_hits = k_res
            except Exception as e:
                logging.error(f"关键词检索失败: {e}")

        # 5. RRF 融合 (算法本身不需要变，因为它天然支持去重和排序)
        return self.reciprocal_rank_fusion(v_hits, k_hits)

    def reciprocal_rank_fusion(self, v_hits, k_hits):
        scores = {}
        doc_map = {}
        for source_hits in [v_hits, k_hits]:
            for rank, h in enumerate(source_hits):
                # 使用更短的 hash key 节省内存
                key = hash(h["raw_content"][:200])
                scores[key] = scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)
                doc_map[key] = h

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[k] for k, s in sorted_docs[:30]]

    @observe(name="RAG-重排阶段")
    def rerank_with_bge(self, query: str, hits: List[Dict]):
        if not hits: return []

        # 1. BGE 评分 (Truncate 设为 True 防止超长文本报错)
        # BGE-Reranker 最大长度是 512 token，虽然我们切了字符，但最好显式截断
        pairs = [[query, h["raw_content"][:1000]] for h in hits]
        bge_scores = self.reranker.predict(pairs)

        q_words = set(self.stop_words_manager.filter_stop_words(query, 1))
        query_lower = query.lower()

        for i, hit in enumerate(hits):
            base = float(bge_scores[i])
            boost = 0.0
            title = hit.get("section_title", "").lower()

            # 策略计算
            t_words = set(self.stop_words_manager.filter_stop_words(title, 1))
            if q_words.intersection(t_words): boost += 25.0
            if ">" in title: boost += 10.0

            # 技术/代码意图奖励
            if any(term in query_lower for term in ["cert", "init", "done", "log", "字段"]):
                if any(term in title for term in ["base", "script", "reference", "manual"]):
                    boost += 15.0

            if hit.get("content_type") == "code" and \
                    any(k in query_lower for k in ["脚本", "代码", "写", "script", "检测"]):
                boost += 20.0

            hit["final_score"] = base + boost

        # 2. 排序
        sorted_hits = sorted(hits, key=lambda x: x["final_score"], reverse=True)

        # 3. 去重
        unique_hits = []
        seen_keys = set()
        for hit in sorted_hits:
            dedup_key = f"{hit.get('doc_title', '')}->{hit.get('section_title', '')}"
            if dedup_key not in seen_keys:
                unique_hits.append(hit)
                seen_keys.add(dedup_key)
            if len(unique_hits) >= RERANK_TOP_K: break

        if unique_hits:
            langfuse.update_current_span(metadata={"top1_score": unique_hits[0]['final_score']})

        return unique_hits

    @observe(name="LLM-生成回答")
    def ask_llm(self, query: str, chunks: List[Dict]):
        if not chunks: return "未找到相关参考资料。"

        context_str = ""
        current_len = 0
        for i, c in enumerate(chunks):
            content = c['raw_content'][:1500] # 单片防御性截断
            chunk_text = f"资料[{i+1}] (来源: {c['section_title']}):\n{content}\n\n"
            if current_len + len(chunk_text) > MAX_CONTEXT_CHARS: break
            context_str += chunk_text
            current_len += len(chunk_text)

        prompt_name = "zeek_script_coder" if any(k in query for k in ["脚本", "代码", "写"]) else "zeek_rag_qa"

        try:
            langfuse_prompt = langfuse.get_prompt(prompt_name)
            compiled_prompt = langfuse_prompt.compile(context=context_str, query=query)

            # 使用 session 发送
            r = self.session.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": compiled_prompt,
                    "stream": False,
                    "options": {
                        "temperature": langfuse_prompt.config.get("temperature", 0),
                        "top_p": langfuse_prompt.config.get("top_p", 0.9)
                    }
                },
                timeout=60
            )
            response = r.json().get("response", "生成失败")
            langfuse.update_current_span(output=response)
            return response
        except Exception as e:
            logging.error(f"LLM 调用异常: {e}")
            return "服务暂时不可用。"

    @observe(name="Zeek-RAG-完整问答链路")
    def chat(self, query: str):
        print(f"\n问: {query}")
        langfuse.update_current_span(input=query, metadata={"collection": COLLECTION_NAME})

        hits = self.retrieve(query)
        top = self.rerank_with_bge(query, hits)

        # 低分熔断
        if not top or top[0]['final_score'] < SCORE_THRESHOLD:
            msg = "抱歉，知识库中似乎没有找到关于此问题的相关信息。"
            print(f"🤖 Zeek AI:\n{msg}\n{'-'*30}\n⚠️ 触发低分熔断")
            langfuse.update_current_span(output=msg, metadata={"status": "rejected"})
            return

        ans = self.ask_llm(query, top)
        print(f"🤖 Zeek AI:\n{ans}\n{'-'*30}")
        for i, c in enumerate(top):
            print(f"Top {i+1}: {c['section_title']} (Score: {c['final_score']:.2f})")

        langfuse.update_current_span(output=ans)

if __name__ == "__main__":
    assistant = ZeekRAGAssistant()

    # 测试用例
    test_queries = [
        # "当前zeek版本号",
        # "what is zeek?",
        # "Zeek 是什么？",
        # "介绍一下 Zeek 的核心功能",
        # "Zeek 和 Suricata 的区别是什么？",
        # "zeek_init 和 zeek_done 函数的区别",
        # "如何用 Zeek 脚本提取 HTTP 请求的 URL？",
        # "Zeek SSL 日志中的 client_cert 字段含义",
        # "Zeek 支持 Python 3.10 吗？",
        "写一个zeek8.0.4版本分析pcap文件中ddos攻击的zeek脚本",
    ]

    for query in test_queries:
        try:
            assistant.chat(query)
            langfuse.flush()    # 记得刷新数据
        except Exception as e:
            logging.error(f"处理失败: {e}")