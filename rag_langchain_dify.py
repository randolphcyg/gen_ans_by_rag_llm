# -*- coding: utf-8 -*-
import time
import requests
from typing import List, Dict, Any, Optional, Tuple

from flask import Flask, request, jsonify
from pydantic import Field

# LangChain Core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# LLM
from langchain_ollama import ChatOllama

# Observability
import langfuse
from langfuse import Langfuse, get_client, observe
from langfuse.langchain import CallbackHandler

# --- 1. 配置区 ---
DIFY_API_KEY = "dataset-MF0p7JRI8hUO5nHXRJ73szfi"
DIFY_DATASET_ID = "ec367307-db47-4449-9624-6e8ae9d6c405"
DIFY_BASE_URL = "http://127.0.0.1:5001/v1"
COLLECTION_TARGET = "Vector_index_ec367307_db47_4449_9624_6e8ae9d6c405_Node"

LANGFUSE_SECRET_KEY = "sk-lf-15beef95-8342-4448-b6d7-eb8cf71897bb"
LANGFUSE_PUBLIC_KEY = "pk-lf-c1be2f02-11b3-422d-95ed-8d43b6fb6e22"
LANGFUSE_BASE_URL = "http://localhost:3100"
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
RERANK_MODEL = "bge-reranker-v2-m3"
LLM_MODEL = "qwen2.5-coder:3b"


# --- 2. 自定义 Dify 检索器 (增强版) ---
class DifyKnowledgeBaseRetriever(BaseRetriever):
    api_key: str = Field(..., description="Dify Dataset API Key")
    dataset_id: str = Field(..., description="Dify Knowledge Base ID")
    base_url: str = Field(default="http://localhost/v1", description="Dify API Base URL")
    top_k: int = Field(default=5, description="Number of docs to retrieve")


    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        url = f"{self.base_url}/datasets/{self.dataset_id}/retrieve"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "query": query,
            "retrieval_model": {
                "search_method": "hybrid_search",
                "reranking_enable": True,
                "top_k": self.top_k,
                "score_threshold_enabled": True,
                "score_threshold": 0.01
            }
        }

        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            latency = time.time() - start_time

            if run_manager:
                run_manager.on_text(f"Dify API Latency: {latency:.2f}s")

            data = response.json()
            records = data.get('records', [])

            langchain_docs = []
            for record in records:
                segment = record.get('segment', {})
                content = segment.get('content', '')
                meta = {
                    "score": record.get('score', 0.0),
                    "doc_id": segment.get('document_id'),
                    "segment_id": segment.get('id'),
                    "source": "dify_api"
                }
                langchain_docs.append(Document(page_content=content, metadata=meta))

            return langchain_docs

        except Exception as e:
            print(f"❌ [Dify API Error] {e}")
            return []

# --- 3. 主服务逻辑 ---
class ZeekLangChainService:
    def __init__(self):
        print("⏳ Service initializing with Dify API...")

        Langfuse(
            secret_key=LANGFUSE_SECRET_KEY,
            public_key=LANGFUSE_PUBLIC_KEY,
            host=LANGFUSE_BASE_URL,
            debug=False
        )

        self.langfuse = get_client()
        self.trace_handler = CallbackHandler()

        self.retriever = DifyKnowledgeBaseRetriever(
            api_key=DIFY_API_KEY,
            dataset_id=DIFY_DATASET_ID,
            base_url=DIFY_BASE_URL,
            top_k=6
        )

        self.llm = ChatOllama(
            base_url=OLLAMA_HOST,
            model=LLM_MODEL,
            temperature=0.1,
            num_ctx=4096,
        )

    def _route_prompt(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["script", "code", "write", "function"]): return "zeek_coder"
        if any(k in q for k in ["error", "fail", "empty", "why", "debug"]): return "zeek_debugger"
        return "zeek_explainer"

    def _get_langfuse_prompt(self, prompt_name: str) -> Tuple[ChatPromptTemplate, Any]:
        lf_prompt = self.langfuse.get_prompt(prompt_name, label="latest")
        messages = lf_prompt.get_langchain_prompt()
        if isinstance(messages, list):
            lc_prompt = ChatPromptTemplate.from_messages(messages)
        elif isinstance(messages, str):
            lc_prompt = ChatPromptTemplate.from_template(messages)
        else:
            lc_prompt = messages

        return lc_prompt, lf_prompt

    def _format_docs(self, docs):
        if not docs: return "No relevant documents found."
        return "\n\n".join([f"### Context (Score: {d.metadata.get('score', 0):.4f}):\n{d.page_content.strip()}" for d in docs])

    @observe(name="Zeek-RAG-Flow")
    def ask_with_refs(self, query: str) -> Dict[str, Any]:
        if not query or len(query.strip()) < 2:
            return {"query": query, "answer": "Enter a valid question", "references": [], "status": "error"}

        prompt_name = self._route_prompt(query)
        lc_prompt, lf_prompt_obj = self._get_langfuse_prompt(prompt_name)

        self.langfuse.update_current_trace(
            tags=[prompt_name, "dify_hybrid", "ollama"],
            user_id="ran9527",
            metadata={
                "prompt_name": prompt_name,
                "prompt_version": str(lf_prompt_obj.version),
                "embed_model": EMBED_MODEL,
                "rerank_model": RERANK_MODEL,
                "llm_model": LLM_MODEL,
                "milvus_collection": COLLECTION_TARGET,
            }
        )

        rag_chain = (
            RunnableParallel(
                context=self.retriever.with_config(run_name="Dify_Retrieval"),
                question=RunnablePassthrough()
            )
            .assign(
                formatted_context=lambda x: self._format_docs(x["context"])
            )
            .assign(
                answer=lambda x: (
                        lc_prompt.partial(context=x["formatted_context"])
                        | self.llm.with_config(run_name="Ollama_Generate")
                        | StrOutputParser()
                ).invoke({"question": x["question"]})
            )
        )

        try:
            res = rag_chain.invoke(
                input=query,
                config={"callbacks": [self.trace_handler]}
            )

            self.langfuse.update_current_trace(
                output=res["answer"],
                metadata={
                    "retrieved_docs_count": len(res["context"]),
                    "top_doc_score": res["context"][0].metadata.get('score', 0) if res["context"] else 0
                }
            )

            refs = []
            for d in res["context"]:
                meta = d.metadata or {}
                refs.append({
                    "score": round(meta.get("score", 0.0), 4),
                    "relevance_score": round(meta.get("score", 0.0), 4),
                    "content": d.page_content[:500].strip(),
                    "doc_id": meta.get("doc_id", "unknown")
                })

            return {
                "query": query,
                "answer": res["answer"].strip(),
                "references": refs,
                "used_prompt": f"{prompt_name} (v{lf_prompt_obj.version})", # 返回给前端版本号
                "status": "success"
            }

        except Exception as e:
            print(f"❌ Pipeline Error: {e}")
            self.langfuse.update_current_trace(
                metadata={"error": str(e)}
            )
            return {"status": "error", "answer": str(e), "references": [], "query": query}

if __name__ == "__main__":
    service = ZeekLangChainService()
    app = Flask(__name__)

    @app.route('/chat', methods=['POST'])
    def chat():
        data = request.json or {}
        query = data.get('query', '').strip()
        start = time.time()
        result = service.ask_with_refs(query)
        result['cost_time'] = round(time.time() - start, 2)
        return jsonify(result), 200 if result["status"] == "success" else 500

    app.run(host="0.0.0.0", port=15000, debug=False, threaded=True)