import json
import time
import requests
from pymilvus import MilvusClient

# ================= 配置信息 =================
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "zeek_rag_v8_0_4"
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "bge-m3:latest"

# ================= 待插入的数据 (黄金样本) =================
# 这里就是你提供的那段 JSON
ddos_data = {
    "doc_id": "script_examples_001",
    "title": "Zeek Script Examples: Network Security",
    "partition": "p_scripts",
    "sections": [
        {
            "title": "DDoS Detection using SumStats",
            "blocks": [
                {
                    "type": "code",
                    "language": "zeek",
                    "code": "@load base/frameworks/sumstats\n@load base/frameworks/notice\n\nmodule DDoS;\n\nexport {\n    redef enum Notice::Type += { Syn_Flood };\n    const threshold: double = 20.0 &redef;\n    const interval_t: interval = 10sec &redef;\n}\n\nevent zeek_init() {\n    local r1: SumStats::Reducer = [$stream=\"syn.flood\", $apply=set(SumStats::SUM)];\n    SumStats::create([$name=\"syn-flood-detect\", $epoch=interval_t, $reducers=set(r1), $threshold_val=threshold, $threshold_crossed=function(key: SumStats::Key, result: SumStats::Result) { NOTICE([$note=Syn_Flood, $msg=fmt(\"Host %s sent too many SYNs\", key$host), $src=key$host]); }]);\n}\n\nevent connection_attempt(c: connection) {\n    SumStats::observe(\"syn.flood\", [$host=c$id$orig_h], [$num=1]);\n}"
                }
            ]
        }
    ]
}

def get_embedding(text):
    """调用 Ollama 生成向量"""
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": EMBED_MODEL, "input": [text]},
            timeout=30
        )
        return resp.json()["embeddings"][0]
    except Exception as e:
        print(f"❌ Embedding 生成失败: {e}")
        return None

def insert_single_doc(doc_data):
    client = MilvusClient(uri=MILVUS_URI)

    records = []

    # 1. 拆解 JSON 结构
    doc_title = doc_data["title"]
    for sec in doc_data["sections"]:
        sec_title = sec["title"]

        for block in sec["blocks"]:
            content = block.get("code", block.get("text", ""))
            if not content: continue

            # 2. 构造增强文本 (用于生成向量)
            # 我们把标题拼进去，增加向量的语义准确度
            text_to_embed = f"Document: {doc_title}\nSection: {sec_title}\nContent:\n{content}"

            # 3. 生成向量
            print(f"🔄 正在生成向量: {sec_title}...")
            vec = get_embedding(text_to_embed)
            if not vec: continue

            # 4. 构造符合 Milvus Schema 的数据行
            record = {
                "partition_tag": doc_data["partition"],
                "doc_id": doc_data["doc_id"],
                "doc_title": doc_title,
                "section_title": sec_title,
                "content_type": block["type"],  # 这里是 "code"
                "raw_content": content,         # 这里是 Zeek 代码原文
                "embedding": vec,               # 1024维向量
                "update_time": int(time.time())
            }
            records.append(record)

    # 5. 执行插入
    if records:
        res = client.insert(collection_name=COLLECTION_NAME, data=records)
        print(f"✅ 成功插入 {len(records)} 条数据！")
        print(f"   Insert IDs: {res['ids']}")
    else:
        print("⚠️ 没有生成有效数据。")

if __name__ == "__main__":
    insert_single_doc(ddos_data)