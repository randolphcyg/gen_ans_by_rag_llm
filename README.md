# Zeek RAG

> 基于dify混合检索,langfuse链路追踪
> 需要先启动dify、langfuse


## 📂 项目结构

```text
gen_rag_by_zeek_doc/
├── rag_langchain_dify.py    # [入口] Flask API 服务，负责模型预加载与 HTTP 接口
├── test_api.py              # [测试] 自动化测试脚本，包含典型测试用例
├── requirements.txt         # 项目依赖清单
├── check_milvus.py          # 工具：检查知识库结构
├── download_model.py        # 工具：模型下载脚本
├── test_rag_index.py        # 工具：早期索引效果对比测试
└── test_langfuse.py         # 工具：Langfuse 连接性测试
```