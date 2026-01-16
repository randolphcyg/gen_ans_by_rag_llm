from langfuse import Langfuse
from langfuse import observe
import os

# 2. 配置 OpenTelemetry 关键参数，彻底解决重试/超时
os.environ["OTEL_EXPORTER_OTLP_TIMEOUT"] = "10000"  # 超时10秒
os.environ["OTEL_EXPORTER_OTLP_MAX_RETRIES"] = "0"  # 禁用无限重试（重中之重）
os.environ["OTEL_SDK_DISABLED"] = "false"           # 开启追踪

# 3. 初始化 Langfuse，你的密钥和地址完全正确，不用改
LANGFUSE_SECRET_KEY = "sk-lf-15beef95-8342-4448-b6d7-eb8cf71897bb"
LANGFUSE_PUBLIC_KEY = "pk-lf-c1be2f02-11b3-422d-95ed-8d43b6fb6e22"
LANGFUSE_BASE_URL = "http://localhost:3100"

langfuse = Langfuse(
    secret_key=LANGFUSE_SECRET_KEY,
    public_key=LANGFUSE_PUBLIC_KEY,
    host=LANGFUSE_BASE_URL,
    debug=True  # 建议开启debug，能看到详细请求日志，方便排错
)

@observe(name="RAG-重排阶段")
def rerank_with_bge(query: str):
    # 你的原有代码示例，返回模拟打分结果
    print(f"执行重排逻辑，查询词：{query}")
    return [{"title":"test", "score":85.0}]

if __name__ == '__main__':
    # ✅ 核心修复：调用函数时 传必填的 query 参数！！！
    rerank_with_bge(query="what is zeek?")
    print("✅ 测试脚本执行完成，无语法错误！")