import requests
import json
import time

# API 地址
API_URL = "http://localhost:5000/chat"

# 🎯 核心测试集：涵盖细节、编程、排错、概念
TEST_CASES = [
    {
        "type": "概念/默认值",
        "query": "What is zeek?"
    },
    {
        "type": "概念/默认值",
        "query": "啥是zeek?"
    },
    {
        "type": "概念/默认值",
        "query": "why zeek?"
    },
    {
        "type": "概念/默认值",
        "query": "为什么选 zeek?"
    },
    {
        "type": "细节检索 (父子索引强项)",
        "query": "Explain the meaning of the `history` field string 'ShADadFf' in `conn.log`."
    },
    {
        "type": "代码编写 (考察 Module 依赖)",
        "query": "Write a script to handle `ssh_auth_successful` event. Do I need to load any module?"
    },
    {
        "type": "故障排查 (考察逻辑分析)",
        "query": "Why is my `notice.log` empty even though I see attacks in `conn.log`?"
    },
    {
        "type": "概念/默认值",
        "query": "What is the default value of `Log::default_rotation_interval`?"
    }
]

def run_test():
    print(f"🚀 开始对 API [{API_URL}] 进行批量测试...\n")

    for i, case in enumerate(TEST_CASES):
        q_type = case['type']
        query = case['query']

        print(f"{'='*30} Test {i+1}: {q_type} {'='*30}")
        print(f"❓ 问题: {query}")

        try:
            start_t = time.time()
            resp = requests.post(API_URL, json={"query": query}, timeout=60)
            cost = time.time() - start_t

            if resp.status_code == 200:
                data = resp.json()
                answer = data.get('answer', '无回答')
                refs = data.get('references', [])

                print(f"⏱️ 服务端耗时: {data.get('cost_time', cost):.2f}s")
                print(f"\n🤖 完整回答:\n{answer}")

                print(f"\n📚 参考引用 (共召回 {len(refs)} 条, 显示 Top 5):")

                # 只打印前 5 条，避免刷屏，但你可以根据需要改
                for j, ref in enumerate(refs):
                    content = ref['content'].strip()
                    score = ref.get('score', 0)
                    doc_id = ref.get('doc_id', 'N/A')

                    # 预览处理：太长则截断中间
                    lines = content.split('\n')
                    if len(lines) > 6:
                        preview_text = "\n".join(lines[:3]) + \
                                       f"\n\n... [省略 {len(lines)-6} 行] ...\n\n" + \
                                       "\n".join(lines[-3:])
                    else:
                        preview_text = content

                    print(f"   ┌──Ref [{j+1}] Score: {score:.4f} | ID: {doc_id} ──────────")
                    # 增加缩进
                    formatted_content = "\n".join([f"   │ {line}" for line in preview_text.split('\n')])
                    print(formatted_content)
                    print(f"   └─────────────────────────────────────────────────────")
            else:
                print(f"❌ 请求失败: Status {resp.status_code} | {resp.text}")

        except Exception as e:
            print(f"❌ 发生异常: {e}")

        print("\n")
        time.sleep(1)

if __name__ == "__main__":
    run_test()