import requests
import json
import time

# API 地址
API_URL = "http://localhost:5000/chat"

# 🎯 核心测试集：涵盖细节、编程、排错、概念
TEST_CASES = [
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

    total_start = time.time()

    for i, case in enumerate(TEST_CASES):
        q_type = case['type']
        query = case['query']

        print(f"{'='*20} Test {i+1}: {q_type} {'='*20}")
        print(f"❓ 问题: {query}")

        try:
            # 发送请求
            start_t = time.time()
            resp = requests.post(API_URL, json={"query": query}, timeout=60)

            if resp.status_code == 200:
                data = resp.json()
                cost = data.get('cost_seconds', 0)
                answer = data.get('answer', '无回答')
                refs = data.get('references', [])

                # 打印结果
                print(f"⏱️ 服务端耗时: {cost}s")
                print(f"🤖 回答预览:\n{answer[:300]}..." if len(answer) > 300 else f"🤖 回答:\n{answer}")

                print(f"\n📚 参考引用 ({len(refs)}条):")
                for j, ref in enumerate(refs):
                    # 打印分数和前50个字符
                    print(f"   [{j+1}] Score: {ref['score']:.4f} | {ref['content'][:60].replace(chr(10), ' ')}...")
            else:
                print(f"❌ 请求失败: Status {resp.status_code} | {resp.text}")

        except requests.exceptions.ConnectionError:
            print("❌ 无法连接到服务器。请确认 'python app.py' 正在运行！")
            break
        except Exception as e:
            print(f"❌ 发生异常: {e}")

        print("\n")
        time.sleep(1) # 稍微停顿，方便观察

    total_cost = time.time() - total_start
    print(f"🏁 所有测试完成，总耗时: {total_cost:.2f}s")

if __name__ == "__main__":
    run_test()