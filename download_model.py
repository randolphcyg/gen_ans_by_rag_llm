import os
from huggingface_hub import snapshot_download


if __name__ == '__main__':
    local_dir = os.path.join(os.getcwd(), "models", 'BAAI/bge-reranker-v2-m3')

    if not os.path.exists(local_dir):
        print(f"正在下载模型到: {local_dir} ...")
        snapshot_download(
            repo_id="BAAI/bge-reranker-v2-m3",
            local_dir=local_dir,
        )
        print("下载完成！")
    else:
        print("模型已存在，跳过下载。")