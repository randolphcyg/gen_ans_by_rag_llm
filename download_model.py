import os
from huggingface_hub import snapshot_download


if __name__ == '__main__':
    local_dir = os.path.join(os.getcwd(), "models", "bge-reranker-base")

    if not os.path.exists(local_dir):
        print(f"正在下载模型到: {local_dir} ...")
        snapshot_download(
            repo_id="BAAI/bge-reranker-base",
            local_dir=local_dir,
            local_dir_use_symlinks=False  # 关键：确保下载的是真实文件而不是软链接
        )
        print("下载完成！")
    else:
        print("模型已存在，跳过下载。")