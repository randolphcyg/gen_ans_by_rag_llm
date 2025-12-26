import os
import jieba
import logging
from typing import Set, List

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STOP_WORDS_FILE = "stop_words.txt"

class StopWordsManager:
    def __init__(self):
        self.stop_words: Set[str] = set()
        self.load_stop_words()
        # 核心：添加 Zeek 专用保留词，防止被 jieba 切碎
        self.special_terms = ["zeek_init", "zeek_done", "client_cert", "ssl_history", "conn_id"]
        for term in self.special_terms:
            jieba.add_word(term)

    def load_stop_words(self) -> None:
        if os.path.exists(STOP_WORDS_FILE):
            try:
                with open(STOP_WORDS_FILE, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word: self.stop_words.add(word.lower())
                logger.info(f"✅ 从文件加载停用词 {len(self.stop_words)} 个")
                return
            except Exception as e:
                logger.error(f"加载停用词文件失败: {e}")
        self._load_default_stop_words()

    def _load_default_stop_words(self) -> None:
        default_stop_words = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
            "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
            "什么", "一下", "吗", "呢", "吧", "啊", "哦", "嗯", "中", "段", "只", "又", "而",
            "what", "is", "the", "a", "an", "of", "for", "with", "and", "or", "in", "to"
        }
        self.stop_words = default_stop_words
        logger.info(f"✅ 加载默认停用词 {len(self.stop_words)} 个")

    def is_stop_word(self, word: str) -> bool:
        return word.lower().strip() in self.stop_words

    def filter_stop_words(self, text: str, min_length: int = 2) -> List[str]:
        # 1. 预处理：保留下划线，这对 Zeek 术语至关重要
        text = text.replace("_", "SUB_PLACEHOLDER")
        words = jieba.lcut(text)

        filtered_words = []
        for word in words:
            word = word.replace("SUB_PLACEHOLDER", "_").lower().strip()
            if (word not in self.stop_words and
                    len(word) >= min_length and
                    not word.isdigit()):
                filtered_words.append(word)
        return filtered_words

# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":
    # 1. 初始化停用词管理器
    stop_words_manager = StopWordsManager()

    # 3. 测试不同查询的关键词过滤
    test_queries = [
        "Zeek 是什么？",
        "介绍一下 Zeek 的核心功能",
        "Zeek 和 Suricata 的区别是什么？",
        "如何用 Zeek 脚本提取 HTTP 请求的 URL？"
    ]

    for query in test_queries:
        keywords = stop_words_manager.filter_stop_words(query)
        print(f"查询: {query}")
        print(f"有效关键词: {keywords}\n")