"""文档摄取入口 —— 读取 docs/ 目录，切片、向量化并写入 Chroma 向量库。

实际逻辑封装在 rag/vector_store.py 的 VectorStoreService 中（按文件 md5 去重，增量摄取）。

用法：
    python ingest.py
"""

from rag.vector_store import VectorStoreService


def ingest() -> None:
    """执行文档摄取：加载 docs/ → 切片 → 向量化 → 写入 Chroma"""
    vector_store_service = VectorStoreService()
    vector_store_service.load_document()


if __name__ == '__main__':
    ingest()