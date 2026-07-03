"""
文档摄取脚本 —— 将 docs/ 目录下的文档切片、向量化并写入 Chroma 向量库。

用法：
    python ingest.py              # 增量：只摄取新文档
    python ingest.py --reset      # 清空后重新摄取全部文档
"""

import os
import sys
import argparse
from typing import List

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

from config import (
    DOCS_PATH,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    DASHSCOPE_API_KEY,
)


def load_documents(docs_path: str) -> List:
    """从 docs_path 加载所有支持的文档（PDF / TXT / MD）。

    Args:
        docs_path: 文档目录路径

    Returns:
        加载到的 LangChain Document 列表
    """
    supported_ext = {".pdf", ".txt", ".md"}
    all_docs = []

    if not os.path.isdir(docs_path):
        print(f"[警告] 目录不存在: {docs_path}")
        return all_docs

    for filename in os.listdir(docs_path):
        filepath = os.path.join(docs_path, filename)
        if not os.path.isfile(filepath):
            continue

        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext not in supported_ext:
            print(f"[跳过] 不支持的文件格式: {filename}")
            continue

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(filepath)
            elif ext == ".md":
                # 用纯文本方式读取 Markdown（避免依赖 heavy 的 unstructured）
                loader = TextLoader(filepath, encoding="utf-8")
            else:  # .txt
                loader = TextLoader(filepath, encoding="utf-8")

            docs = loader.load()
            # 注入来源文件名到 metadata
            for doc in docs:
                doc.metadata["source"] = filename
            all_docs.extend(docs)
            print(f"[加载] {filename} ({len(docs)} 页/段)")

        except Exception as e:
            print(f"[错误] 加载 {filename} 失败: {e}")

    return all_docs


def split_documents(docs: List) -> List:
    """使用 RecursiveCharacterTextSplitter 将文档切分为重叠的文本块。

    Args:
        docs: 原始 Document 列表

    Returns:
        切分后的 Document 列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[切片] {len(docs)} 个文档 → {len(chunks)} 个文本块 (chunk={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def create_embeddings() -> DashScopeEmbeddings:
    """创建阿里云 DashScope Embeddings 实例。

    Returns:
        DashScopeEmbeddings 实例
    """
    print(f"[Embedding] 使用 DashScope 模型: {EMBEDDING_MODEL}...")
    return DashScopeEmbeddings(
        model=EMBEDDING_MODEL,
        dashscope_api_key=DASHSCOPE_API_KEY,
    )


def ingest(reset: bool = False) -> None:
    """执行文档摄取主流程：加载 → 切片 → 向量化 → 写入 Chroma。

    Args:
        reset: 若为 True，先清空已有向量库再摄取；否则增量添加
    """
    # 1. 加载文档
    print("=" * 60)
    print("Step 1/4: 加载文档...")
    docs = load_documents(DOCS_PATH)
    if not docs:
        print("[终止] 未找到任何文档，请将文件放入 docs/ 目录后重试。")
        return

    # 2. 切片
    print("\nStep 2/4: 文本切片...")
    chunks = split_documents(docs)

    # 3. 创建 Embedding
    print("\nStep 3/4: 加载 Embedding 模型...")
    embeddings = create_embeddings()

    # 4. 写入 Chroma
    print("\nStep 4/4: 写入向量库...")
    if reset:
        print("[重置] 清空已有向量库...")
        Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH,
        ).delete_collection()
        print("[重置] 已清空。")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH,
    )
    print(f"[完成] 已写入 {vectorstore._collection.count()} 条向量到 {CHROMA_DB_PATH}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="SpaceX RAG 文档摄取 — 将 docs/ 目录内容索引到 Chroma 向量库"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="清空已有向量库后重新摄取全部文档",
    )
    args = parser.parse_args()
    ingest(reset=args.reset)


if __name__ == "__main__":
    main()
