import os

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from utils.config_handler import chroma_config
from model.factory import embed_model
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, dirlist, get_file_md5_hex
from utils.LoggerHandler import logger


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            persist_directory=get_abs_path(chroma_config["persist_directory"]),
            embedding_function=embed_model,
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len,
        )

    def retriever(self):
        """获取 MMR（最大边际相关性）向量检索器，减少重复文档"""
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": chroma_config["k"],
                "fetch_k": chroma_config["fetch_k"],
            },
        )

    def load_document(self):
        """加载 docs/ 下的知识库文档，切片后写入向量库（按 md5 去重，避免重复摄取）"""

        def check_md5_hex(md5_for_check: str):
            md5_store_path = get_abs_path(chroma_config["md5_hex_store"])
            if not os.path.exists(md5_store_path):
                open(md5_store_path, "w", encoding="utf-8").close()
                return False
            with open(md5_store_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == md5_for_check:
                        return True
            return False

        def save_md5_hex(md5_for_add: str):
            with open(get_abs_path(chroma_config["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_add + "\n")

        def get_file_documents(file_path: str):
            if file_path.endswith((".txt", ".md")):
                return txt_loader(file_path)
            if file_path.endswith(".pdf"):
                return pdf_loader(file_path)
            return []

        allow_file_path = dirlist(
            get_abs_path(chroma_config["data_path"]),
            tuple(chroma_config.get("allow_knowledge_file_type", ["txt", "pdf", "md"])),
        )

        for path in allow_file_path:
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"加载路径{path}已经在向量库中")
                continue

            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本,跳过")
                    continue

                splits_document: list[Document] = self.splitter.split_documents(documents)
                if not splits_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本,跳过")
                    continue

                self.vector_store.add_documents(splits_document)
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path}内容加载成功")
            except Exception as e:
                logger.error(f"[加载知识库]{path}失败:{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.retriever()
    res = retriever.invoke("SpaceX估值")
    for line in res:
        print(line.page_content)
        print("-" * 20)