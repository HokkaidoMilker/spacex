import os
import hashlib

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from utils.LoggerHandler import logger


def get_file_md5_hex(file_path: str):
    """文件转化为 md5"""
    if not os.path.exists(file_path):
        logger.warning("文件路径不存在")
    if not os.path.isfile(file_path):
        logger.warning("请传入文件路径")

    md5_obj = hashlib.md5()
    chunk_size = 4096  # 避免文件过大爆内存

    try:
        with open(file_path, "rb") as f:
            chunk = f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            return md5_obj.hexdigest()
    except Exception:
        logger.error(f"获取文件md5失败,文件名{file_path}")


def dirlist(file_path: str, file_types: tuple[str, ...]):
    """列出目录下指定后缀的文件"""
    files = []
    if not os.path.isdir(file_path):
        logger.warning("请传入一个文件夹")
        return tuple()
    for f in os.listdir(file_path):
        if f.endswith(file_types):
            files.append(os.path.join(file_path, f))
    return tuple(files)


def pdf_loader(file_path: str, password=None) -> list[Document]:
    """pdf 加载器"""
    return PyPDFLoader(file_path, password).load()


def txt_loader(file_path: str, password=None) -> list[Document]:
    """txt / md 加载器"""
    return TextLoader(file_path, encoding="utf-8").load()