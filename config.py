"""
SpaceX 股票分析问答助手 — 统一配置文件
所有参数集中管理，修改配置只需改此文件。
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Streamlit Secrets 支持（优先于 .env）
# ============================================================
def _get_secret(key: str, default: str = "") -> str:
    """尝试从 Streamlit Secrets 获取配置，回退到环境变量。

    Streamlit Cloud 通过 st.secrets 管理密钥，本地开发使用 .env。
    此函数在非 Streamlit 环境下安全回退到 os.getenv。

    Args:
        key: 配置键名
        default: 默认值

    Returns:
        配置值
    """
    try:
        import streamlit as st
        # st.secrets 可能是 dict-like 或 AttrDict，多种方式尝试
        secrets = st.secrets if hasattr(st, "secrets") else None
        if secrets is not None:
            # 方式 1: dict-style access
            try:
                val = secrets[key]
                if val:
                    return str(val)
            except (KeyError, TypeError):
                pass
            # 方式 2: attribute-style access
            try:
                val = getattr(secrets, key, None)
                if val:
                    return str(val)
            except (TypeError, AttributeError):
                pass
    except Exception:
        pass
    return os.getenv(key, default)


# ============================================================
# LLM 模型（Groq Cloud — 免费 tier，兼容 OpenAI SDK）
# ============================================================
# 注册获取免费 API Key: https://console.groq.com
# 免费额度：每分钟 30 请求，每天 14,400 请求（日常使用完全够）
LLM_MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY = _get_secret("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ============================================================
# Embedding 模型（免费本地 FastEmbed，无需 API Key）
# ============================================================
# FastEmbed 使用 ONNX 模型本地运行，首次运行会自动下载模型（~100MB）
# 支持中英文：BAAI/bge-small-zh-v1.5
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# ============================================================
# 路径配置
# ============================================================
CHROMA_DB_PATH = "./chroma_db"
DOCS_PATH = "./docs"
PROMPTS_PATH = "./prompts"

# ============================================================
# 检索参数
# ============================================================
RETRIEVER_K = 5            # MMR 最终返回文档数
RETRIEVER_FETCH_K = 20     # MMR 候选池大小
CHUNK_SIZE = 800           # 文档切片大小（字符）
CHUNK_OVERLAP = 100        # 切片重叠量（字符）

# ============================================================
# 对话参数
# ============================================================
MAX_HISTORY_TURNS = 6      # 保留最近 N 轮对话历史

# ============================================================
# Chroma 配置
# ============================================================
CHROMA_COLLECTION_NAME = "spacex_docs"

# ============================================================
# 工具开关
# ============================================================
ENABLE_WEB_SEARCH = True       # 是否启用互联网搜索（DuckDuckGo）
ENABLE_STOCK_LOOKUP = True     # 是否启用股价查询（yfinance）
WEB_SEARCH_MAX_RESULTS = 5     # 每次搜索返回条数

# 相关上市公司股票代码（航天/国防赛道）
SPACE_TICKERS = [
    "RKLB",   # Rocket Lab
    "LUNR",   # Intuitive Machines (月球着陆器)
    "RDW",    # Redwire (太空基础设施)
    "PL",     # Planet Labs (地球观测)
    "SPCE",   # Virgin Galactic (太空旅游)
    "BA",     # Boeing (ULA 合资方)
    "LMT",    # Lockheed Martin
    "NOC",    # Northrop Grumman
    "RTX",    # RTX (Raytheon)
]

# ============================================================
# 辅助函数
# ============================================================
def get_embeddings():
    """创建 FastEmbed 本地 Embedding 实例（免费，无需 API Key）。

    FastEmbed 使用 ONNX 模型本地运行，首次调用会自动下载模型缓存。
    模型约 100MB，下载一次后缓存复用。

    Returns:
        FastEmbedEmbeddings 实例，兼容 LangChain 检索器接口
    """
    from langchain_community.embeddings import FastEmbedEmbeddings
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


def load_prompt(filename: str) -> str:
    """从 prompts/ 目录加载提示词文件内容。

    Args:
        filename: 提示词文件名（如 'system.txt'）

    Returns:
        提示词文本内容（UTF-8 编码）

    Raises:
        FileNotFoundError: 若文件不存在
    """
    filepath = os.path.join(PROMPTS_PATH, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
