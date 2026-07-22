import os
import yaml
from dotenv import load_dotenv

from utils.path_tool import get_abs_path

load_dotenv()


def load_rag_config(config_path: str = get_abs_path("config/rag.yml"), encoding='utf-8'):
    with open(config_path, encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(config_path: str = get_abs_path("config/chroma.yml"), encoding='utf-8'):
    with open(config_path, encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str = get_abs_path("config/prompts.yml"), encoding='utf-8'):
    with open(config_path, encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str = get_abs_path("config/agent.yml"), encoding='utf-8'):
    with open(config_path, encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


rag_config = load_rag_config()
chroma_config = load_chroma_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()

# ============================================================
# 密钥类配置统一从环境变量读取，禁止硬编码到代码或 yml 中
# ============================================================
# 阿里云百炼（DashScope）—— 聊天模型（OpenAI 兼容接口）与 Embedding 模型共用同一个 Key
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
dashscope_base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
access_password = os.getenv("ACCESS_PASSWORD", "yuxinlin")