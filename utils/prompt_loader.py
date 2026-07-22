from utils.config_handler import prompts_config
from utils.path_tool import get_abs_path
from utils.LoggerHandler import logger


def _load_prompt(path_key: str) -> str:
    try:
        prompt_path = get_abs_path(prompts_config[path_key])
    except KeyError as e:
        logger.error(f"{e},没有找到{path_key}")
        return ""
    try:
        return open(prompt_path, "r", encoding="utf-8").read()
    except FileNotFoundError as e:
        logger.error(f"{e},提示词文件解析错误")
        return ""


def load_system_prompt() -> str:
    """导入系统提示词"""
    return _load_prompt("system_prompt_path")


def load_qa_chain_prompt() -> str:
    """导入 RAG 问答合成提示词"""
    return _load_prompt("qa_chain_prompt_path")


def load_retrieval_prompt() -> str:
    """导入检索 Query 改写提示词"""
    return _load_prompt("retrieval_prompt_path")


def load_classifier_prompt() -> str:
    """导入问题分类提示词"""
    return _load_prompt("classifier_prompt_path")


if __name__ == '__main__':
    load_system_prompt()
    load_qa_chain_prompt()
    load_retrieval_prompt()
    load_classifier_prompt()