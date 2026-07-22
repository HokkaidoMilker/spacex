from utils.config_handler import agent_config
from tools.common import clean_text

WEB_SEARCH_MAX_RESULTS = agent_config["web_search_max_results"]


def search_web(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> str:
    """使用 DuckDuckGo 搜索互联网，返回格式化的搜索结果。

    Args:
        query: 搜索关键词
        max_results: 最大返回条数

    Returns:
        格式化的搜索结果文本，含标题、摘要、URL
    """
    try:
        from ddgs import DDGS
    except ImportError:
        # 回退到旧包名
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "[错误] 搜索引擎未安装，请运行 pip install ddgs"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"[搜索失败] 网络连接异常，可能被防火墙阻止: {e}"

    if not results:
        return f"[无结果] 未找到与 '{query}' 相关的网页。"

    lines = [f"=== 互联网搜索结果: 「{query}」 ==="]
    for i, r in enumerate(results, 1):
        title = clean_text(r.get("title", "无标题"))
        href = r.get("href", "")
        body = clean_text(r.get("body", "无摘要"))
        lines.append(f"\n{i}. {title}")
        lines.append(f"   URL: {href}")
        lines.append(f"   摘要: {body}")

    return "\n".join(lines)
