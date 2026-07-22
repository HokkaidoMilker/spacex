from tools.web_search import search_web


def get_spacex_news() -> str:
    """专门搜索 SpaceX 最新估值、融资、业务进展新闻。

    注意：SpaceX 是私有公司，搜索时加上 "private company" 以过滤掉
    虚假的 SpaceX IPO/股价新闻。

    Returns:
        格式化的 SpaceX 实时信息文本
    """
    queries = [
        "SpaceX private company valuation 2025 2026 latest funding round -IPO -stock",
        "SpaceX Starlink subscribers revenue 2025 2026",
        "SpaceX Starship launch progress 2025 2026",
    ]

    all_lines = ["🚀 SpaceX 实时信息（联网搜索）\n"]

    for q in queries:
        result = search_web(q, max_results=3)
        # 去掉第一行标题，避免重复
        lines = result.split("\n")
        # 跳过 "=== 互联网搜索" 标题行
        content_lines = [l for l in lines if not l.startswith("===")]
        all_lines.extend(content_lines)
        all_lines.append("")

    return "\n".join(all_lines)
