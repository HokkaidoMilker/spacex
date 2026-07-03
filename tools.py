"""
外部数据工具 —— 互联网搜索、股价查询、SpaceX 估值抓取。

所有工具返回格式化字符串，供 LLM Agent 消费。
"""

from typing import Optional
from config import WEB_SEARCH_MAX_RESULTS, SPACE_TICKERS


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
        title = r.get("title", "无标题")
        href = r.get("href", "")
        body = r.get("body", "无摘要")
        lines.append(f"\n{i}. {title}")
        lines.append(f"   URL: {href}")
        lines.append(f"   摘要: {body}")

    return "\n".join(lines)


def get_stock_price(ticker: str) -> str:
    """查询上市公司实时股价与基本信息。

    支持单个代码或逗号分隔的多个代码（如 "RKLB,LUNR"）。

    Args:
        ticker: 股票代码（如 RKLB），可逗号分隔多个

    Returns:
        格式化的股价信息文本
    """
    try:
        import yfinance as yf
    except ImportError:
        return "[错误] yfinance 未安装，请运行 pip install yfinance"

    # 支持逗号分隔的多个代码
    tickers = [t.strip().upper() for t in ticker.replace("，", ",").split(",") if t.strip()]

    if not tickers:
        return "[错误] 未提供有效的股票代码"

    results = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info

            if not info or "regularMarketPrice" not in info:
                results.append(f"❌ {t}: 未找到数据（可能已退市或代码错误）")
                continue

            price = info.get("regularMarketPrice", "N/A")
            prev_close = info.get("previousClose", "N/A")
            change_pct = info.get("regularMarketChangePercent", "N/A")
            if isinstance(change_pct, (int, float)):
                change_str = f"{change_pct:+.2f}%"
            else:
                change_str = str(change_pct)

            market_cap = info.get("marketCap", "N/A")
            if isinstance(market_cap, (int, float)) and market_cap != "N/A":
                if market_cap >= 1e12:
                    cap_str = f"${market_cap / 1e12:.2f}T"
                elif market_cap >= 1e9:
                    cap_str = f"${market_cap / 1e9:.2f}B"
                else:
                    cap_str = f"${market_cap / 1e6:.2f}M"
            else:
                cap_str = "N/A"

            high_52w = info.get("fiftyTwoWeekHigh", "N/A")
            low_52w = info.get("fiftyTwoWeekLow", "N/A")
            name = info.get("longName", t)
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")

            lines = [
                f"📊 {name} ({t})",
                f"   最新价: ${price}  |  涨跌幅: {change_str}  |  昨收: ${prev_close}",
                f"   市值: {cap_str}  |  52周高: ${high_52w}  |  52周低: ${low_52w}",
                f"   行业: {industry} / {sector}",
            ]
            results.append("\n".join(lines))

        except Exception as e:
            results.append(f"❌ {t}: 查询失败 - {e}")

    return "\n\n".join(results)


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


def get_related_stocks_overview() -> str:
    """获取航天/国防赛道所有相关上市公司股价概览。

    Returns:
        格式化的多股票对比文本
    """
    ticker_str = ",".join(SPACE_TICKERS)
    return get_stock_price(ticker_str)


# ============================================================
# LangChain Tool 包装函数（用于 Agent tool calling）
# 每个函数签名需匹配 LangChain Tool 的要求：单字符串输入，单字符串输出
# ============================================================

def search_web_tool(query: str) -> str:
    """搜索互联网获取最新信息。当知识库文档不足或询问时效性问题时使用。
    输入：搜索关键词（中文或英文）
    输出：搜索结果摘要（含URL）
    """
    return search_web(query)


def get_stock_price_tool(ticker: str) -> str:
    """查询上市公司实时股价。支持美股代码如 RKLB, LUNR, BA, LMT, NOC 等。
    输入：股票代码（单个如 "RKLB"，或多个逗号分隔如 "RKLB,LUNR"）
    输出：最新价、涨跌幅、市值、52周区间、行业信息
    """
    return get_stock_price(ticker)


def get_spacex_news_tool(_: str = "") -> str:
    """获取 SpaceX 最新估值、融资、业务进展（联网搜索）。
    无需参数，返回 SpaceX 相关的最新新闻和估值信息。
    """
    return get_spacex_news()


def get_related_stocks_overview_tool(_: str = "") -> str:
    """获取航天/国防赛道所有相关上市公司股价一览表。
    无需参数，返回 RKLB/LUNR/BA/LMT 等全部股票实时行情。
    """
    return get_related_stocks_overview()
