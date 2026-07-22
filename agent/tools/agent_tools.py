from langchain_core.tools import tool

from rag.rag_service import RagSummerizeService
from tools.web_search import search_web
from tools.stock_price import get_stock_price
from tools.spacex_news import get_spacex_news
from tools.related_stocks import get_related_stocks_overview
from utils.config_handler import agent_config

rag = RagSummerizeService()


@tool(description="从SpaceX知识库中检索参考资料，用于回答商业模式、发射数据、竞争对手、财务指标等问题，入参为检索词query")
def retrieve_knowledge(query: str) -> str:
    return rag.rag_summerize(query)


@tool(description="搜索互联网获取最新的估值、发射新闻等时效信息，入参为搜索关键词query")
def search_web_online(query: str) -> str:
    return search_web(query, max_results=agent_config["web_search_max_results"])


@tool(description="查询上市公司实时股价，入参为股票代码ticker，支持RKLB/LUNR/BA/LMT/NOC/SPCE等，多个代码用逗号分隔")
def get_stock_price_info(ticker: str) -> str:
    return get_stock_price(ticker)


@tool(description="无入参，联网获取SpaceX最新估值、融资、Starlink与星舰研发进展")
def get_spacex_latest_news() -> str:
    return get_spacex_news()


@tool(description="无入参，获取航天/国防赛道相关上市公司股价一览（RKLB、LUNR、BA、LMT等）")
def get_related_stocks() -> str:
    return get_related_stocks_overview()