from utils.config_handler import agent_config
from tools.stock_price import get_stock_price

SPACE_TICKERS = agent_config["space_tickers"]


def get_related_stocks_overview() -> str:
    """获取航天/国防赛道所有相关上市公司股价概览。

    Returns:
        格式化的多股票对比文本
    """
    ticker_str = ",".join(SPACE_TICKERS)
    return get_stock_price(ticker_str)
