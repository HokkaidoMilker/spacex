from tools.common import clean_text


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
            name = clean_text(info.get("longName", t))
            sector = clean_text(info.get("sector", "N/A"))
            industry = clean_text(info.get("industry", "N/A"))

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
