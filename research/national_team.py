"""国家队入场识别和"防线"价位。

国家队式入场日：上交所宽基 ETF 单日净申购 ≥ MIN_YI 亿元，且沪深300 当天涨幅 < 2%（排除大涨日的追涨申购，
例如 2024-09-26/30）。相隔不超过 GAP 个交易日的入场日算同一轮。
每一轮给出：按申购额加权的收盘价（近似成本价）和入场那几天的最低点（入场期最低点）。

2015 年以来的检验（research 里的统计，样本只有 4 轮）：
- 2025-04 那一轮之后各指数从未收盘跌破入场期最低点；2024-07 那一轮第二天就跌破。
- 上证指数、上证50 的入场期最低点守住得多；创业板几乎每次都跌破，最多低于成本 15%。
ETF 份额次日才公布，所以入场日要到下一个交易日的数据里才能确认。
"""
import pandas as pd

MIN_YI = 150.0
MAX_RET = 0.02
GAP = 20


def episodes(etf_net_yi: pd.Series, hs300: pd.DataFrame) -> list[pd.DataFrame]:
    """返回每一轮的入场日（DataFrame：index 为日期，列 net 为当天净申购亿元）。"""
    ret = hs300.close / hs300.close.shift() - 1
    f = pd.DataFrame({"net": etf_net_yi, "ret": ret}).dropna()
    days = f[(f.net >= MIN_YI) & (f.ret < MAX_RET)]
    idx = hs300.index
    out, cur = [], []
    for t in days.index:
        if cur and idx.get_loc(t) - idx.get_loc(cur[-1]) > GAP:
            out.append(days.loc[cur, ["net"]])
            cur = []
        cur.append(t)
    if cur:
        out.append(days.loc[cur, ["net"]])
    return out


def defense(df: pd.DataFrame, ep: pd.DataFrame) -> dict:
    """某个指数在这一轮里的成本价、入场期最低点，以及之后是否收盘跌破。"""
    c = df.close.reindex(ep.index)
    cost = float((c * ep.net).sum() / ep.net.sum())
    low = float(df.low.reindex(ep.index).min())
    after = df.close[df.index > ep.index[-1]]
    last = float(df.close.iloc[-1])
    return {"nt_cost": cost, "nt_low": low, "nt_low_date": df.low.reindex(ep.index).idxmin(),
            "nt_cost_broken": bool((after < cost).any()), "nt_low_broken": bool((after < low).any()),
            "nt_vs_low": last / low - 1, "nt_vs_cost": last / cost - 1}
