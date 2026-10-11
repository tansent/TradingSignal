"""融资余额 + 成交额过热研究：数据下载与指标计算。

数据：沪深两市每日融资余额/融资买入额（2010-03 起）、两融账户统计（2012-09 起）、
     全 A 成交额（breadth_daily.parquet，来自通达信全部个股，2010 起）。
所有"相对历史"的指标都只用过去的数据（滚动窗口），不含未来信息。
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def download():
    os.environ["NO_PROXY"] = "*"
    import akshare as ak
    sh = ak.macro_china_market_margin_sh()
    sz = ak.macro_china_market_margin_sz()
    acc = ak.stock_margin_account_info()
    sh.to_csv(DATA / "margin_sh_full.csv", index=False)
    sz.to_csv(DATA / "margin_sz_full.csv", index=False)
    acc.to_csv(DATA / "margin_account_info.csv", index=False)


def load_margin() -> pd.DataFrame:
    """全市场融资数据用两融账户统计（2012-09 起，逐日完整，单位亿元）。
    交易所分市场数据深市在 2024-2025 年缺 177 天，不用来合计。"""
    acc = pd.read_csv(DATA / "margin_account_info.csv", parse_dates=["日期"]).set_index("日期").sort_index()
    m = pd.DataFrame({
        "融资余额": acc["融资余额"] * 1e8,
        "融资买入额": acc["融资买入额"] * 1e8,
        "平均维持担保比例": acc["平均维持担保比例"],
        "参与交易的投资者数量": acc["参与交易的投资者数量"],
    })
    return m


def indicators() -> pd.DataFrame:
    b = pd.read_parquet(DATA / "breadth_daily.parquet")
    a = b[b.board == "all"].set_index("date").sort_index()
    amt = a["amount"]
    m = load_margin().reindex(amt.index).ffill(limit=3)
    f = pd.DataFrame(index=amt.index)
    f["成交额(亿)"] = amt / 1e8
    f["成交额/年均"] = amt.rolling(20).mean() / amt.rolling(250, min_periods=120).mean()
    f["融资余额(亿)"] = m["融资余额"] / 1e8
    f["融资余额20日增速"] = m["融资余额"].pct_change(20)
    f["融资余额60日增速"] = m["融资余额"].pct_change(60)
    f["融资买入占比"] = (m["融资买入额"] / amt).rolling(5).mean()
    f["融资余额/20日成交额"] = m["融资余额"] / amt.rolling(20).mean()
    f["维持担保比例"] = m["平均维持担保比例"]
    for col in ["成交额/年均", "融资余额20日增速", "融资余额60日增速", "融资买入占比", "融资余额/20日成交额"]:
        f[col + "_3年分位"] = f[col].rolling(750, min_periods=250).rank(pct=True)
    return f


if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:
        download()
    f = indicators()
    f.to_parquet(DATA / "margin_turnover.parquet")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    days = ["2014-12-09", "2015-04-28", "2015-05-28", "2015-06-12", "2015-06-16", "2018-01-24", "2019-04-08",
            "2020-07-09", "2021-02-18", "2024-10-08", "2025-08-27", "2026-02-02", "2026-05-14", "2026-06-23",
            "2026-10-09"]
    cols = ["成交额(亿)", "成交额/年均", "融资余额(亿)", "融资余额20日增速", "融资余额60日增速", "融资买入占比",
            "融资余额/20日成交额", "维持担保比例", "成交额/年均_3年分位", "融资买入占比_3年分位", "融资余额60日增速_3年分位"]
    print(f.reindex(pd.to_datetime(days))[cols].round(3).to_string())
