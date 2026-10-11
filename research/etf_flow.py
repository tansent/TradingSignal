"""宽基 ETF 同时放量：疑似国家队（中央汇金等）出手的识别。

用成交额（amount）而不是成交量，避免 ETF 份额拆分的影响。
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "research")
from lib import load, forward_stats  # noqa: E402

NAMES = {"sh510300": "华泰300", "sh510310": "易方达300", "sh510330": "华夏300", "sz159919": "嘉实300",
         "sh510050": "华夏50", "sh510500": "南方500", "sh512100": "南方1000", "sz159915": "易方达创业板",
         "sh588000": "华夏科创50", "sh510180": "华安180"}


def panel() -> pd.DataFrame:
    """每只 ETF 的成交额相对前 20 日均值的倍数。"""
    out = {}
    for f in glob.glob("data/etf/*.csv"):
        code = Path(f).stem
        df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        amt = df["amount"].astype(float) if "amount" in df else df["volume"].astype(float) * df["close"]
        out[NAMES.get(code, code)] = amt / amt.rolling(20).mean().shift(1)
    return pd.DataFrame(out)


if __name__ == "__main__":
    p = panel()
    n = p.notna().sum(axis=1)
    cnt2 = (p >= 2).sum(axis=1)                 # 成交额 >= 2 倍的 ETF 数
    share = cnt2 / n
    med = p.median(axis=1)
    days = ["2015-07-06", "2015-07-08", "2015-08-26", "2016-01-28", "2018-10-19", "2020-02-04", "2023-10-23",
            "2024-01-24", "2024-02-05", "2024-02-06", "2025-04-07", "2025-04-08", "2026-10-08", "2026-10-09"]
    print("日期          中位倍数  >=2倍的ETF占比   各ETF倍数")
    for d in days:
        d = pd.Timestamp(d)
        if d in p.index:
            row = p.loc[d].dropna().round(1)
            print(f"{d.date()}   {med[d]:5.2f}    {share[d]:5.0%}      " + " ".join(f"{k}{v}" for k, v in row.items()))
    sig = (share >= 0.7) & (med >= 2.0)
    s = sig[sig].index
    print(f"\n历史上 >=70% 的宽基ETF成交额同时翻倍（中位>=2倍）的日子：{len(s)} 天")
    hs = load("sh000300")
    r = forward_stats(hs, [d for d in s if d in hs.index], horizons=(20, 60))
    r["chg_day"] = [hs.close.pct_change().loc[d] for d in r.date]
    print(r.assign(date=r.date.dt.date)[["date", "chg_day", "ret20", "ret60", "mae60"]].to_string(index=False, float_format=lambda v: f"{v:+.1%}"))
