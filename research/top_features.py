"""顶部信号：计算上证指数每日特征，并列出他 4 次顶部信号当天的数值。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lib import load  # noqa: E402
from replica_sentiment import rsi, kdj_k  # noqa: E402

HIS_TOPS = ["2015-04-28", "2015-05-28", "2015-06-16", "2026-06-23"]


def td_counts(c: pd.Series):
    """神奇九转（TD Setup）：连续收盘高于 4 天前收盘的计数；以及 TD Countdown（13）。"""
    up = (c > c.shift(4)).to_numpy()
    setup = np.zeros(len(c), dtype=int)
    for i in range(len(c)):
        setup[i] = setup[i - 1] + 1 if (i > 0 and up[i]) else (1 if up[i] else 0)
    return pd.Series(setup, index=c.index)


def features(df: pd.DataFrame) -> pd.DataFrame:
    c, o, h, l, v = df.close, df.open, df.high, df.low, df.volume
    r1 = c.pct_change()
    sig = r1.rolling(250).std().shift(1)
    MA = pd.DataFrame({n: c.rolling(n).mean() for n in (5, 10, 20, 30, 60)})
    f = pd.DataFrame(index=df.index)
    f["涨跌"] = r1
    f["涨跌σ"] = r1 / sig
    f["前日涨跌"] = r1.shift(1)
    f["高开幅度"] = o / c.shift(1) - 1
    f["实体"] = c / o - 1
    f["上影"] = (h - np.maximum(o, c)) / c.shift(1)
    f["距250日高"] = c / h.rolling(250).max() - 1
    f["距750日高"] = c / h.rolling(750, min_periods=500).max() - 1
    f["20日内创750日新高"] = (h >= h.rolling(750, min_periods=500).max()).rolling(20).max()
    f["均线发散度"] = (MA.max(axis=1) - MA.min(axis=1)) / c
    f["收盘vsMA5"] = c / MA[5] - 1
    f["收盘vsMA60"] = c / MA[60] - 1
    f["开盘在均线上方数"] = (MA.lt(o, axis=0)).sum(axis=1)
    f["收盘在均线下方数"] = (MA.gt(c, axis=0)).sum(axis=1)
    f["量比5"] = v / v.rolling(5).mean().shift(1)
    f["量比20"] = v / v.rolling(20).mean().shift(1)
    f["量在250日分位"] = v.rolling(250).rank(pct=True)
    rs5, k = rsi(c, 5), kdj_k(df)
    f["情绪(复刻)"] = (-11.59 + 0.885 * rs5 + 0.317 * k).clip(0, 100)
    f["前日情绪"] = f["情绪(复刻)"].shift(1)
    f["5日最高情绪"] = f["情绪(复刻)"].rolling(5).max()
    f["九转计数(前日)"] = td_counts(c).shift(1)
    f["20日涨幅"] = c / c.shift(20) - 1
    f["60日涨幅"] = c / c.shift(60) - 1
    f["250日涨幅"] = c / c.shift(250) - 1
    wk = c.resample("W-FRI").last()
    wrsi = rsi(wk, 6).reindex(c.index, method="ffill")
    f["周RSI6(上周)"] = wrsi.shift(5)
    return f


if __name__ == "__main__":
    df = load("sh000001")
    df = df[df.index >= "1993-01-01"]
    f = features(df)
    f.to_parquet(Path(__file__).resolve().parent.parent / "data" / "top_features_sh.parquet")
    pd.set_option("display.width", 250)
    pct = f[f.index >= "2005-01-01"].rank(pct=True)
    t = f.loc[HIS_TOPS].T
    t.columns = [d for d in HIS_TOPS]
    p = pct.loc[HIS_TOPS].T.add_suffix("分位")
    out = t.round(3).astype(str)
    for d in HIS_TOPS:
        out[d] = out[d] + " (" + (pct.loc[d] * 100).round(0).astype("Int64").astype(str) + ")"
    print("括号内：该值在 2005 年以来所有交易日中的百分位")
    print(out.to_string())
