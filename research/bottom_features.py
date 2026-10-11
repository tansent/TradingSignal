"""底部信号：各指数每日特征（收盘口径 + 盘中最低价口径），并接入市场宽度和情绪温度。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from replica_sentiment import kdj_k, rsi  # noqa: E402
from tdx_reader import read_day  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V = Path("D:/TDXapp/vipdoc")
INDEXES = {"上证指数": "sh000001", "沪深300": "sh000300", "深证成指": "sz399001", "创业板指": "sz399006",
           "中证500": "sh000905", "中证1000": "sh000852"}


def load_index(code: str) -> pd.DataFrame:
    return read_day(V / code[:2] / "lday" / f"{code}.day")


def senti_with(df: pd.DataFrame, p: pd.Series) -> pd.Series:
    """复刻情绪，但当天价格用 p（盘中口径时 p=最低价）。"""
    c = df.close
    lc = c.shift(1)
    d = c - lc
    up = d.clip(lower=0).ewm(alpha=1 / 5, adjust=False).mean().shift(1)
    ab = d.abs().ewm(alpha=1 / 5, adjust=False).mean().shift(1)
    dp = p - lc
    rsi5 = (up * 0.8 + dp.clip(lower=0) / 5) / (ab * 0.8 + dp.abs() / 5) * 100
    ll = pd.concat([df.low.rolling(9).min(), p], axis=1).min(axis=1)
    hh = df.high.rolling(9).max()
    rsv = (p - ll) / (hh - ll) * 100
    k_prev = kdj_k(df).shift(1)
    k = k_prev * 2 / 3 + rsv / 3
    return (-11.59 + 0.885 * rsi5 + 0.317 * k).clip(0, 100)


def features(df: pd.DataFrame) -> pd.DataFrame:
    c, o, h, l, v = df.close, df.open, df.high, df.low, df.volume
    r1 = c.pct_change()
    sig = r1.rolling(250, min_periods=120).std().shift(1)
    f = pd.DataFrame(index=df.index)
    f["情绪_收盘"] = (-11.59 + 0.885 * rsi(c, 5) + 0.317 * kdj_k(df)).clip(0, 100)
    f["情绪_盘中低"] = senti_with(df, l)
    for p, tag in ((c, "收盘"), (l, "盘中低")):
        f[f"距120日高_{tag}"] = p / h.rolling(120).max() - 1
        f[f"距120日高σ_{tag}"] = f[f"距120日高_{tag}"] / sig
        f[f"20日跌幅σ_{tag}"] = (p / c.shift(20) - 1) / (sig * np.sqrt(20))
        f[f"5日跌幅σ_{tag}"] = (p / c.shift(5) - 1) / (sig * np.sqrt(5))
    f["当日涨跌"] = r1
    f["盘中最大跌幅"] = l / c.shift(1) - 1
    f["下影"] = (np.minimum(o, c) - l) / c.shift(1)
    f["收在区间位置"] = (c - l) / (h - l).replace(0, np.nan)
    f["量比20"] = v / v.rolling(20).mean().shift(1)
    f["距120日高天数"] = h.rolling(120).apply(lambda x: len(x) - 1 - np.argmax(x), raw=True)
    ma250 = c.rolling(250).mean()
    f["环境"] = np.select([(c > ma250) & (ma250 > ma250.shift(20)), (c < ma250) & (ma250 < ma250.shift(20))], ["牛", "熊"], "震荡")
    return f


def market_context() -> pd.DataFrame:
    b = pd.read_parquet(ROOT / "data" / "breadth_daily.parquet")
    a = b[b.board == "all"].set_index("date").sort_index()
    t = pd.read_parquet(ROOT / "data" / "temperature_v1.parquet")
    return pd.DataFrame({
        "下跌占比": a.down / a.n, "跌停占比": a.dt / a.n, "60日新低占比": a.nl60 / a.n,
        "站上20日线": a.above20 / a.n, "短线温度": t["短线温度"], "中线温度": t["中线温度"],
    })


def all_features() -> dict[str, pd.DataFrame]:
    ctx = market_context()
    return {name: features(load_index(code)).join(ctx, how="left") for name, code in INDEXES.items()}


if __name__ == "__main__":
    F = all_features()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    dates = ["2015-08-26", "2016-01-28", "2018-10-18", "2019-01-03", "2020-02-03", "2022-04-26", "2022-10-31",
             "2024-02-05", "2024-09-18", "2025-04-07", "2026-10-09"]
    cols = ["情绪_收盘", "情绪_盘中低", "距120日高_盘中低", "距120日高σ_盘中低", "20日跌幅σ_盘中低", "盘中最大跌幅",
            "下影", "量比20", "距120日高天数", "下跌占比", "跌停占比", "60日新低占比", "中线温度", "环境"]
    for name in ("沪深300", "创业板指"):
        print(f"\n===== {name} =====")
        x = F[name].reindex(pd.to_datetime(dates))[cols]
        print(x.round(2).to_string())
