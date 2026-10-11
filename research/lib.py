"""数据加载与特征计算（研究用）。

约定：所有特征只用当日及以前的数据，不引用未来。
P 代表"判断用价格"：收盘口径用 CLOSE，盘中口径用当日 LOW（底）/ HIGH（顶）近似午间极值。
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

NAMES = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
    "sh000300": "沪深300",
    "sh000688": "科创50",
    "sh000905": "中证500",
    "sh000852": "中证1000",
    "sh000685": "科创芯片",
    "ths_半导体": "同花顺半导体",
    "ths_通信设备": "同花顺通信设备",
}


def load(code: str) -> pd.DataFrame:
    if code.startswith("ths_"):
        df = pd.read_csv(DATA / f"{code}.csv")
        df = df.rename(columns={"日期": "date", "开盘价": "open", "最高价": "high",
                                "最低价": "low", "收盘价": "close", "成交量": "volume"})
    else:
        df = pd.read_csv(DATA / f"{code}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "open", "high", "low", "close", "volume"]].set_index("date").sort_index()
    df = df.astype(float)
    return _append_today_from_30m(code, df)


def _append_today_from_30m(code: str, df: pd.DataFrame) -> pd.DataFrame:
    """新浪深市指数日线当天未更新时，用 30 分钟线合成当天日线（成交量按前一日比例换算）。"""
    f = DATA / f"{code}_30m.csv"
    if not f.exists():
        return df
    m = pd.read_csv(f)
    m["d"] = pd.to_datetime(m["day"].astype(str).str[:10])
    last_m = m["d"].max()
    if last_m <= df.index[-1]:
        return df
    prev = df.index[-1]
    pm = m[m["d"] == prev]
    scale = df.loc[prev, "volume"] / pm["volume"].astype(float).sum() if len(pm) else np.nan
    t = m[m["d"] == last_m]
    df.loc[last_m] = [t["open"].iloc[0], t["high"].max(), t["low"].min(), t["close"].iloc[-1],
                      t["volume"].astype(float).sum() * scale]
    return df


def rsi(c: pd.Series, p: pd.Series, n: int) -> pd.Series:
    """通达信/同花顺口径 RSI：SMA(MAX(C-LC,0),N,1)/SMA(ABS(C-LC),N,1)*100。
    历史各根用收盘价，只有当前这一根用 P（盘中口径下 P 是盘中价格）。"""
    lc = c.shift(1)
    d = c - lc
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean().shift(1)
    ab = d.abs().ewm(alpha=1 / n, adjust=False).mean().shift(1)
    dp = p - lc
    up = up * (1 - 1 / n) + dp.clip(lower=0) / n
    ab = ab * (1 - 1 / n) + dp.abs() / n
    return up / ab * 100


def ma_with(c: pd.Series, p: pd.Series, n: int) -> pd.Series:
    """N 日均线，但当前这一根用 P 代替收盘价。"""
    return (c.rolling(n).sum() - c + p) / n


def features(df: pd.DataFrame, mode: str = "close") -> pd.DataFrame:
    """mode='close' 收盘口径；mode='intraday' 盘中口径（底用 LOW，顶用 HIGH，成交量只用到前一日）。"""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    lc = c.shift(1)
    pb = l if mode == "intraday" else c   # 判断底部用的价格
    pt = h if mode == "intraday" else c   # 判断顶部用的价格
    vv = v.shift(1) if mode == "intraday" else v

    f = pd.DataFrame(index=df.index)
    ret1 = c.pct_change()
    f["vol250"] = ret1.rolling(250, min_periods=120).std().shift(1)  # 日波动率（过去一年，不含当天）
    sig = f["vol250"]

    hh120 = h.rolling(120, min_periods=60).max()
    ll120 = l.rolling(120, min_periods=60).min()
    f["dd120"] = pb / hh120 - 1                                       # 距120日高点回撤
    f["z_dd120"] = f["dd120"] / sig                                   # 以日波动率为单位
    f["days_from_high"] = _bars_since_extreme(h, 120, "max")
    for n in (5, 10, 20, 60):
        f[f"r{n}"] = pb / c.shift(n) - 1
        f[f"z{n}"] = f[f"r{n}"] / (sig * np.sqrt(n))                  # 波动率归一化的 N 日跌幅
    for n in (20, 60, 250):
        ma = ma_with(c, pb, n)
        f[f"bias{n}"] = pb / ma - 1
        f[f"zbias{n}"] = f[f"bias{n}"] / (sig * np.sqrt(n))
    f["rsi6"] = rsi(c, pb, 6)
    f["rsi14"] = rsi(c, pb, 14)
    f["down20"] = (ret1 < 0).rolling(20).mean()                      # 近20日下跌天数占比
    f["vr20"] = vv / v.rolling(20).mean().shift(1)                    # 量比（相对前20日均量）
    f["vr120"] = vv / v.rolling(120).mean().shift(1)
    # 顶部用
    f["ru120"] = pt / ll120 - 1
    f["z_ru120"] = f["ru120"] / sig
    f["tbias20"] = pt / ma_with(c, pt, 20) - 1
    f["tzbias60"] = (pt / ma_with(c, pt, 60) - 1) / (sig * np.sqrt(60))
    f["trsi6"] = rsi(c, pt, 6)
    # 环境
    ma250 = c.rolling(250).mean()
    slope = ma250 / ma250.shift(20) - 1
    f["regime"] = np.select([(c > ma250) & (slope > 0), (c < ma250) & (slope < 0)], ["牛", "熊"], "震荡")
    return f


def _bars_since_extreme(s: pd.Series, n: int, how: str) -> pd.Series:
    arr = s.to_numpy()
    out = np.full(len(arr), np.nan)
    for i in range(n - 1, len(arr)):
        w = arr[i - n + 1:i + 1]
        j = np.argmax(w) if how == "max" else np.argmin(w)
        out[i] = n - 1 - j
    return pd.Series(out, index=s.index)


def forward_stats(df: pd.DataFrame, dates, horizons=(5, 20, 60)) -> pd.DataFrame:
    """信号日收盘后买入：未来 N 日收益、期间最大不利幅度（最低价相对信号日收盘）。"""
    c, l, h = df["close"], df["low"], df["high"]
    idx = df.index
    rows = []
    for d in dates:
        i = idx.get_loc(d)
        r = {"date": d, "close": c.iloc[i]}
        for n in horizons:
            if i + n < len(idx):
                r[f"ret{n}"] = c.iloc[i + n] / c.iloc[i] - 1
                r[f"mae{n}"] = l.iloc[i + 1:i + n + 1].min() / c.iloc[i] - 1
                r[f"mfe{n}"] = h.iloc[i + 1:i + n + 1].max() / c.iloc[i] - 1
        rows.append(r)
    return pd.DataFrame(rows)
