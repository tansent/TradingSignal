"""顶部信号：几种思路对比。信号日收盘后，看未来 20/60 日走势（希望下跌）。"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "research")
from lib import NAMES, load, forward_stats
from backtest import INDEXES, START, first_trigger

def build(df):
    c, h, l, v = df.close, df.high, df.low, df.volume
    r1 = c.pct_change(); sig = r1.rolling(250, min_periods=120).std().shift(1)
    z1 = r1 / sig
    hh250 = h.rolling(250, min_periods=120).max(); ll250 = l.rolling(250, min_periods=120).min()
    at_high = h >= hh250                                   # 当天创 250 日新高
    near_high = at_high.rolling(10).max().astype(bool)     # 最近 10 天内创过 250 日新高
    rally = (hh250 / ll250 - 1) / (sig * np.sqrt(250))     # 一年内涨幅（波动率归一化）
    ma250 = c.rolling(250).mean(); up = (c > ma250) & (ma250 > ma250.shift(20))
    rng = (h - l).replace(0, np.nan); weak_close = (c - l) / rng <= 0.3
    big_drop = (z1 <= -2) & weak_close
    prev_drop = big_drop.shift(1).rolling(9).max().fillna(0).astype(bool)   # 之前 9 天内有过大阴线
    # 失败的冲高：最近一次大阴线之后的最高价没有超过它之前的高点
    v = {}
    v["T1 高位大阴线(首次)"] = big_drop & near_high & up & (rally >= 1.0)
    v["T2 高位第二根大阴线"] = big_drop & prev_drop & near_high & up & (rally >= 1.0)
    v["T3 T2且放量"] = v["T2 高位第二根大阴线"] & (df.volume >= df.volume.rolling(20).mean().shift(1))
    v["T0 高位任意大跌日(对照)"] = (z1 <= -2) & near_high
    return v, df.index

def evaluate():
    res = {}
    for code in INDEXES:
        df = load(code)
        v, _ = build(df)
        for name, cond in v.items():
            s = first_trigger(cond, 20); s = s[s.index >= START]
            st = forward_stats(df, s[s].index, horizons=(20, 60)); st["index"] = NAMES[code]
            res.setdefault(name, []).append(st)
    print(f"{'规则':<20}{'次数':>5}{'20日均':>8}{'20日跌概率':>9}{'60日均':>8}{'60日跌概率':>9}{'60日内最大跌幅(中位)':>14}{'60日内最大反向涨幅(中位)':>16}")
    for name, lst in res.items():
        r = pd.concat(lst, ignore_index=True)
        r20, r60 = r.ret20.dropna(), r.ret60.dropna()
        print(f"{name:<20}{len(r):>5}{r20.mean():>+8.1%}{(r20<0).mean():>9.0%}{r60.mean():>+8.1%}{(r60<0).mean():>9.0%}{r.mae60.median():>+14.1%}{r.mfe60.median():>+16.1%}")
    return res

if __name__ == "__main__":
    res = evaluate()
    for name in ("T2 高位第二根大阴线",):
        r = pd.concat(res[name], ignore_index=True)
        print(f"\n{name} 明细:")
        print(r[["index","date","ret20","ret60","mae60","mfe60"]].assign(date=lambda x: x.date.dt.date).to_string(index=False, float_format=lambda v: f"{v:+.1%}"))
