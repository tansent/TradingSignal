"""底部信号：几种思路对比。"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "research")
from lib import NAMES, features, forward_stats, load
from backtest import INDEXES, START, first_trigger

def prior_low_info(df, f, look=60, gap=5):
    """当前之前 [t-look, t-gap] 区间内的最低价，以及那根K线当时的 RSI14 / z20。"""
    l = df["low"].to_numpy(); r = f["rsi14"].to_numpy(); z = f["z20"].to_numpy()
    n = len(l); pl = np.full(n, np.nan); pr = np.full(n, np.nan); pz = np.full(n, np.nan); pd_ = np.full(n, np.nan)
    for i in range(look, n):
        w = l[i - look:i - gap + 1]
        j = i - look + int(np.argmin(w))
        pl[i], pr[i], pz[i], pd_[i] = l[j], r[j], z[j], i - j
    return pd.Series(pl, df.index), pd.Series(pr, df.index), pd.Series(pz, df.index), pd.Series(pd_, df.index)

def oversold(f, z5=-1.5, z20=-1.3, zb60=-0.85, zdd=-12, rsi=30):
    return (f.z5 <= z5) & (f.z20 <= z20) & (f.zbias60 <= zb60) & (f.z_dd120 <= zdd) & (f.rsi14 <= rsi)

def build(df, f, mode):
    c, h, l, o = df.close, df.high, df.low, df.open
    pl, pr, pz, pdist = prior_low_info(df, f)
    p = l if mode == "intraday" else c
    os_ = oversold(f)
    os_loose = (f.z_dd120 <= -10) & (f.rsi14 <= 35) & (f.zbias60 <= -0.6)
    newlow = p < pl
    v = {}
    v["A 超卖(基线)"] = os_
    v["B 超卖+下跌已持续>=40天"] = os_ & (f.days_from_high >= 40)
    v["C 创新低+RSI底背离"] = os_loose & newlow & (f.rsi14 > pr + 5) & (pr <= 35)
    v["D 底背离+持续>=40天"] = os_loose & newlow & (f.rsi14 > pr + 5) & (pr <= 35) & (f.days_from_high >= 40)
    if mode == "close":
        rng = (h - l).replace(0, np.nan)
        hammer = ((c - l) / rng >= 0.6) & ((np.minimum(o, c) - l) / rng >= 0.4)
        v["E 超卖+长下影收回"] = oversold(f, z5=-1.0, z20=-1.0, zb60=-0.7, zdd=-10, rsi=40) & hammer & (l <= l.rolling(20).min())
    return v

def evaluate(mode):
    res = {}
    for code in INDEXES:
        df = load(code); f = features(df, mode)
        for name, cond in build(df, f, mode).items():
            s = first_trigger(cond); s = s[s.index >= START]
            st = forward_stats(df, s[s].index, horizons=(20, 60))
            st["index"] = NAMES[code]; st["regime"] = f.loc[s[s].index, "regime"].to_numpy()
            res.setdefault(name, []).append(st)
    print(f"\n######## {mode} ########")
    print(f"{'规则':<22}{'次数':>5}{'20日均':>8}{'20日胜':>7}{'60日均':>8}{'60日胜':>7}{'60日内最大涨幅(中位)':>14}{'60日内最大浮亏(中位)':>14}")
    for name, lst in res.items():
        r = pd.concat(lst, ignore_index=True)
        r20, r60 = r.ret20.dropna(), r.ret60.dropna()
        print(f"{name:<22}{len(r):>5}{r20.mean():>+8.1%}{(r20>0).mean():>7.0%}{r60.mean():>+8.1%}{(r60>0).mean():>7.0%}{r.mfe60.median():>+14.1%}{r.mae60.median():>+14.1%}")
    return res

if __name__ == "__main__":
    for m in ("close", "intraday"):
        res = evaluate(m)
        if m == "intraday":
            for name in ("C 创新低+RSI底背离", "D 底背离+持续>=40天"):
                r = pd.concat(res[name], ignore_index=True)
                print(f"\n{name} 明细:")
                print(r[["index","date","regime","ret20","ret60","mfe60","mae60"]].assign(date=lambda x: x.date.dt.date).to_string(index=False, float_format=lambda v: f"{v:+.1%}"))
