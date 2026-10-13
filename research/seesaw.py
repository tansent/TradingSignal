"""检验"红利低波和科技板块是跷跷板"：走势是否反向，买卖信号是否同时出现。

红利低波：中证红利低波动指数（H30269，中证官网日线）；科技：创业板指（399006）、科创50（000688，2020 年起）。
信号与 DINGDI 相同（research/dingdi.py）。"同时出现"＝另一边在前后 W 个交易日内也出现了对应信号，
并与"随便挑一天，前后 W 天内碰到该信号"的基准概率比较。
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dingdi  # noqa: E402
import sector_backtest as sbt  # noqa: E402
from tdx_reader import read_day  # noqa: E402

os.environ["NO_PROXY"] = "*"


def load():
    d = {"红利低波": sbt.daily("csi:H30269"),
         "创业板指": read_day("D:/TDXapp/vipdoc/sz/lday/sz399006.day"),
         "科创50": read_day("D:/TDXapp/vipdoc/sh/lday/sh000688.day")}
    sig = {}
    for k, df in d.items():
        r = dingdi.compute(df)
        sig[k] = pd.DataFrame({"close": df.close, "B": r["底部一级"] | r["底部二级"], "S": r["顶部"] | r["顶部一级"]})
    return sig


def near(flag: pd.Series, idx: pd.DatetimeIndex, w: int) -> pd.Series:
    """idx 中每一天，flag 在前后 w 个交易日内是否出现过。"""
    f = flag.reindex(idx).fillna(False).astype(float)
    return (f.rolling(2 * w + 1, center=True, min_periods=1).max() > 0)


def cooccur(sig, a: str, a_col: str, b: str, b_col: str, w: int, start: str):
    idx = sig[a].index.intersection(sig[b].index)
    idx = idx[idx >= start]
    hit = near(sig[b][b_col], idx, w)
    days = sig[a][a_col].reindex(idx).fillna(False).astype(bool)
    return days.sum(), hit[days].mean(), hit.mean()


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    sig = load()
    px = pd.DataFrame({k: v.close for k, v in sig.items()})
    print("== 1. 走势是否反向 ==")
    for tech, start in (("创业板指", "2011-01-01"), ("科创50", "2021-01-01")):
        p = px[["红利低波", tech]].dropna()
        p = p[p.index >= start]
        r1 = p.pct_change().dropna()
        r20 = (p / p.shift(20) - 1).dropna().iloc[::20]
        r60 = (p / p.shift(60) - 1).dropna().iloc[::60]
        roll = r1["红利低波"].rolling(250).corr(r1[tech]).dropna()
        print(f"红利低波 vs {tech}（{start[:4]} 起）：日涨跌相关 {r1.corr().iloc[0, 1]:+.2f}；20 日涨跌相关 {r20.corr().iloc[0, 1]:+.2f}；"
              f"60 日涨跌相关 {r60.corr().iloc[0, 1]:+.2f}；滚动一年日相关 最低 {roll.min():+.2f} 最高 {roll.max():+.2f} 现在 {roll.iloc[-1]:+.2f}")
        both = r60.apply(lambda x: "同涨" if (x > 0).all() else "同跌" if (x < 0).all() else "一涨一跌", axis=1).value_counts(normalize=True)
        print("   每 60 天一段：", ", ".join(f"{k} {v:.0%}" for k, v in both.items()))
        rel = (p["红利低波"] / p[tech])
        y = rel.resample("YE").last().pct_change().dropna() * 100
        print("   红利低波相对", tech, "的年度超额（%）：", " ".join(f"{d.year}:{v:+.0f}" for d, v in y.items()))
    print("\n== 2. 买卖点是否同时出现（前后 W 个交易日内）==")
    for w in (5, 10, 20):
        print(f"-- W = {w}")
        for tech, start in (("创业板指", "2011-08-01"), ("科创50", "2021-01-01")):
            for (a, ac, b, bc, lab) in (("红利低波", "B", tech, "S", f"红利低波出 △B 时，{tech}附近出卖点"),
                                        ("红利低波", "S", tech, "B", f"红利低波出卖点时，{tech}附近出 △B"),
                                        (tech, "S", "红利低波", "B", f"{tech}出卖点时，红利低波附近出 △B"),
                                        ("红利低波", "B", tech, "B", f"红利低波出 △B 时，{tech}附近也出 △B（同跌）")):
                n, p_hit, base = cooccur(sig, a, ac, b, bc, w, start)
                print(f"   {lab:<34} 次数 {n:>3}  命中 {p_hit:5.0%}  随机基准 {base:5.0%}  倍数 {p_hit / base if base else np.nan:4.1f}")
