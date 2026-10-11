"""过热分数 = 成交额/年均、融资买入占比、融资余额60日增速 三项的 3 年分位的平均（0~1）。
评估：不同分数区间、"极热"、"过热后降温"三种状态下，指数未来 20/60/120 日表现。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest import first_trigger  # noqa: E402
from tdx_reader import read_day  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IDX = {"上证指数": "sh000001", "沪深300": "sh000300", "中证500": "sh000905", "创业板指": "sz399006"}


def heat() -> pd.DataFrame:
    f = pd.read_parquet(ROOT / "data" / "margin_turnover.parquet")
    cols = ["成交额/年均_3年分位", "融资买入占比_3年分位", "融资余额60日增速_3年分位"]
    h = pd.DataFrame({"H": f[cols].mean(axis=1, skipna=False)})
    h["H60max"] = h.H.rolling(60).max()
    h["极热"] = h.H >= 0.90
    h["过热后降温"] = (h.H60max >= 0.90) & (h.H <= 0.70)
    return h.dropna(subset=["H"])


def fwd(c: pd.Series, h: pd.Series) -> pd.DataFrame:
    l = c  # 用收盘近似
    out = pd.DataFrame(index=c.index)
    for n in (20, 60, 120):
        out[f"r{n}"] = c.shift(-n) / c - 1
    out["dd120"] = l[::-1].rolling(120, min_periods=1).min()[::-1].shift(-1) / c - 1
    out["near_high"] = c >= 0.95 * c.rolling(250).max()
    return out


if __name__ == "__main__":
    h = heat()
    rows = []
    for name, code in IDX.items():
        c = read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day").close
        x = fwd(c, h.H).join(h, how="inner")
        x["index"] = name
        rows.append(x)
    A = pd.concat(rows)
    A = A[A.index >= "2014-01-01"]
    A["bin"] = pd.cut(A.H, [0, .2, .4, .6, .8, .9, 1.01], right=False)
    print("按过热分数分组（4 个指数合计，2014 起）")
    g = A.groupby("bin", observed=True).agg(n=("r60", "size"), r20=("r20", "mean"), r60=("r60", "mean"),
                                            p60=("r60", lambda s: (s.dropna() > 0).mean()), r120=("r120", "mean"),
                                            dd=("dd120", "median"))
    print(g.round(3).to_string())
    print()
    for state in ("极热", "过热后降温"):
        for cond_name, cond in ((state, A[state]), (state + "且指数在高位(距一年高点<5%)", A[state] & A.near_high)):
            s = A[cond]
            print(f"{cond_name:<26} 天数{len(s):>5} | 20日{s.r20.mean():+.1%} 60日{s.r60.mean():+.1%}(涨概率{(s.r60.dropna() > 0).mean():.0%}) 120日{s.r120.mean():+.1%} | 120日内最大跌中位{s.dd120.median():+.1%}")
    ev = first_trigger(h["过热后降温"], 60)
    print("\n'过热后降温' 首次出现日期：", ", ".join(str(d.date()) for d in ev[ev].index))
    ev2 = first_trigger(h["极热"], 60)
    print("'极热' 首次出现日期：", ", ".join(str(d.date()) for d in ev2[ev2].index))
