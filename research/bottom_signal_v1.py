"""底部信号 v1（分级）：
一级（关注区）：盘中情绪 ≤10（复刻情绪，用当日最低价）且离 120 日高点的回撤 ≥ 11 倍日波动率；
二级（较强）：一级 + 宽度恐慌（跌停≥3% 或下跌家数≥92%）或 国家队宽基 ETF 同时放量。
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest import first_trigger  # noqa: E402
from bottom_backtest import START, fwd  # noqa: E402
from bottom_features import INDEXES, all_features, load_index  # noqa: E402
from etf_flow import panel  # noqa: E402


def etf_spike(index) -> pd.Series:
    E = panel()
    E = E[E.index >= "2013-01-01"]
    s = (E.median(axis=1) >= 1.5) & ((E >= 2).sum(axis=1) / E.notna().sum(axis=1) >= 0.3)
    return s.reindex(index).fillna(False).astype(bool)


def tiers(f: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    t1 = ((f["情绪_盘中低"] <= 10) & (f["距120日高σ_盘中低"] <= -11)).fillna(False)
    panic = ((f["跌停占比"] >= 0.03) | (f["下跌占比"] >= 0.92)).fillna(False)
    t2 = t1 & (panic | etf_spike(f.index))
    return t1, t2


if __name__ == "__main__":
    F = all_features()
    rows = {}
    for name, code in INDEXES.items():
        df = load_index(code)
        f = F[name]
        t1, t2 = tiers(f)
        variants = {"一级": t1, "二级": t2, "二级 + 距高点≥30天": t2 & (f["距120日高天数"] >= 30)}
        for vn, cond in variants.items():
            s = first_trigger(cond, 20)
            s = s[s.index >= START]
            rows.setdefault(vn, []).append(fwd(df, s[s].index).assign(index=name, env=f.loc[s[s].index, "环境"].to_numpy()))
    for vn, lst in rows.items():
        r = pd.concat(lst)
        r2 = r.dropna(subset=["r60"])
        print(f"{vn:<14} {len(r):>3}次 | 20日胜{(r2.r20 > 0).mean():.0%} 均{r2.r20.mean():+.1%} | 60日胜{(r2.r60 > 0).mean():.0%} 均{r2.r60.mean():+.1%} | 60日内最大跌中位{r2.dd60.median():+.1%}")
        print("     按环境:", {k: f"{len(v)}次 60日胜{(v.r60.dropna() > 0).mean():.0%} 均{v.r60.mean():+.1%}" for k, v in r.groupby("env")})
