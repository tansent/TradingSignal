"""第一个 △B 只买半档，回测（2026-10-19）。

基准是现行规则：分档（△B 20%、★B 40%，中线温度 ≤10、国家队 ≥200 亿各 +20%）+ 估值调整（分位 ≤20% 每档 +20%、≥80% 减半）
+ ★S/▽S 每轮卖 20% + 破月BBI 清仓（research/valuation_backtest.py 的 run）。
"一段"：20 个交易日内连续出现的 B 点算同一段；"第一个 △B" = 一段里的第一个 B 点、且是 △B（不是 ★B）。
  A：第一个 △B 只买半档（不论趋势）
  B：只在下跌趋势（收盘在月线 BBI 下方）时，第一个 △B 只买半档
  C：前两个 △B 都只买半档
同一轮（5 个交易日）内再出 B 点会按规则补足到整档，所以"半档"只在之后 5 天内没有新 B 点时才真正少买。

结果（6 个有估值数据的指数平均）：A、B 全期年化都是 +3.9%（和现行规则一样），最大回撤 -25.2% 对 -25.5%，两段都几乎不变；
C 年化降到 +3.7%。改动几乎没有影响：B 点扎堆时同一轮会补足，半档很快被补回；△B 本来只有 20%。不改规则。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import valuation  # noqa: E402
from strategy_backtest import load_market, signals, stats  # noqa: E402
from valuation_backtest import IDX, run, tier  # noqa: E402


def add_rank(s: pd.DataFrame, gap: int = 20) -> pd.DataFrame:
    b = (s.b1 | s.b2).to_numpy()
    rank = np.zeros(len(s), dtype=int)
    k, last = 0, -10**9
    for i in np.flatnonzero(b):
        k = k + 1 if i - last <= gap else 1
        rank[i], last = k, i
    s = s.copy()
    s["rank"] = rank
    s["down"] = s.close < s.bbi
    return s


def base(r):
    t = tier(r)
    return t + 0.2 if r["v"] == "low" else t / 2 if r["v"] == "high" else t


VARIANTS = {
    "现行规则": base,
    "A 第一个△B 半档": lambda r: base(r) / 2 if (r["rank"] == 1 and not r["b2"]) else base(r),
    "B 下跌趋势中第一个△B 半档": lambda r: base(r) / 2 if (r["rank"] == 1 and not r["b2"] and r["down"]) else base(r),
    "C 前两个△B 半档": lambda r: base(r) / 2 if (r["rank"] <= 2 and not r["b2"]) else base(r),
}


def main():
    mkt = load_market()
    rows = []
    for name, code in IDX:
        s = signals(code, mkt, 200.0)
        v, _ = valuation.series(name)
        q = valuation.pct_asof(v).reindex(s.index, method="ffill")
        s["v"] = np.where(q <= valuation.CHEAP, "low", np.where(q >= valuation.RICH, "high", "mid"))
        s = add_rank(s)
        for lo, hi, per in (("2011-08-01", "2026-12-31", "全期"), ("2011-08-01", "2018-12-31", "2011-2018"), ("2019-01-01", "2026-12-31", "2019-2026")):
            x = s[(s.index >= lo) & (s.index <= hi)]
            for k, fn in VARIANTS.items():
                r = run(x, fn, lambda r: 0.2, lambda r: True)
                st = stats(r["equity"], r["pos"], 0)
                rows.append({"指数": name, "区间": per, "方法": k, "年化": st["年化收益"], "最大回撤": st["最大回撤"], "平均仓位": st["平均仓位"]})
    R = pd.DataFrame(rows)
    pd.set_option("display.unicode.east_asian_width", True)
    print(R.groupby(["区间", "方法"], sort=False)[["年化", "最大回撤", "平均仓位"]].mean().map(lambda x: f"{x:+.1%}").to_string())
    print()
    print(R[R["区间"] == "全期"].pivot(index="方法", columns="指数", values="年化").map(lambda x: f"{x:+.1%}").to_string())


if __name__ == "__main__":
    main()
