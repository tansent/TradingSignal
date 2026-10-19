"""持续阴跌（连续多天收在 5 日线下方）时，△B 要不要买（2026-10-19，用户提议）。

1. 信号：8 个宽基 2011-08 起，按 B 点出现时已经连续几天收在 5 日线下方分组，看之后 60 天。
   门槛 5 天：△B 连续 <5 天 355 次，60 天上涨 62%、中位 +3.5%；连续 ≥5 天 928 次，54%、+1.5%（平时 52%）。
   ★B 连续 ≥5 天 71 次，73%、+8.2%。
   "等第一次收盘站上 5 日线再买"没用：平均多付 1～2%，之后收益不比 B 点当天好。
2. 分档仓位回测（research/valuation_backtest.run，6 个有估值数据的指数，基准是现行规则）：
   连续 ≥N 天时 △B 不买、★B 照买，N = 4/5/7/10 都比现行规则好（年化 +4.6/+4.4/+4.2/+4.1% 对 +3.9%，
   最大回撤 -22.2/-24.1/-25.3/-25.4% 对 -25.5%），2011-2018、2019-2026 两段都成立。采用 N = 5（一周，不挑回测最好的 4）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import valuation  # noqa: E402
from first_b_half import base  # noqa: E402
from strategy_backtest import load_market, signals, stats  # noqa: E402
from valuation_backtest import IDX, run  # noqa: E402


def streak(close: pd.Series) -> np.ndarray:
    c, ma5 = close.to_numpy(), close.rolling(5).mean().to_numpy()
    st = np.zeros(len(c), dtype=int)
    for i in range(1, len(c)):
        st[i] = st[i - 1] + 1 if c[i] < ma5[i] else 0
    return st


def main():
    mkt = load_market()
    variants = {"现行规则": base}
    for n in (4, 5, 7, 10):
        variants[f"连续≥{n}天时 △B 不买"] = (lambda n: (lambda r: 0.0 if (r["st"] >= n and not r["b2"]) else base(r)))(n)
    rows = []
    for name, code in IDX:
        s = signals(code, mkt, 200.0)
        v, _ = valuation.series(name)
        q = valuation.pct_asof(v).reindex(s.index, method="ffill")
        s["v"] = np.where(q <= valuation.CHEAP, "low", np.where(q >= valuation.RICH, "high", "mid"))
        s["st"] = streak(s.close)
        for lo, hi, per in (("2011-08-01", "2026-12-31", "全期"), ("2011-08-01", "2018-12-31", "2011-2018"), ("2019-01-01", "2026-12-31", "2019-2026")):
            x = s[(s.index >= lo) & (s.index <= hi)]
            for k, fn in variants.items():
                r = run(x, fn, lambda r: 0.2, lambda r: True)
                st = stats(r["equity"], r["pos"], 0)
                rows.append({"区间": per, "方法": k, "年化": st["年化收益"], "最大回撤": st["最大回撤"]})
    R = pd.DataFrame(rows)
    pd.set_option("display.unicode.east_asian_width", True)
    print(R.groupby(["方法", "区间"], sort=False)[["年化", "最大回撤"]].mean().unstack("区间").map(lambda x: f"{x:+.1%}").to_string())


if __name__ == "__main__":
    main()
