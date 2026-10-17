"""按估值调整分档仓位，回测是否比原来的 C 方法（分档 + 破月BBI离场）更好（2026-10-17）。

原 C 方法：△B 补到 20%、★B 40%，中线温度 ≤10 再 +20%、国家队 ≥200 亿再 +20%；★S/▽S 每次卖 20%；破月BBI 清仓。
估值用当时可知的分位（research/valuation.py），≤20% 算便宜、≥80% 算贵，中间或没有数据时按原规则。
  买：便宜时档位 +20%，贵时档位减半
  卖：贵时每个 S 卖 40%，便宜时卖 10%
  离场：便宜时不因破月BBI 清仓
  合并：以上三条一起
指数：有估值数据的 6 个（上证50、沪深300、中证500、中证1000、上证指数用中证市盈率；创业板指用蛋卷市盈率，2019 年后才有分位）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import valuation  # noqa: E402
from strategy_backtest import CASH_YIELD, COST, EPISODE, load_market, signals, stats  # noqa: E402

IDX = [("上证50", "sh000016"), ("沪深300", "sh000300"), ("中证500", "sh000905"), ("中证1000", "sh000852"),
       ("上证指数", "sh000001"), ("创业板指", "sz399006")]


def run(s: pd.DataFrame, buy, sell, exit_ok) -> pd.DataFrame:
    c = s.close.to_numpy()
    p, pos, turn = 0.0, np.zeros(len(s)), np.zeros(len(s))
    buy_end, buy_added, sell_end = -1, 0.0, -1
    for i, r in enumerate(s.to_dict("records")):
        new = p
        want = buy(r) if (r["b1"] or r["b2"]) else 0.0
        if want > 0:
            if i > buy_end:
                buy_end, buy_added = i + EPISODE, 0.0
            add = min(max(0.0, want - buy_added), 1 - new)
            if add > 0:
                new += add
                buy_added += add
        if r["s"] and i > sell_end and new > 0:
            sell_end = i + EPISODE
            new = max(0.0, new - sell(r))
        if r["bbi_two"] and new > 0 and exit_ok(r):
            new = 0.0
        turn[i], p, pos[i] = abs(new - p), new, new
    ret = np.r_[0, c[1:] / c[:-1] - 1]
    held = np.r_[0, pos[:-1]]
    eq = np.cumprod(1 + held * ret + (1 - held) * CASH_YIELD / 252 - turn * COST)
    return pd.DataFrame({"pos": pos, "equity": eq}, index=s.index)


def tier(r):
    return (0.4 if r["b2"] else 0.2) + (0.2 if r["cold"] else 0) + (0.2 if r["etf"] else 0)


VARIANTS = {
    "原 C 方法": (tier, lambda r: 0.2, lambda r: True),
    "买：便宜+20%、贵减半": (lambda r: tier(r) + 0.2 if r["v"] == "low" else tier(r) / 2 if r["v"] == "high" else tier(r),
                    lambda r: 0.2, lambda r: True),
    "卖：贵卖40%、便宜卖10%": (tier, lambda r: 0.4 if r["v"] == "high" else 0.1 if r["v"] == "low" else 0.2, lambda r: True),
    "离场：便宜时不清仓": (tier, lambda r: 0.2, lambda r: r["v"] != "low"),
}
VARIANTS["合并三条"] = (VARIANTS["买：便宜+20%、贵减半"][0], VARIANTS["卖：贵卖40%、便宜卖10%"][1], VARIANTS["离场：便宜时不清仓"][2])


def main():
    mkt = load_market()
    rows = []
    for name, code in IDX:
        s = signals(code, mkt, 200.0)
        v, _ = valuation.series(name)
        q = valuation.pct_asof(v).reindex(s.index, method="ffill")
        s["v"] = np.where(q <= valuation.CHEAP, "low", np.where(q >= valuation.RICH, "high", "mid"))
        for lo, hi, per in (("2011-08-01", "2026-12-31", "全期"), ("2011-08-01", "2018-12-31", "2011-2018"), ("2019-01-01", "2026-12-31", "2019-2026")):
            x = s[(s.index >= lo) & (s.index <= hi)]
            for k, (b, se, ex) in VARIANTS.items():
                r = run(x, b, se, ex)
                st = stats(r["equity"], r["pos"], 0)
                rows.append({"指数": name, "区间": per, "方法": k, "年化": st["年化收益"], "最大回撤": st["最大回撤"], "平均仓位": st["平均仓位"]})
        print(name, "估值便宜天数", int((s.v == "low").sum()), "贵", int((s.v == "high").sum()))
    R = pd.DataFrame(rows)
    out = R.groupby(["区间", "方法"], sort=False)[["年化", "最大回撤", "平均仓位"]].mean()
    pd.set_option("display.unicode.east_asian_width", True)
    print(out.map(lambda x: f"{x:+.1%}").to_string())
    print()
    print(R[R["区间"] == "全期"].pivot(index="方法", columns="指数", values="年化").map(lambda x: f"{x:+.1%}").to_string())


if __name__ == "__main__":
    main()
