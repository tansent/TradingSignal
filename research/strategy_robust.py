"""strategy_backtest 的稳健性检查：更多指数、前后两段、仓位比例网格。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from strategy_backtest import load_market, run, signals, stats, strategies  # noqa: E402

IDX = [("上证50", "sh000016"), ("沪深300", "sh000300"), ("中证500", "sh000905"), ("中证1000", "sh000852"),
       ("上证指数", "sh000001"), ("深证成指", "sz399001"), ("创业板指", "sz399006")]
PERIODS = [("2011-08-01", "2018-12-31"), ("2019-01-01", "2026-12-31"), ("2011-08-01", "2026-12-31")]

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    mkt = load_market()
    S = {name: signals(code, mkt, 200.0) for name, code in IDX}
    strat = strategies(etf_ok=True)
    rows = []
    for a, b in PERIODS:
        for name, s0 in S.items():
            s = s0[(s0.index >= a) & (s0.index <= b)]
            if len(s) < 500:
                continue
            bh = s.close / s.close.iloc[0]
            base = stats(bh, pd.Series(1.0, index=s.index), 1)
            rows.append({"区间": f"{a[:4]}-{b[:4]}", "指数": name, "方法": "买入持有", **base})
            for k, (bs, ss, ex) in strat.items():
                r = run(s, bs, ss, ex)
                rows.append({"区间": f"{a[:4]}-{b[:4]}", "指数": name, "方法": k[:2].strip(), **stats(r["equity"], r["pos"], int(r.trades.iloc[0]))})
    df = pd.DataFrame(rows)
    for col in ("年化收益", "最大回撤", "平均仓位"):
        print(f"\n== {col}（行=指数，列=方法）==")
        for per, g in df.groupby("区间", sort=False):
            t = g.pivot(index="指数", columns="方法", values=col).reindex([n for n, _ in IDX])
            t.loc["平均"] = t.mean()
            print(f"-- {per}")
            print((t * 100).round(1).to_string())
    # 仓位比例网格（全区间，7 个指数平均）
    print("\n== 不设BBI、只用△B/★B 和 ★S/▽S：买入比例 × 卖出比例 的 7 指数平均年化 / 平均最大回撤 ==")
    grid = []
    for bsz in (0.25, 1 / 3, 0.5, 1.0):
        for ssz in (0.2, 1 / 3, 0.5, 1.0):
            res = []
            for name, s0 in S.items():
                s = s0[s0.index >= "2011-08-01"]
                r = run(s, lambda row, v=bsz: v, ssz, None)
                res.append(stats(r["equity"], r["pos"], 0))
            grid.append({"买": round(bsz, 2), "卖": round(ssz, 2), "年化": sum(x["年化收益"] for x in res) / len(res),
                         "回撤": sum(x["最大回撤"] for x in res) / len(res), "仓位": sum(x["平均仓位"] for x in res) / len(res)})
    g = pd.DataFrame(grid)
    print(g.pivot(index="买", columns="卖", values="年化").mul(100).round(1).to_string())
    print(g.pivot(index="买", columns="卖", values="回撤").mul(100).round(1).to_string())
    print(g.pivot(index="买", columns="卖", values="仓位").mul(100).round(0).to_string())
