"""第二轮 ★B 和上一轮价格差不多时，还要不要按档位再买（2026-10-21，用户提问）。

一、逐笔：★B（新一轮，距上一个 ★B 超过 5 个交易日）且 60 天内出现过上一轮 ★B，按"比上一轮 ★B 收盘"分组：
    更低 ≥5%、差不多（±5% 以内）、更高 ≥5%；看之后 20/60 天。8 个宽基 2011-08 起、125 个行业 2012 起。
二、分档仓位回测（现行规则，6 个有估值数据的指数）：手里已有仓位时，新一轮买入要求收盘比上一轮买入价低 X%（X = 0/3/5/8），
    否则这一轮不买；空仓时不限制。

结果：
  逐笔：行业 660 次第二轮 ★B，价格和上一轮差不多（±5%）的 60 天上涨 51%、中位 +0.4%（没有优势），低 ≥5% 的 70%、+9.1%，
  高 ≥5% 的 24%、-8.8%；宽基只有 24 次，差不多的 64%、+6.1%，低 ≥5% 的 75%、+12.5%。
  分档回测，宽基（6 个指数）：要求更低 0/3/5/8% 时年化 +4.2/+4.3/+4.0/+3.9%（现行 +4.4%），回撤 -23.6/-22.3/-20.9/-18.1%（现行 -24.1%），
  收益略降、回撤降低，主要靠仓位变小。
  行业（126 个，无估值调整）：现行 +2.6% / -33.5%；更低 0/3/5/8%：+2.7/+2.8/+3.0/+3.1%，回撤 -32.7/-31.1/-29.4/-26.3%，
  2012-2018、2019-2026 两段的收益和回撤都随门槛单调变好。板块上值得用。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dingdi  # noqa: E402
import valuation  # noqa: E402
from first_b_half import base  # noqa: E402
from ma5_streak import streak  # noqa: E402
from point_line_surface import cfg  # noqa: E402
from stop_loss import IDX as BROAD  # noqa: E402
from strategy_backtest import CASH_YIELD, COST, EPISODE, load_market, signals, stats  # noqa: E402
from tdx_reader import read_day  # noqa: E402
from valuation_backtest import IDX  # noqa: E402


def pairs(d: pd.DataFrame, start: str) -> list[dict]:
    r = dingdi.compute(d)
    C = d.close.to_numpy()
    b2 = np.flatnonzero(r["底部二级"].to_numpy())
    out, prev = [], None
    for i in b2:
        if prev is not None and i - prev <= 5:
            continue                                  # 同一轮
        if prev is not None and i - prev <= 60 and d.index[i] >= pd.Timestamp(start) and i + 60 < len(C):
            x = C[i] / C[prev] - 1
            out.append({"组": "比上一轮低 ≥5%" if x <= -0.05 else "比上一轮高 ≥5%" if x >= 0.05 else "差不多（±5%）",
                        "f20": C[i + 20] / C[i] - 1, "f60": C[i + 60] / C[i] - 1})
        prev = i
    return out


def part1():
    for label, items, start in (("8 个宽基 2011-08 起", [f"D:/TDXapp/vipdoc/{c[:2]}/lday/{c}.day" for _, c in BROAD], "2011-08-01"),
                                ("125 个行业 2012 起", [f"D:/TDXapp/vipdoc/sh/lday/sh{f[1]}.day" for f in cfg() if len(f[5]) == 5], "2012-01-01")):
        rows = []
        for p in items:
            try:
                d = read_day(p)
            except Exception:  # noqa: BLE001
                continue
            if len(d) > 800:
                rows += pairs(d, start)
        R = pd.DataFrame(rows)
        print(f"== 一、{label}：60 天内的第二轮 ★B {len(R)} 次 ==")
        print(R.groupby("组").agg(次数=("f60", "size"), 二十天涨=("f20", lambda x: f"{(x > 0).mean():.0%}"),
                                 六十天涨=("f60", lambda x: f"{(x > 0).mean():.0%}"), 六十天中位=("f60", lambda x: f"{x.median():+.1%}")).to_string())


def buy_rule(r):
    return 0.0 if (r["st"] >= 5 and not r["b2"]) else base(r)


def run(s, x):
    c = s.close.to_numpy()
    p, pos, turn = 0.0, np.zeros(len(s)), np.zeros(len(s))
    buy_end, buy_added, sell_end, last_px = -1, 0.0, -1, None
    for i, r in enumerate(s.to_dict("records")):
        new = p
        want = buy_rule(r) if (r["b1"] or r["b2"]) else 0.0
        if want > 0:
            if i > buy_end:
                if x is not None and new > 0 and last_px is not None and c[i] > last_px * (1 - x):
                    want = 0.0                        # 已有仓位、价格没比上一轮低 X%：这一轮不买
                else:
                    buy_end, buy_added, last_px = i + EPISODE, 0.0, c[i]
            if want > 0:
                add = min(max(0.0, want - buy_added), 1 - new)
                if add > 0:
                    new += add
                    buy_added += add
        if r["s"] and i > sell_end and new > 0:
            sell_end = i + EPISODE
            new = max(0.0, new - 0.2)
        if r["bbi_two"] and new > 0:
            new = 0.0
        if new == 0:
            last_px = None
        turn[i], p, pos[i] = abs(new - p), new, new
    ret = np.r_[0, c[1:] / c[:-1] - 1]
    held = np.r_[0, pos[:-1]]
    eq = np.cumprod(1 + held * ret + (1 - held) * CASH_YIELD / 252 - turn * COST)
    return pd.Series(eq, index=s.index), pd.Series(pos, index=s.index)


def part2():
    mkt = load_market()
    rows = []
    for name, code in IDX:
        s = signals(code, mkt, 200.0)
        v, _ = valuation.series(name)
        q = valuation.pct_asof(v).reindex(s.index, method="ffill")
        s["v"] = np.where(q <= valuation.CHEAP, "low", np.where(q >= valuation.RICH, "high", "mid"))
        s["st"] = streak(s.close)
        for lo, hi, per in (("2011-08-01", "2026-12-31", "全期"), ("2011-08-01", "2018-12-31", "2011-2018"), ("2019-01-01", "2026-12-31", "2019-2026")):
            m = (s.index >= lo) & (s.index <= hi)
            for lab, x in (("现行规则", None), ("已有仓位时新一轮要更低 0%", 0.0), ("更低 3%", 0.03), ("更低 5%", 0.05), ("更低 8%", 0.08)):
                eq, pos = run(s[m], x)
                a = stats(eq, pos, 0)
                rows.append({"区间": per, "方法": lab, "年化": a["年化收益"], "最大回撤": a["最大回撤"], "平均仓位": a["平均仓位"]})
    R = pd.DataFrame(rows)
    print("\n== 二、分档仓位回测（6 个指数平均）==")
    t = R.groupby(["方法", "区间"], sort=False)[["年化", "最大回撤"]].mean().unstack("区间").map(lambda v: f"{v:+.1%}")
    t["平均仓位"] = R[R.区间 == "全期"].groupby("方法", sort=False).平均仓位.mean().map("{:.0%}".format)
    print(t.to_string())


if __name__ == "__main__":
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 220)
    part1()
    part2()
