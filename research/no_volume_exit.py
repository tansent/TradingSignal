"""B 点之后 N 天内一直没有补量，要不要卖掉或少加（2026-10-21，用户提议，接 research/weak_to_strong.py）。

补量 = 连续 2 天收盘 > 5 日线且成交量 > 5 日均量（MA(V,5)）。N = 5/10/15/20 个交易日。
一、逐笔（8 个宽基 2011-08 起，现行规则下可买的 B：★B 或不在阴跌中的 △B，间隔 ≥20 天）：
    N 天内没补量的比例；没补量时，如果第 N 天收盘卖掉，之后 20/60 天指数还会怎么走（卖得对不对）；和有补量的对比。
二、分档仓位回测（现行规则，6 个有估值数据的指数）：每一轮买入开始后第 N 天检查，
    卖本轮：期间没有补量就卖掉这一轮买的仓位；全卖：卖掉全部仓位；
    少加：期间没有补量，之后的 B 点档位减半，直到出现补量为止。

结果：
  逐笔（222 轮）：没补量的 B 点到第 N 天已经又跌了一段（N=10 中位 -1.1%，N=20 -4.5%），但从第 N 天往后反而比有补量的好：
  N=10 之后 60 天上涨 70%、中位 +4.2%（有补量的 55%、+1.4%）；N=20 是 67%、+5.0%（有补量 61%、+2.8%）。第 N 天卖等于卖在反弹前。
  分档回测：卖出版本年化都降低（N=5～15：+2.8%～+4.0% 对 +4.4%），回撤变小主要因为平均仓位降低（5 天全卖平均仓位 9%、回撤 -13.8%、年化 +2.9%）；
  N=20 卖本轮 +4.4% / -22.3%，但 2019-2026 变差（+5.3% 对 +5.6%）。少加版本 +4.1%～+4.3%，回撤几乎不变。不采用。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import valuation  # noqa: E402
from first_b_half import base  # noqa: E402
from ma5_streak import streak  # noqa: E402
from stop_loss import IDX as BROAD  # noqa: E402
from strategy_backtest import CASH_YIELD, COST, EPISODE, load_market, signals, stats  # noqa: E402
from tdx_reader import read_day  # noqa: E402
from valuation_backtest import IDX  # noqa: E402
from weak_to_strong import strong  # noqa: E402

V = "D:/TDXapp/vipdoc"
NS = (5, 10, 15, 20)


def bl(code: str, index) -> np.ndarray:
    return strong(read_day(f"{V}/{code[:2]}/lday/{code}.day"))["补量2天"].reindex(index, fill_value=False).to_numpy()


def part1(mkt):
    rows = []
    for name, code in BROAD:
        s = signals(code, mkt, 200.0)
        z = bl(code, s.index)
        C = s.close.to_numpy()
        st = streak(s.close)
        ok = (s.b2 | (s.b1 & (st < 5))).to_numpy()
        last = -999
        for i in np.flatnonzero(ok):
            if s.index[i] < pd.Timestamp("2011-08-01") or i - last < 20 or i + 80 >= len(C):
                continue
            last = i
            for n in NS:
                got = z[i + 1:i + n + 1].any()
                j = i + n
                rows.append({"N": n, "补量": got, "B到第N天": C[j] / C[i] - 1, "第N天后20": C[j + 20] / C[j] - 1,
                             "第N天后60": C[j + 60] / C[j] - 1, "B后60": C[i + 60] / C[i] - 1})
    R = pd.DataFrame(rows)
    print(f"== 一、逐笔：{len(R) // len(NS)} 轮可买的 B 点 ==")
    for n, g in R.groupby("N"):
        for got, h in g.groupby("补量"):
            print(f"  N={n:2d} {'有补量' if got else '没补量'} {len(h):3d} 轮（{len(h) / len(g):.0%}）：B 到第 N 天 中位 {h.B到第N天.median():+.1%}；"
                  f"第 N 天之后 20 天上涨 {(h.第N天后20 > 0).mean():.0%}、中位 {h.第N天后20.median():+.1%}；"
                  f"60 天上涨 {(h.第N天后60 > 0).mean():.0%}、中位 {h.第N天后60.median():+.1%}")


def buy_rule(r):
    return 0.0 if (r["st"] >= 5 and not r["b2"]) else base(r)


def run(s, z, n, mode):
    c = s.close.to_numpy()
    p, pos, turn = 0.0, np.zeros(len(s)), np.zeros(len(s))
    buy_end, buy_added, sell_end = -1, 0.0, -1
    pending = []                      # [开始日, 本轮买入量]
    weak = False                      # 少加模式：有没确认的轮次时，B 点档位减半
    for i, r in enumerate(s.to_dict("records")):
        new = p
        if z[i]:
            pending, weak = [], False
        want = buy_rule(r) if (r["b1"] or r["b2"]) else 0.0
        if mode == "less" and weak:
            want /= 2
        if want > 0:
            if i > buy_end:
                buy_end, buy_added = i + EPISODE, 0.0
                pending.append([i, 0.0])
            add = min(max(0.0, want - buy_added), 1 - new)
            if add > 0:
                new += add
                buy_added += add
                if pending:
                    pending[-1][1] += add
        if r["s"] and i > sell_end and new > 0:
            sell_end = i + EPISODE
            new = max(0.0, new - 0.2)
        keep = []
        for i0, amt in pending:
            if i - i0 >= n and mode != "none":
                if mode == "round":
                    new = max(0.0, new - amt)
                elif mode == "all":
                    new = 0.0
                elif mode == "less":
                    weak = True
            else:
                keep.append([i0, amt])
        pending = keep if mode != "none" else [x for x in pending if i - x[0] < 60]
        if r["bbi_two"] and new > 0:
            new = 0.0
        turn[i], p, pos[i] = abs(new - p), new, new
    ret = np.r_[0, c[1:] / c[:-1] - 1]
    held = np.r_[0, pos[:-1]]
    eq = np.cumprod(1 + held * ret + (1 - held) * CASH_YIELD / 252 - turn * COST)
    return pd.Series(eq, index=s.index), pd.Series(pos, index=s.index)


def part2(mkt):
    rows = []
    for name, code in IDX:
        s = signals(code, mkt, 200.0)
        v, _ = valuation.series(name)
        q = valuation.pct_asof(v).reindex(s.index, method="ffill")
        s["v"] = np.where(q <= valuation.CHEAP, "low", np.where(q >= valuation.RICH, "high", "mid"))
        s["st"] = streak(s.close)
        z = bl(code, s.index)
        for lo, hi, per in (("2011-08-01", "2026-12-31", "全期"), ("2011-08-01", "2018-12-31", "2011-2018"), ("2019-01-01", "2026-12-31", "2019-2026")):
            m = (s.index >= lo) & (s.index <= hi)
            x, zz = s[m], z[m]
            variants = [("现行规则", 0, "none")] + [(f"{n}天没补量·{lab}", n, md) for n in NS
                                                  for md, lab in (("round", "卖本轮"), ("all", "全卖"), ("less", "之后少加"))]
            for lab, n, md in variants:
                eq, pos = run(x, zz, n, md)
                a = stats(eq, pos, 0)
                rows.append({"区间": per, "方法": lab, "年化": a["年化收益"], "最大回撤": a["最大回撤"], "平均仓位": a["平均仓位"]})
    R = pd.DataFrame(rows)
    print("\n== 二、分档仓位回测（6 个指数平均）==")
    t = R.groupby(["方法", "区间"], sort=False)[["年化", "最大回撤"]].mean().unstack("区间").map(lambda v: f"{v:+.1%}")
    t["平均仓位"] = R[R.区间 == "全期"].groupby("方法", sort=False).平均仓位.mean().map("{:.0%}".format)
    print(t.to_string())


def main():
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 220)
    mkt = load_market()
    part1(mkt)
    part2(mkt)


if __name__ == "__main__":
    main()
