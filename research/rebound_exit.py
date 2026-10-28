"""在月线 BBI 下方买的 B 点（博反弹）怎么卖（2026-10-20，用户提问）。

现行规则的漏洞：破月BBI 要"先在 BBI 上方、再连续两天收在下方"才标；在 BBI 下方买入后如果一直没站上 BBI，
就永远等不到破月BBI，只能靠 ★S/▽S 每轮卖 20 个点。

一、逐笔：8 个宽基 2011-08 起，每轮第一个可买的 B（★B，或不在阴跌中的 △B），间隔 ≥20 个交易日，收盘买入、收盘卖出，最长 250 天。
二、分档仓位回测（现行规则：估值调整 + 阴跌 △B 不买；6 个有估值数据的指数）：一轮买入开始时收盘在月 BBI 下方，
   这批仓位记为"反弹仓"，直到收盘站上月 BBI 为止，期间按不同规则处理。

结果：
  1. 月 BBI 下方买的 B 是常态（219 笔里 187 笔），而且是好的那一类：现行规则平均 +9.6%、中位 +6.4%、赚钱 83%；
     月 BBI 上方买的 32 笔（刚从高位跌下来）平均 -2.8%、赚钱只有 19%。
     下方买入后 250 天内 96% 站上了月 BBI（中位 56 个交易日）；没站上的 8 笔平均 -17.8%。
  2. "站上月 BBI 就卖"把逐笔平均从 +9.6% 压到 +5.7%，分档回测年化 +4.4% → +3.6%（卖一半 +4.0%）：反弹常常变成反转，不采用。
     "反弹仓第一个 S 就全卖"和现行一样（S 点很少出现在站上 BBI 之前）。
  3. "N 天没站上月 BBI 就清仓"：年化没有提高（N=20～120 在 +3.5%～+4.6% 之间来回）；N=30 时最大回撤 -24.1% → -16.2%、
     平均仓位 23% → 13%，但主要来自 2018 年阴跌（中证1000、中证500、上证），N=35 回撤约 -18%、N=40 -21.7%、N≥60 约 -23～-24%。
     对 N 太敏感，不采用；现行规则不变。
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
from valuation_backtest import IDX  # noqa: E402

HOLD = 250


def buy_rule(r):
    return 0.0 if (r["st"] >= 5 and not r["b2"]) else base(r)


def part1(mkt):
    rows = []
    for name, code in BROAD:
        s = signals(code, mkt, 200.0)
        c, bbi, brk, top = s.close.to_numpy(), s.bbi.to_numpy(), s.bbi_two.to_numpy(), s.s.to_numpy()
        st = streak(s.close)
        ok = (s.b2 | (s.b1 & (st < 5))).to_numpy()
        last = -999
        for i in np.flatnonzero(ok):
            if s.index[i] < pd.Timestamp("2011-08-01") or i - last < 20 or i + HOLD >= len(c):
                continue
            last = i
            seg = slice(i + 1, i + HOLD + 1)
            first = lambda m: i + 1 + int(np.flatnonzero(m)[0]) if m.any() else i + HOLD  # noqa: E731
            ret = lambda j: c[j] / c[i] - 1  # noqa: E731
            below = c[i] < bbi[i]
            touch = first(c[seg] >= bbi[seg])
            j_brk = first(brk[seg])
            r = {"位置": "月BBI下方" if below else "月BBI上方", "级别": "★B" if s.b2.iloc[i] else "△B",
                 "现行（破月BBI，最长250天）": ret(j_brk), "持有20天": ret(i + 20), "持有60天": ret(i + 60),
                 "站上月BBI就卖": ret(touch), "第一个★S/▽S就卖": ret(min(first(top[seg]), j_brk)),
                 "站上卖一半，剩下等破月BBI": (ret(touch) + ret(j_brk)) / 2 if touch < j_brk else ret(j_brk),
                 "60天没站上就卖，站上了等破月BBI": ret(j_brk if touch <= i + 60 else min(i + 60, j_brk)),
                 "站上月BBI天数": touch - i if touch < i + HOLD else np.nan, "破月BBI天数": j_brk - i}
            rows.append(r)
    R = pd.DataFrame(rows)
    methods = [k for k in R.columns if k not in ("位置", "级别", "站上月BBI天数", "破月BBI天数")]
    for pos, g in R.groupby("位置", sort=False):
        print(f"\n== 一、逐笔：{pos}买入 {len(g)} 笔（★B {(g.级别 == '★B').sum()} 笔）==")
        if pos == "月BBI下方":
            t = g.站上月BBI天数
            print(f"  250 天内站上月 BBI 的 {t.notna().mean():.0%}，站上的中位 {t.median():.0f} 天；"
                  f"没站上的 {t.isna().sum()} 笔现行规则平均 {g[t.isna()]['现行（破月BBI，最长250天）'].mean():+.1%}")
        out = pd.DataFrame({k: {"平均": f"{g[k].mean():+.1%}", "中位": f"{g[k].median():+.1%}", "赚钱": f"{(g[k] > 0).mean():.0%}",
                                "最差": f"{g[k].min():+.1%}"} for k in methods}).T
        print(out.to_string())


def run(s, mode):
    c, bbi = s.close.to_numpy(), s.bbi.to_numpy()
    p, pos, turn = 0.0, np.zeros(len(s)), np.zeros(len(s))
    buy_end, buy_added, sell_end = -1, 0.0, -1
    reb, reb_start = False, -1
    for i, r in enumerate(s.to_dict("records")):
        new = p
        want = buy_rule(r) if (r["b1"] or r["b2"]) else 0.0
        if want > 0:
            if i > buy_end:
                buy_end, buy_added = i + EPISODE, 0.0
                reb, reb_start = c[i] < bbi[i], i
            add = min(max(0.0, want - buy_added), 1 - new)
            if add > 0:
                new += add
                buy_added += add
        if r["s"] and i > sell_end and new > 0:
            sell_end = i + EPISODE
            new = max(0.0, new - (1.0 if (reb and mode == "first_s") else 0.2))
        if reb and new > 0:
            if c[i] >= bbi[i]:
                if mode == "touch_all":
                    new = 0.0
                elif mode in ("touch_half", "half_timeout"):
                    new *= 0.5
                reb = False
            elif (mode.startswith("timeout") and i - reb_start >= int(mode[7:])) or (mode == "half_timeout" and i - reb_start >= 120):
                new, reb = 0.0, False
        if r["bbi_two"] and new > 0:
            new = 0.0
        if new == 0:
            reb = False
        turn[i], p, pos[i] = abs(new - p), new, new
    ret = np.r_[0, c[1:] / c[:-1] - 1]
    held = np.r_[0, pos[:-1]]
    eq = np.cumprod(1 + held * ret + (1 - held) * CASH_YIELD / 252 - turn * COST)
    return pd.Series(eq, index=s.index), pd.Series(pos, index=s.index)


MODES = {"现行规则": "none", "反弹仓站上月BBI全卖": "touch_all", "反弹仓站上月BBI卖一半": "touch_half",
         "反弹仓60天没站上就清仓": "timeout60", "反弹仓120天没站上就清仓": "timeout120",
         "站上卖一半 + 120天没站上清仓": "half_timeout", "反弹仓第一个S就全卖": "first_s"}


def part2(mkt):
    rows = []
    for name, code in IDX:
        s = signals(code, mkt, 200.0)
        v, _ = valuation.series(name)
        q = valuation.pct_asof(v).reindex(s.index, method="ffill")
        s["v"] = np.where(q <= valuation.CHEAP, "low", np.where(q >= valuation.RICH, "high", "mid"))
        s["st"] = streak(s.close)
        for lo, hi, per in (("2011-08-01", "2026-12-31", "全期"), ("2011-08-01", "2018-12-31", "2011-2018"), ("2019-01-01", "2026-12-31", "2019-2026")):
            x = s[(s.index >= lo) & (s.index <= hi)]
            for k, m in MODES.items():
                eq, pos = run(x, m)
                st = stats(eq, pos, 0)
                rows.append({"区间": per, "方法": k, "年化": st["年化收益"], "最大回撤": st["最大回撤"], "平均仓位": st["平均仓位"]})
    R = pd.DataFrame(rows)
    print("\n== 二、分档仓位回测（6 个指数平均）==")
    t = R.groupby(["方法", "区间"], sort=False)[["年化", "最大回撤"]].mean().unstack("区间")
    print(t.map(lambda x: f"{x:+.1%}").to_string())
    print(R[R.区间 == "全期"].groupby("方法", sort=False).平均仓位.mean().map("{:.0%}".format).to_string())


def main():
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 220)
    mkt = load_market()
    part1(mkt)
    part2(mkt)


if __name__ == "__main__":
    main()
