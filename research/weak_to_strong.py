"""B 点之后"弱转强"（站上 5 日线 + 放量）再买，是不是更安全（2026-10-21，用户转述的量价经验）。

说法：光有 B 点不够，下跌趋势没有底；要等价格站上 5 日线且量能大于均量（"弱转强"），或几根守住 5 日线、量大于均量的小 K 线
（"补量"），或放量光头大阳线，才入场。

"强"的几种定义（都要求在 B 点之后 N 天内出现，取第一次）：
  站上5日线+量>5日均量：收盘 > MA5，成交量 > 前 5 日均量（MA(V,5)，含当天，和通达信 VOL 副图的均量线一致）；
  站上5日线+量>10日均量：同上，均量用 MA(V,10)；
  补量2天：连续 2 天收盘 > MA5 且成交量 > MA(V,5)；
  放量光头阳线：涨幅 ≥1.5%，上影线 ≤0.2%，成交量 > MA(V,5)，收盘 > MA5。
一、逐笔（8 个宽基 2011-08 起，每轮第一个 B，含灰色 △B，间隔 ≥20 天）：B 点当天买 vs 第一次"强"那天买，之后 20/60/120 天；
    N 天内等不到"强"的 B 点后来怎么样。
二、分档仓位回测（现行规则，6 个有估值数据的指数）：B 点之后 N 天内出现"强"才按 B 的档位买，没出现就不买。

结果：
  用户举的两个例子都能标出：上证 2025-12-19～30 几次"站上 5 日线 + 量 > 5 日均量"；2026-04-08 四种定义都满足（唯一的放量光头阳线），
  之前 3-25～4-01 几次站上 5 日线但量都低于均量。
  逐笔（294 轮）："站上 5 日线 + 放量"太常见，99% 的 B 点 20 天内都会出现（平均第 5 天），等它多付 1.1%，之后 60 天上涨 53% 对 B 当天 52%，
  120 天反而更差（48% / -0.8% 对 52% / +0.9%）。补量 2 天：87% 等得到，多付 1.5%，之后不比 B 当天好（60 天中位 +0.7% 对 +1.8%）；
  但等不到的 37 轮 B 当天买 60 天只有 27% 上涨、中位 -4.3%——"一直没有补量"的 B 点确实差，只是事后才知道。
  ★B 76 轮：全部 20 天内等到"强"，多付 2～4%，60 天中位 +2.4～4.0% 对 B 当天 +4.6%。
  分档回测：所有"等强再买"的版本年化都更低（+3.3%～+4.1% 对 +4.4%），回撤略小（最好 -19.9% 对 -24.1%，主要因为平均仓位从 23% 降到 17%），
  收益/回撤比没有改善；"B 当天买一半、强再补一半"+4.3% / -24.6%。不采用，DINGDI 不加"强"标签。
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

V = "D:/TDXapp/vipdoc"


def strong(d: pd.DataFrame) -> dict[str, pd.Series]:
    C, O, H, Vo = d.close, d.open, d.high, d.volume
    ma5 = C.rolling(5).mean()
    up = C > ma5
    v5, v10 = Vo > Vo.rolling(5).mean(), Vo > Vo.rolling(10).mean()
    s1 = up & v5
    return {
        "站上5日线+量>5日均量": s1,
        "站上5日线+量>10日均量": up & v10,
        "补量2天": s1 & s1.shift(1, fill_value=False),
        "放量光头阳线": (C / C.shift() - 1 >= 0.015) & ((H - C) / C <= 0.002) & v5 & up,
    }


def part1(mkt, N=20):
    rows = []
    for name, code in BROAD:
        d = read_day(f"{V}/{code[:2]}/lday/{code}.day")
        s = signals(code, mkt, 200.0)
        st = strong(d)
        C = s.close.to_numpy()
        ok = (s.b1 | s.b2).to_numpy()
        last = -999
        for i in np.flatnonzero(ok):
            if s.index[i] < pd.Timestamp("2011-08-01") or i - last < 20 or i + N + 120 >= len(C):
                continue
            last = i
            r = {"级别": "★B" if s.b2.iloc[i] else "△B", "B当天买_f60": C[i + 60] / C[i] - 1, "B当天买_f120": C[i + 120] / C[i] - 1}
            for k, ser in st.items():
                f = np.flatnonzero(ser.reindex(s.index, fill_value=False).to_numpy()[i + 1:i + N + 1])
                if len(f):
                    j = i + 1 + f[0]
                    r[k] = (j - i, C[j] / C[i] - 1, C[j + 20] / C[j] - 1, C[j + 60] / C[j] - 1, C[j + 120] / C[j] - 1, C[i + 60] / C[i] - 1)
                else:
                    r[k] = None
            rows.append(r)
    R = pd.DataFrame(rows)
    print(f"== 一、逐笔：{len(R)} 轮 B 点（★B {(R.级别 == '★B').sum()} 轮），'强'要在 B 点后 {N} 天内 ==")
    print(f"  基准 B 当天买：60 天上涨 {(R.B当天买_f60 > 0).mean():.0%}、中位 {R.B当天买_f60.median():+.1%}；120 天上涨 {(R.B当天买_f120 > 0).mean():.0%}、中位 {R.B当天买_f120.median():+.1%}")
    for k in strong(read_day(f"{V}/sh/lday/sh000001.day")).keys():
        got = R[R[k].notna()]
        miss = R[R[k].isna()]
        t = np.array(got[k].tolist())
        print(f"  {k}：{len(got)} 轮等到（{len(got) / len(R):.0%}），平均第 {t[:, 0].mean():.1f} 天，比 B 当天多付 中位 {np.median(t[:, 1]):+.1%}；"
              f"从'强'那天买 60 天上涨 {(t[:, 3] > 0).mean():.0%}、中位 {np.median(t[:, 3]):+.1%}，120 天 {(t[:, 4] > 0).mean():.0%}、{np.median(t[:, 4]):+.1%}"
              f"（同一批 B 当天买 60 天 {(t[:, 5] > 0).mean():.0%}、{np.median(t[:, 5]):+.1%}）；"
              f"等不到的 {len(miss)} 轮 B 当天买 60 天上涨 {(miss.B当天买_f60 > 0).mean():.0%}、中位 {miss.B当天买_f60.median():+.1%}")


def run(s, strong_ser, N, mode):
    """mode: none 现行；confirm 等'强'再按 B 的档位买；half B 当天买一半、'强'再补一半；gray 同 confirm 但灰色 △B 确认后也买 20%。"""
    c, z = s.close.to_numpy(), strong_ser.to_numpy()
    p, pos, turn = 0.0, np.zeros(len(s)), np.zeros(len(s))
    buy_end, buy_added, sell_end = -1, 0.0, -1
    pend, pend_until = 0.0, -1
    for i, r in enumerate(s.to_dict("records")):
        new = p
        tier = base(r) if (r["b1"] or r["b2"]) else 0.0
        gray = (r["b1"] and not r["b2"] and r["st"] >= 5)
        now = 0.0 if gray else tier
        if mode == "none":
            want = now
        else:
            if r["b1"] or r["b2"]:
                t = tier if mode == "gray" else now
                if t > 0:
                    pend, pend_until = max(pend if i <= pend_until else 0.0, t), i + N
            want = 0.0
            if mode == "half":
                want = now / 2
            if i <= pend_until and pend > 0 and z[i] and i > 0:
                want = max(want, pend)
                pend = 0.0
        if want > 0:
            if i > buy_end:
                buy_end, buy_added = i + EPISODE, 0.0
            add = min(max(0.0, want - buy_added), 1 - new)
            if add > 0:
                new += add
                buy_added += add
        if r["s"] and i > sell_end and new > 0:
            sell_end = i + EPISODE
            new = max(0.0, new - 0.2)
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
        st = {k: x.reindex(s.index, fill_value=False) for k, x in strong(read_day(f"{V}/{code[:2]}/lday/{code}.day")).items()}
        for lo, hi, per in (("2011-08-01", "2026-12-31", "全期"), ("2011-08-01", "2018-12-31", "2011-2018"), ("2019-01-01", "2026-12-31", "2019-2026")):
            m = (s.index >= lo) & (s.index <= hi)
            x = s[m]
            eq, pos = run(x, st["站上5日线+量>5日均量"][m], 20, "none")
            a = stats(eq, pos, 0)
            rows.append({"区间": per, "方法": "现行规则", "年化": a["年化收益"], "最大回撤": a["最大回撤"], "平均仓位": a["平均仓位"]})
            for k, ser in st.items():
                for N in (10, 20):
                    for mode, lab in (("confirm", "等强再买"), ("half", "B买一半+强补一半"), ("gray", "等强再买(含灰△B)")):
                        if mode != "confirm" and (N != 20 or k != "站上5日线+量>5日均量"):
                            continue
                        eq, pos = run(x, ser[m], N, mode)
                        a = stats(eq, pos, 0)
                        rows.append({"区间": per, "方法": f"{k}·{N}天·{lab}", "年化": a["年化收益"], "最大回撤": a["最大回撤"], "平均仓位": a["平均仓位"]})
    R = pd.DataFrame(rows)
    print("\n== 二、分档仓位回测（6 个指数平均）==")
    t = R.groupby(["方法", "区间"], sort=False)[["年化", "最大回撤"]].mean().unstack("区间").map(lambda v: f"{v:+.1%}")
    t["平均仓位"] = R[R.区间 == "全期"].groupby("方法", sort=False).平均仓位.mean().map("{:.0%}".format)
    print(t.to_string())


def examples():
    d = read_day(f"{V}/sh/lday/sh000001.day")
    st = strong(d)
    for lo, hi in (("2025-12-15", "2026-01-08"), ("2026-03-20", "2026-04-15")):
        x = pd.DataFrame({k: v.loc[lo:hi].map({True: "强", False: ""}) for k, v in st.items()})
        x.insert(0, "量/5日均量", (d.volume / d.volume.rolling(5).mean()).loc[lo:hi].round(2))
        x.insert(0, "收盘/MA5", (d.close / d.close.rolling(5).mean()).loc[lo:hi].round(3))
        print(x.to_string())


def main():
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 220)
    mkt = load_market()
    examples()
    part1(mkt, 20)
    part2(mkt)


if __name__ == "__main__":
    main()
