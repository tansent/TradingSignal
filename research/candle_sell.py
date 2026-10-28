"""把"高位大涨大跌"（离一年高点 10% 以内、20 天平均振幅在过去 500 天的前三分之一）当作减仓提示，放进分档仓位回测（2026-10-20）。

基准是现行规则（估值调整 + 阴跌 △B 不买 + ★S/▽S 每轮卖 20 个点 + 破月BBI 清仓），6 个有估值数据的指数。
减仓方式：
  单独一轮卖 20 点：进入区间的日子按"轮"（5 个交易日）卖 20 个点，和 ★S/▽S 各算各的；
  并入 S 点：区间里的日子当作 S 点，和 ★S/▽S 共用一轮；
  进入就卖一半 / 进入就清仓：离开区间 ≥5 天后再次进入才算"进入"。
门槛：振幅分位 ≥2/3、≥0.8、≥0.9；离高点 10% 以内或 5% 以内。

结果（2011-08 起，6 个指数平均）：
  默认门槛（振幅前三分之一、离高点 10% 内，每个指数每年约 35 天在区间里），单独一轮卖 20 点：
  年化 +5.1% 对 +4.4%，最大回撤 -22.0% 对 -24.1%；2011-2018 +1.9% 对 +1.7%，2019-2026 +6.2% 对 +5.6%，两段都更好。
  96 次减仓后 60 天指数下跌 74%、中位 -3.6%。6 个指数里 5 个变好；沪深300 变差（+9.6% → +8.6%），
  因为在 2012-12、2014-12 牛市启动时大涨大跌，被提前卖了。中证500 回撤 -29% → -21%，上证 -20% → -16%。
  16 种组合全期年化都比现行好（+4.5%～+5.1%）；但更严的门槛在 2011-2018 略差（+1.3%～+1.5%），只有默认门槛两段都好。
  "进入就卖一半/清仓"不如每轮卖 20 点。
试用的版本（final()，通达信能画的"20 天平均振幅 ≥ 500 天均值 × 1.1"，从进入区间起每 6 天标一次，和 ★S/▽S 共用一轮）：
  年化 +5.0% 对 +4.4%，最大回撤 -22.4% 对 -24.1%，2011-2018 +1.8% 对 +1.7%，2019-2026 +6.3% 对 +5.6%；每个指数每年约 6 次。
  全部 486 个标记之后 60 天下跌 58%、中位 -1.5%（平时约 48%）；回测里持仓时实际减仓的 89 次之后下跌 75%、中位 -3.6%。
没有采用（2026-10-20）：加进 DINGDI 后看图，震S 常和 B 点、其他 S 点挨得很近，单个标记的参考价值不大（全部标记 60 天下跌只有 58%），
回测的改善主要来自"持仓时卖出"的那部分，用户决定去掉，DINGDI、日报恢复原样。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import valuation  # noqa: E402
from candle_size import candle  # noqa: E402
from ma5_streak import streak  # noqa: E402
from rebound_exit import buy_rule  # noqa: E402
from strategy_backtest import CASH_YIELD, COST, EPISODE, load_market, signals, stats  # noqa: E402
from tdx_reader import read_day  # noqa: E402
from valuation_backtest import IDX  # noqa: E402


def run(s, zone, mode):
    c, z = s.close.to_numpy(), zone.to_numpy()
    p, pos, turn = 0.0, np.zeros(len(s)), np.zeros(len(s))
    buy_end, buy_added, sell_end, zone_end, last_z = -1, 0.0, -1, -1, -999
    sells = []
    for i, r in enumerate(s.to_dict("records")):
        new = p
        want = buy_rule(r) if (r["b1"] or r["b2"]) else 0.0
        if want > 0:
            if i > buy_end:
                buy_end, buy_added = i + EPISODE, 0.0
            add = min(max(0.0, want - buy_added), 1 - new)
            if add > 0:
                new += add
                buy_added += add
        is_s = r["s"] or (mode == "shared" and z[i])
        if is_s and i > sell_end and new > 0:
            sell_end = i + EPISODE
            new = max(0.0, new - 0.2)
            if mode == "shared" and z[i] and not r["s"]:
                sells.append(i)
        if z[i] and new > 0:
            if mode == "own" and i > zone_end:
                zone_end = i + EPISODE
                new = max(0.0, new - 0.2)
                sells.append(i)
            elif mode in ("half", "clear") and i - last_z > EPISODE:
                new = new * 0.5 if mode == "half" else 0.0
                sells.append(i)
        if z[i]:
            last_z = i
        if r["bbi_two"] and new > 0:
            new = 0.0
        turn[i], p, pos[i] = abs(new - p), new, new
    ret = np.r_[0, c[1:] / c[:-1] - 1]
    held = np.r_[0, pos[:-1]]
    eq = np.cumprod(1 + held * ret + (1 - held) * CASH_YIELD / 252 - turn * COST)
    return pd.Series(eq, index=s.index), pd.Series(pos, index=s.index), sells


MODES = {"单独一轮卖20点": "own", "并入S点": "shared", "进入就卖一半": "half", "进入就清仓": "clear"}


def main():
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 220)
    mkt = load_market()
    rows, sell_fwd, freq = [], [], []
    for name, code in IDX:
        s = signals(code, mkt, 200.0)
        v, _ = valuation.series(name)
        q = valuation.pct_asof(v).reindex(s.index, method="ffill")
        s["v"] = np.where(q <= valuation.CHEAP, "low", np.where(q >= valuation.RICH, "high", "mid"))
        s["st"] = streak(s.close)
        d = read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day").reindex(s.index)
        kp = candle(d).rolling(500, min_periods=250).rank(pct=True)
        dd = s.close / s.close.rolling(250, min_periods=200).max() - 1
        C = s.close.to_numpy()
        for lo, hi, per in (("2011-08-01", "2026-12-31", "全期"), ("2011-08-01", "2018-12-31", "2011-2018"), ("2019-01-01", "2026-12-31", "2019-2026")):
            m = (s.index >= lo) & (s.index <= hi)
            x = s[m]
            eq, pos, _ = run(x, pd.Series(False, index=x.index), "none")
            st = stats(eq, pos, 0)
            rows.append({"区间": per, "门槛": "—", "方法": "现行规则", "年化": st["年化收益"], "最大回撤": st["最大回撤"], "平均仓位": st["平均仓位"]})
            for kth, ddth in ((2 / 3, -0.10), (0.8, -0.10), (0.9, -0.10), (2 / 3, -0.05)):
                zone = ((kp >= kth) & (dd >= ddth)).fillna(False)[m]
                tag = f"振幅前{1 - kth:.0%}·离高点{-ddth:.0%}内"
                if per == "全期":
                    freq.append({"门槛": tag, "每年天数": zone.sum() / ((x.index[-1] - x.index[0]).days / 365.25)})
                for k, md in MODES.items():
                    eq, pos, sells = run(x, zone, md)
                    st = stats(eq, pos, 0)
                    rows.append({"区间": per, "门槛": tag, "方法": k, "年化": st["年化收益"], "最大回撤": st["最大回撤"], "平均仓位": st["平均仓位"]})
                    if per == "全期" and kth == 2 / 3 and ddth == -0.10 and md == "own":
                        off = np.flatnonzero(m)[0]
                        for j in sells:
                            i = off + j
                            if i + 60 < len(C):
                                sell_fwd.append(C[i + 60] / C[i] - 1)
    R = pd.DataFrame(rows)
    print("区间内每年天数（6 个指数平均）：" + "，".join(f"{f['门槛']} {f['每年天数']:.0f}" for f in pd.DataFrame(freq).groupby("门槛", sort=False).mean().reset_index().to_dict("records")))
    t = R.groupby(["门槛", "方法", "区间"], sort=False)[["年化", "最大回撤"]].mean().unstack("区间")
    print(t.map(lambda v: f"{v:+.1%}").to_string())
    print(R[R.区间 == "全期"].groupby(["门槛", "方法"], sort=False).平均仓位.mean().map("{:.0%}".format).to_string())
    f = np.array(sell_fwd)
    print(f"\n默认门槛、单独一轮卖 20 点：共卖 {len(f)} 次，卖出后 60 天指数下跌 {(f < 0).mean():.0%}、中位 {np.median(f):+.1%}")


if __name__ == "__main__":
    main()


def zhen_s(d: pd.DataFrame, top: pd.Series) -> pd.Series:
    """试用过的通达信版震S（2026-10-20 加进 DINGDI，同一天看图后去掉）：离一年最高收盘 10% 以内，且 20 天平均振幅 ≥ 500 天均值 × 1.1；
    从进入区间那天（前 5 天都不在区间里）起每 6 天标一次，和 ★S/▽S 同一天时不标。
    通达信写法：振幅均:=MA((H-L)/REF(C,1),20); 高位震荡:=振幅均>=MA(振幅均,500)*1.1 AND C>=HHV(C,250)*0.9;
    进入震荡:=高位震荡 AND COUNT(REF(高位震荡,1),5)=0; 震荡顶:=高位震荡 AND MOD(BARSLAST(进入震荡),6)=0 AND NOT(顶部 OR 顶部一级);"""
    H, L, C = d.high, d.low, d.close
    amp = ((H - L) / C.shift()).rolling(20).mean()
    zone = ((amp >= amp.rolling(500).mean() * 1.1) & (C >= C.rolling(250).max() * 0.9)).fillna(False)
    entry = zone & (zone.shift(1, fill_value=False).astype(int).rolling(5, min_periods=1).sum() == 0)
    since = entry.astype(int).groupby(entry.cumsum()).cumcount().where(entry.cumsum() > 0)
    return zone & (since % 6 == 0) & ~top


def final():
    """试用过的版本：zhen_s()（和当时通达信 DINGDI 的震S 一样），和 ★S/▽S 共用一轮，每轮卖 20 点。"""
    import dingdi
    mkt = load_market()
    rows, per_idx = [], []
    for name, code in IDX:
        s = signals(code, mkt, 200.0)
        v, _ = valuation.series(name)
        q = valuation.pct_asof(v).reindex(s.index, method="ffill")
        s["v"] = np.where(q <= valuation.CHEAP, "low", np.where(q >= valuation.RICH, "high", "mid"))
        s["st"] = streak(s.close)
        d = read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day")
        r = dingdi.compute(d)
        zs = zhen_s(d, (r["顶部"] | r["顶部一级"])).reindex(s.index, fill_value=False)
        for lo, hi, per in (("2011-08-01", "2026-12-31", "全期"), ("2011-08-01", "2018-12-31", "2011-2018"), ("2019-01-01", "2026-12-31", "2019-2026")):
            m = (s.index >= lo) & (s.index <= hi)
            x = s[m]
            for k, z, md in (("现行规则", pd.Series(False, index=x.index), "none"), ("加震S（每轮卖 20 点）", zs[m], "shared")):
                eq, pos, sells = run(x, z, md)
                st = stats(eq, pos, 0)
                rows.append({"区间": per, "方法": k, "年化": st["年化收益"], "最大回撤": st["最大回撤"]})
                if per == "全期":
                    per_idx.append({"指数": name, "方法": k, "年化": f"{st['年化收益']:+.1%}", "回撤": f"{st['最大回撤']:+.1%}",
                                    "震S每年": f"{zs[m].sum() / ((x.index[-1] - x.index[0]).days / 365.25):.1f}"})
    R = pd.DataFrame(rows)
    print(R.groupby(["方法", "区间"], sort=False)[["年化", "最大回撤"]].mean().unstack("区间").map(lambda v: f"{v:+.1%}").to_string())
    print(pd.DataFrame(per_idx).pivot(index="指数", columns="方法", values=["年化", "回撤"]).to_string())
    print(pd.DataFrame(per_idx).drop_duplicates("指数")[["指数", "震S每年"]].to_string(index=False))
