"""周线定区域、日线分批：和买入持有、日线 C 方法、周线全进全出放在一起比较。

周线区域（每周最后一个交易日收盘确定，下一个交易日起生效，不用未来数据）：
  周线 △B/★B 之后 ZONE 周为底部区域；周线 ★S/▽S 之后 ZONE 周为顶部区域；新信号结束反方向的区域。
日线分批：
  只在底部区域里按日线底部信号分档买入（△B 20%、★B 40%、中线温度≤10 或国家队 ≥200 亿各加 20%，5 日内同一轮）；
  只在顶部区域里按日线 ★S/▽S 每次卖 20%，日线破月BBI 时卖出剩余全部。
变体：
  Z+  周线信号本身也算一档：周线底部信号当周收盘买 20%（本轮还没买过时），周线顶部信号当周收盘卖 20%。
  Z+安全  Z+ 之外，任何时候周线连续 2 周收在月线 BBI 下方就清仓（防止底部区域买入后长期阴跌）。
起点空仓；空仓资金年化 2%；成本 0.05%；信号当天/当周收盘成交。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dingdi  # noqa: E402
import strategy_backtest as sb  # noqa: E402
from tdx_reader import read_day  # noqa: E402
from weekly_signals import weekly  # noqa: E402

IDX = [("上证指数", "sh000001"), ("深证成指", "sz399001"), ("创业板指", "sz399006"), ("沪深300", "sh000300"),
       ("上证50", "sh000016"), ("中证500", "sh000905"), ("中证1000", "sh000852")]
TIER = lambda r: (0.4 if r["b2"] else 0.2) + (0.2 if r["cold"] else 0) + (0.2 if r["etf"] else 0)  # noqa: E731


def zones(daily: pd.DataFrame, zone_weeks: int) -> pd.DataFrame:
    """周线信号与区域，映射到日线（只用已收盘的周）。"""
    w = weekly(daily)
    r = dingdi.compute(w)
    wb = (r["底部一级"] | r["底部二级"]).to_numpy()
    wt = (r["顶部"] | r["顶部一级"]).to_numpy()
    bot_end = top_end = -1
    in_bot, in_top = [], []
    for i in range(len(w)):
        if wb[i]:
            bot_end, top_end = i + zone_weeks, -1
        if wt[i]:
            top_end, bot_end = i + zone_weeks, -1
        in_bot.append(i <= bot_end)
        in_top.append(i <= top_end)
    W = pd.DataFrame({"z_bot": in_bot, "z_top": in_top, "w_buy": wb, "w_sell": wt, "w_brk": r["破月BBI"].to_numpy()}, index=w.index)
    # 区域从信号周之后的第一个交易日起生效（与通达信 ZQ 公式的 REF(区域,1)#WEEK 一致）；
    # 周线信号本身在该周最后一个交易日收盘成交
    zone = W[["z_bot", "z_top"]].astype(float).reindex(daily.index, method="ffill").shift(1).fillna(0).astype(bool)
    ev = W[["w_buy", "w_sell", "w_brk"]].reindex(daily.index).fillna(False).astype(bool)
    return zone.join(ev)


def run_zone(s: pd.DataFrame, weekly_tranche: bool, safety: bool):
    rows = s.to_dict("records")
    n = len(rows)
    p, pos, turn = 0.0, np.zeros(n), np.zeros(n)
    bend, badd, send = -1, 0.0, -1
    for i, r in enumerate(rows):
        q = p
        in_bot = r["z_bot"] or (weekly_tranche and r["w_buy"])
        want = 0.0
        if in_bot and (r["b1"] or r["b2"]):
            want = TIER(r)
        if weekly_tranche and r["w_buy"]:
            want = max(want, 0.2)
        if want > 0:
            if i > bend:
                bend, badd = i + sb.EPISODE, 0.0
            add = min(max(0.0, want - badd), 1 - q)
            q += add
            badd += add
        in_top = r["z_top"] or (weekly_tranche and r["w_sell"])
        if in_top and q > 0:
            if (r["s"] or (weekly_tranche and r["w_sell"])) and i > send:
                send = i + sb.EPISODE
                q = max(0.0, q - 0.2)
            if r["bbi_two"]:
                q = 0.0
        if safety and r["w_brk"]:
            q = 0.0
        turn[i] = abs(q - p)
        p = q
        pos[i] = p
    c = s.close.to_numpy()
    ret = np.r_[0, c[1:] / c[:-1] - 1]
    held = np.r_[0, pos[:-1]]
    eq = pd.Series(np.cumprod(1 + held * ret + (1 - held) * sb.CASH_YIELD / 252 - turn * sb.COST), index=s.index)
    return eq, pd.Series(pos, index=s.index)


def weekly_all_in_out(s: pd.DataFrame):
    p, pos = 0.0, []
    for r in s.to_dict("records"):
        if p == 0 and r["w_buy"]:
            p = 1.0
        elif p == 1 and r["w_sell"]:
            p = 0.0
        pos.append(p)
    pos = np.array(pos)
    c = s.close.to_numpy()
    ret = np.r_[0, c[1:] / c[:-1] - 1]
    held = np.r_[0, pos[:-1]]
    turn = np.abs(np.diff(np.r_[0, pos]))
    eq = pd.Series(np.cumprod(1 + held * ret + (1 - held) * sb.CASH_YIELD / 252 - turn * sb.COST), index=s.index)
    return eq, pd.Series(pos, index=s.index)


def table(zone_weeks=26, start="2015-06-01", mkt=None):
    mkt = sb.load_market() if mkt is None else mkt
    out = []
    for name, code in IDX:
        d = read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day")
        s = sb.signals(code, mkt, 200.0).join(zones(d, zone_weeks))
        s = s[s.index >= start]
        res = {"买入持有": (s.close / s.close.iloc[0], pd.Series(1.0, index=s.index))}
        r = sb.run(s, TIER, 0.2, "bbi_two")
        res["日线 C"] = (r["equity"], r["pos"])
        res["周线全进全出"] = weekly_all_in_out(s)
        res["Z 周线区域+日线分批"] = run_zone(s, False, False)
        res["Z+ 周线信号也算一档"] = run_zone(s, True, False)
        res["Z+安全 周线破BBI清仓"] = run_zone(s, True, True)
        for k, (eq, pos) in res.items():
            out.append({"指数": name, "方法": k, **sb.stats(eq, pos, 0)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    mkt = sb.load_market()
    order = ["买入持有", "日线 C", "周线全进全出", "Z 周线区域+日线分批", "Z+ 周线信号也算一档", "Z+安全 周线破BBI清仓"]
    T = table(26, mkt=mkt)
    print("===== 2015-06 至 2026-10，区域 26 周 =====")
    for col in ("年化收益", "最大回撤", "平均仓位"):
        t = T.pivot(index="指数", columns="方法", values=col).reindex(columns=order, index=[n for n, _ in IDX])
        t.loc["7 指数平均"] = t.mean()
        print(f"\n-- {col}（%）")
        print((t * 100).round(1).to_string())
    print("\n===== 区域长度敏感性（7 指数平均：年化 / 最大回撤 / 平均仓位）=====")
    for zw in (13, 26, 52):
        T2 = table(zw, mkt=mkt)
        a = T2.groupby("方法")[["年化收益", "最大回撤", "平均仓位"]].mean().reindex(order[3:])
        print(f"区域 {zw} 周：" + "；".join(f"{k} {v['年化收益']:+.1%}/{v['最大回撤']:.0%}/{v['平均仓位']:.0%}" for k, v in a.iterrows()))
