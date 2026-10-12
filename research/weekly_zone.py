"""周线区域，与通达信公式 ZQ 相同。

周线 △B/★B 之后 26 周为底部区域，周线 ★S/▽S 之后 26 周为顶部区域；新信号重新计时，反方向信号结束当前区域。
区域从信号周之后的第一个交易日起生效，只用已经收盘的周线。
"""
import pandas as pd

import dingdi
from weekly_signals import weekly

ZONE_WEEKS = 26


def weekly_table(daily: pd.DataFrame, zone_weeks: int = ZONE_WEEKS) -> pd.DataFrame:
    """每根周线的信号和区域（区域值是"这一周收盘后"的状态，下一周起生效）。"""
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
    sig = pd.Series("", index=w.index)
    for col, lab in (("底部一级", "△B"), ("底部二级", "★B"), ("顶部一级", "▽S"), ("顶部", "★S")):
        sig[r[col].to_numpy()] = lab
    return pd.DataFrame({"z_bot": in_bot, "z_top": in_top, "w_buy": wb, "w_sell": wt, "w_brk": r["破月BBI"].to_numpy(),
                         "w_sig": sig.to_numpy()}, index=w.index)


def zones(daily: pd.DataFrame, zone_weeks: int = ZONE_WEEKS) -> pd.DataFrame:
    """映射到日线：z_bot/z_top 是当天生效的区域；w_buy/w_sell/w_brk 只在周线信号那周的最后一个交易日为真。"""
    W = weekly_table(daily, zone_weeks)
    zone = W[["z_bot", "z_top"]].astype(float).reindex(daily.index, method="ffill").shift(1).fillna(0).astype(bool)
    ev = W[["w_buy", "w_sell", "w_brk"]].astype(float).reindex(daily.index).fillna(0).astype(bool)
    return zone.join(ev)


def current(daily: pd.DataFrame, zone_weeks: int = ZONE_WEEKS) -> dict:
    """最新交易日的周线区域：名称、从哪天起生效、没有新信号时预计到哪天、最近一次周线信号。"""
    W = weekly_table(daily, zone_weeks)
    if daily.index[-1].weekday() != 4:          # 本周还没走完，本周的周线信号还不算数
        W = W.iloc[:-1]
    z = zones(daily, zone_weeks)
    last = z.iloc[-1]
    name = "顶部区域" if last.z_top else "底部区域" if last.z_bot else ""
    out = {"wzone": name, "wzone_since": None, "wzone_until": None, "wsig": None, "wsig_date": None}
    fired = W[W["w_sig"] != ""]
    if len(fired):
        out["wsig"], out["wsig_date"] = fired["w_sig"].iloc[-1], fired.index[-1]
    if name:
        col = "z_top" if last.z_top else "z_bot"
        run = z[col].to_numpy()
        k = len(run) - 1
        while k > 0 and run[k - 1]:
            k -= 1
        out["wzone_since"] = z.index[k]
        # 区域覆盖信号周之后的 zone_weeks+1 周，到那一周的周五为止
        out["wzone_until"] = out["wsig_date"] + pd.Timedelta(weeks=zone_weeks + 1) if out["wsig_date"] is not None else None
    return out
