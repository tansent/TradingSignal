"""破月BBI 之后的"冷却期"：刚跌破月线 BBI 的一段时间里出现的底部信号，是不是不该接？（2011-08 起，8 个指数）

破月BBI：收盘连续 2 天在月线 BBI 下方（与 DINGDI 相同，月线 BBI 当月用当天收盘代替）。
从破位那天起算一轮"破位期"，直到收盘连续 10 天站回月线 BBI 上方才结束（期间短暂站上不算结束）。
底部信号（△B/★B，同一指数 20 天内算一次）按出现时处在破位期的第几天分组；不在破位期的算"月线BBI上方"。
再把"破位后 N 天内不接底部信号"放进 strategy_backtest 的"C 分档 + 破月BBI离场"里，看收益和回撤。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dingdi  # noqa: E402
import strategy_backtest as sb  # noqa: E402
from monthly_bbi import asof_bbi  # noqa: E402
from yangge_rules import IDX, START, GAP, load, up  # noqa: E402

RECOVER = 10


def days_since_break(c: pd.Series) -> pd.Series:
    """每天处在破位期的第几天（破位当天 = 0）；不在破位期为 NaN。"""
    bbi = asof_bbi(c)
    below = (c < bbi).to_numpy()
    out = np.full(len(c), np.nan)
    start, above_run = None, 0
    for i in range(len(c)):
        if start is None:
            if i >= 1 and below[i] and below[i - 1]:
                start, above_run = i, 0
        else:
            above_run = 0 if below[i] else above_run + 1
            if above_run >= RECOVER:
                start = None
        if start is not None:
            out[i] = i - start
    return pd.Series(out, index=c.index)


def bucket(x: float) -> str:
    if np.isnan(x):
        return "0 月线BBI上方（不在破位期）"
    return "1 破位后 0–20 天" if x <= 20 else "2 破位后 21–60 天" if x <= 60 else "3 破位后 61–120 天" if x <= 120 else "4 破位 120 天以上"


def events() -> pd.DataFrame:
    rows = []
    for name, code in IDX:
        d = load(code)
        r = dingdi.compute(d)
        c, lo = d.close, d.low
        ds = days_since_break(c)
        sig = (r["底部一级"] | r["底部二级"]).to_numpy()
        star = r["底部二级"].to_numpy()
        last = -10**9
        for k in np.flatnonzero(sig):
            first, last = k - last > GAP, k
            if not first or c.index[k] < pd.Timestamp(START) or k + 1 >= len(c):
                continue
            f = lambda n: c.iloc[k + n] / c.iloc[k] - 1 if k + n < len(c) else np.nan  # noqa: E731
            rows.append({"指数": name, "日期": c.index[k], "★B": bool(star[k:k + GAP].any()), "破位天数": ds.iloc[k],
                         "分组": bucket(ds.iloc[k]), "20天": f(20), "60天": f(60), "120天": f(120), "250天": f(250),
                         "60天内再跌": lo.iloc[k + 1:k + 61].min() / c.iloc[k] - 1})
    return pd.DataFrame(rows)


def table(E: pd.DataFrame, by) -> pd.DataFrame:
    g = E.groupby(by)
    return pd.DataFrame({
        "次数": g.size(), "其中有★B": g["★B"].sum(),
        "20天上涨比例": g["20天"].apply(up), "60天上涨比例": g["60天"].apply(up), "60天中位": g["60天"].median(),
        "120天上涨比例": g["120天"].apply(up), "120天中位": g["120天"].median(),
        "250天上涨比例": g["250天"].apply(up), "250天中位": g["250天"].median(),
        "60天内再跌": g["60天内再跌"].median(),
    })


def strategy() -> pd.DataFrame:
    mkt = sb.load_market()
    etf_ok = mkt["net_yi"].first_valid_index() is not None
    tier = lambda r: (0.4 if r["b2"] else 0.2) + (0.2 if r["cold"] else 0) + (0.2 if (etf_ok and r["etf"]) else 0)  # noqa: E731
    lvl3 = lambda r: r["cold"] or (etf_ok and r["etf"])  # noqa: E731
    out = []
    for name, code, start in (("沪深300", "sh000300", "2011-08-01"), ("上证指数", "sh000001", "2011-08-01"),
                              ("创业板指", "sz399006", "2011-08-01"), ("科创50", "sh000688", "2021-01-04")):
        d = load(code)
        s = sb.signals_df(d, mkt, 200.0)
        s["ds"] = days_since_break(d.close)
        s = s[s.index >= start]
        rules = {"现行：C 分档 + 破月BBI离场": tier}
        for n in (20, 60, 120):
            rules[f"破位后 {n} 天内不接B"] = (lambda n: lambda r: 0.0 if (r["ds"] == r["ds"] and r["ds"] <= n) else tier(r))(n)
            rules[f"破位后 {n} 天内只接★★B级"] = (lambda n: lambda r: (tier(r) if lvl3(r) else 0.0)
                                            if (r["ds"] == r["ds"] and r["ds"] <= n) else tier(r))(n)
        for k, bs in rules.items():
            res = sb.run(s, bs, 0.2, "bbi_two")
            out.append({"指数": name, "方法": k, **sb.stats(res["equity"], res["pos"], int(res.trades.iloc[0]))})
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    E = events()
    print("== 底部信号按破月BBI后的天数分组 ==")
    print(table(E, "分组").round(3).to_string())
    print("\n== 只看有 ★B 的 ==")
    print(table(E[E["★B"]], "分组").round(3).to_string())
    print("\n== 策略：破位后 N 天内不接B / 只接★★B级 ==")
    df = strategy()
    for k in ("年化收益", "最大回撤", "平均仓位"):
        df[k] = df[k].map(lambda v: f"{v:+.1%}" if k != "平均仓位" else f"{v:.0%}")
    df["期末倍数"] = df["期末倍数"].round(2)
    print(df.to_string(index=False))
    print("\n现在：")
    for name, code in IDX:
        ds = days_since_break(load(code).close)
        print(f"  {name}: {'破位第 %d 天' % ds.iloc[-1] if ds.iloc[-1] == ds.iloc[-1] else '不在破位期'}")
