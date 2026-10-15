"""底部信号的两个检验（8 个指数，2011-08 起）：

1. 收盘时恐慌是否确认：★B（收盘下跌家数 ≥92%）对比"差一点"（△B 当天收盘下跌家数 80%–92%）和普通 △B。
   盘中涨跌家数没有历史记录，"盘中出现过 ★B、收盘消失"无法逐日还原，"差一点"是最接近的替代。
2. 信号出现时所处的月线趋势，用已走完的上一个月判断（和月线图上看到的一致，不受当天暴跌影响）：
   - 向上且在上方：上月月线 BBI 高于前一个月，且上月收盘在 BBI 之上（2025-04-07 属于这种）；
   - 向下且在下方：上月月线 BBI 低于前一个月，且上月收盘在 BBI 之下；
   - 其他。
收益都从信号当天收盘算起。同一指数、同一类信号间隔不超过 20 个交易日的算同一次，只取第一次。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dingdi  # noqa: E402
from monthly_bbi import month_end_bbi  # noqa: E402
from tdx_reader import read_day  # noqa: E402

IDX = [("上证50", "sh000016"), ("沪深300", "sh000300"), ("中证500", "sh000905"), ("中证1000", "sh000852"),
       ("上证指数", "sh000001"), ("深证成指", "sz399001"), ("创业板指", "sz399006"), ("科创50", "sh000688")]
GAP = 20


def regime_of(me: pd.DataFrame, t: pd.Timestamp) -> str:
    prev = me[me.index < t.to_period("M").start_time].dropna()
    if len(prev) < 2:
        return "其他"
    up = prev.bbi.iloc[-1] > prev.bbi.iloc[-2]
    above = prev.close.iloc[-1] > prev.bbi.iloc[-1]
    return "向上且在上方" if up and above else "向下且在下方" if not up and not above else "其他"


def events(start="2011-08-01") -> pd.DataFrame:
    rows = []
    for name, code in IDX:
        d = read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day")
        r = dingdi.compute(d)
        c, hi, lo, bbi = d.close, d.high, d.low, r["月线BBI"]
        me = month_end_bbi(c)
        ratio = (d.dec / (d.adv + d.dec)).to_numpy()
        b1, b2 = r["底部一级"].to_numpy(), r["底部二级"].to_numpy()
        cat = np.where(b2, "★B（收盘下跌≥92%）", np.where(b1 & (ratio >= 0.8), "差一点（△B，收盘下跌80–92%）",
                                                       np.where(b1, "普通△B", "")))
        last = {}
        for k in np.flatnonzero(cat != ""):
            t, ca = d.index[k], cat[k]
            first = k - last.get(ca, -10**9) > GAP
            last[ca] = k
            if not first or t < pd.Timestamp(start) or k + 1 >= len(c):
                continue
            f = lambda n: c.iloc[k + n] / c.iloc[k] - 1 if k + n < len(c) else np.nan  # noqa: E731
            rows.append({"指数": name, "日期": t, "类别": ca, "趋势": regime_of(me, t),
                         "20天内最大反弹": hi.iloc[k + 1:k + 21].max() / c.iloc[k] - 1,
                         "60天": f(60), "120天": f(120),
                         "60天内再跌": lo.iloc[k + 1:k + 61].min() / c.iloc[k] - 1,
                         "120天后在月BBI上方": float(c.iloc[k + 120] > bbi.iloc[k + 120]) if k + 120 < len(c) else np.nan})
    return pd.DataFrame(rows)


def summarize(E: pd.DataFrame, by) -> pd.DataFrame:
    g = E.groupby(by)
    up = lambda x: (x.dropna() > 0).mean()  # noqa: E731
    return pd.DataFrame({
        "次数": g.size(),
        "20天内最大反弹": g["20天内最大反弹"].median(),
        "60天上涨比例": g["60天"].apply(up), "60天中位": g["60天"].median(),
        "120天上涨比例": g["120天"].apply(up), "120天中位": g["120天"].median(),
        "60天内再跌": g["60天内再跌"].median(),
        "120天后站上月BBI": g["120天后在月BBI上方"].mean(),
    })


def current() -> pd.DataFrame:
    rows = []
    for name, code in IDX:
        d = read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day")
        r = dingdi.compute(d)
        t = d.index[-1]
        rows.append({"指数": name, "数据日期": t.date(), "收盘": d.close.iloc[-1], "月线BBI": round(r["月线BBI"].iloc[-1], 1),
                     "离月线BBI": d.close.iloc[-1] / r["月线BBI"].iloc[-1] - 1,
                     "趋势（按上月）": regime_of(month_end_bbi(d.close), t + pd.offsets.MonthBegin(0))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    E = events()
    print("== 1. 收盘时恐慌是否确认 ==")
    print(summarize(E, "类别").round(3).to_string())
    print("\n== 2. 按月线趋势分组 ==")
    print(summarize(E, ["类别", "趋势"]).round(3).to_string())
    x = E[E["类别"].str.startswith("★B")].copy()
    x["月"] = x["日期"].dt.strftime("%Y-%m")
    print("\n== ★B 逐次（按月合并）==")
    for m, g in x.groupby("月"):
        print(f"  {m}  {len(g)} 个指数  {'/'.join(sorted(set(g['趋势'])))}  20天内最大反弹 {g['20天内最大反弹'].median():+.0%}"
              f"  60天 {g['60天'].median():+.0%}  120天 {g['120天'].median():+.0%}  60天内再跌 {g['60天内再跌'].median():+.0%}")
    print("\n== 现在 ==")
    print(current().round(3).to_string(index=False))
