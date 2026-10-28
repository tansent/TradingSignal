"""调整中"抗跌"或"跌深后迅速拉起"的板块以后更强、"跌深且久久不拉起"的板块难涨，能不能用数据证实（2026-10-19，用户转述的说法）。

做法：125 个通达信行业指数，2012 年起每个月末给每个板块算：
  回撤 = 收盘 / 一年最高收盘 - 1（跌了多少）；反弹 = 收盘 / 近 60 天最低收盘 - 1（从低点拉起多少）；
  离高点天数 = 距一年最高收盘的交易日数（跌了多久）。
看之后 3/6/12 个月的"超额"：板块涨幅减去当月全部板块的平均涨幅（扣掉大盘涨跌），以及跑赢一半板块的比例。
每个月把板块按回撤、反弹各分三档（当月横向排名），比较各组的平均超额和跑赢比例；
另外单看"调整中"的月份（全部板块等权平均离一年高点跌了 10% 以上）。

结果（177 个月末，2012-01 到 2026-09）：
  1. 跌得少的三分之一，之后 12 个月超额 +2.0%、跑赢一半板块 55%；跌得深的 -2.1%、44%。14 年里 11 年成立，
     反过来的是 2014（-10%）、2022（-1%）、2024（-9%）——大行情启动时跌得最深的反弹最猛。
     只看调整中的月份，差距变小（+1.3% 对 -1.4%），3 个月内还略反过来。
  2. 跌得深的板块里，从低点拉起多的和没拉起的没有区别（12 个月 -1.8% 对 -2.9%，跑赢都是 43%），按年 7 正 7 负。
     "跌深 ≥30%、离高点 ≥120 天、拉起 <8%"和"跌深 ≥30%、已拉起 ≥15%"都是 -1.7%、47%。
  3. 上证 ★B 那天（11 轮），跌得深的板块 60 天反弹 +7.6%，跌得少的 +4.6%（11 轮里 7 轮如此）。
  结论：说法的前一半（抗跌的板块以后更强）有，但幅度不大、大底时反过来；"迅速拉起"和"久久不起"的区分没有证据。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from macd_monthly import industries  # noqa: E402


def panel() -> pd.DataFrame:
    C = pd.DataFrame(industries()).sort_index()
    C = C[C.index >= "2011-01-01"]
    dd = C / C.rolling(250, min_periods=200).max() - 1
    rb = C / C.rolling(60, min_periods=50).min() - 1
    since = C.rolling(250, min_periods=200).apply(lambda x: len(x) - 1 - np.nanargmax(x), raw=True)
    mkt = C.pct_change(fill_method=None).mean(axis=1).fillna(0).add(1).cumprod()
    mdd = mkt / mkt.rolling(250, min_periods=200).max() - 1
    ends = C.groupby(C.index.to_period("M")).apply(lambda x: x.index.max())
    ends = ends[(ends >= pd.Timestamp("2012-01-01"))].iloc[:-1]           # 最后一个月没走完
    me = C.loc[ends.to_numpy()]
    rows = []
    for k, t in enumerate(ends):
        fut = {h: (me.iloc[k + h] / me.iloc[k] - 1) if k + h < len(me) else None for h in (3, 6, 12)}
        x = pd.DataFrame({"dd": dd.loc[t], "rb": rb.loc[t], "since": since.loc[t]})
        for h, f in fut.items():
            x[f"x{h}"] = (f - f.mean()) if f is not None else np.nan
            x[f"w{h}"] = (f > f.median()).where(f.notna()) if f is not None else np.nan
        x["dd档"] = pd.qcut(x.dd.rank(method="first"), 3, labels=["跌得深", "中间", "跌得少"])
        x["rb档"] = pd.qcut(x.rb.rank(method="first"), 3, labels=["没拉起", "中间", "拉起多"])
        x["月"], x["调整中"] = t, mdd.loc[t] <= -0.10
        rows.append(x.dropna(subset=["dd", "rb"]))
    return pd.concat(rows)


def show(P: pd.DataFrame, by, label: str):
    g = P.groupby(by, observed=True)
    out = pd.DataFrame({f"{h}个月超额": g[f"x{h}"].mean().map(lambda v: f"{v * 100:+.1f}%") for h in (3, 6, 12)})
    for h in (6, 12):
        out[f"{h}个月跑赢一半板块"] = g[f"w{h}"].apply(lambda s: f"{s.dropna().mean():.0%}")
    out["样本"] = g.size()
    print(f"\n== {label} ==")
    print(out.to_string())


def main():
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)
    P = panel()
    print(f"{P.月.nunique()} 个月末，{len(P)} 个板块-月；其中调整中的月份 {P[P.调整中].月.nunique()} 个")
    show(P, "dd档", "一、按跌了多少分（全部月份）")
    show(P[P.调整中], "dd档", "一'、按跌了多少分（只看调整中的月份）")
    show(P, ["dd档", "rb档"], "二、跌了多少 × 从低点拉起多少（全部月份）")
    deep = P[P.dd <= -0.30]
    deep = deep.assign(组=np.select([(deep.rb >= 0.15), (deep.rb < 0.08) & (deep.since >= 120)],
                                    ["跌深 ≥30%，已从低点拉起 ≥15%", "跌深 ≥30%，离高点 ≥120 天，拉起 <8%"], "其他跌深"))
    show(deep, "组", "三、跌深的板块里：快速拉起 vs 久久不起")
    for lo, hi in (("2012", "2018-12-31"), ("2019", "2026-12-31")):
        show(P[(P.月 >= lo) & (P.月 <= hi)], ["dd档", "rb档"], f"二、分段 {lo[:4]}–{hi[:4]}")
    sp = P.groupby([P.月.dt.year, "dd档"], observed=True).x12.mean().unstack()
    yr = (sp["跌得少"] - sp["跌得深"]).dropna()
    print("\n跌得少减跌得深，之后 12 个月超额，按年：" + "  ".join(f"{y} {v * 100:+.0f}%" for y, v in yr.items()))
    D = P[P.dd档 == "跌得深"]
    dp = D.groupby([D.月.dt.year, "rb档"], observed=True).x12.mean().unstack()
    yr2 = (dp["拉起多"] - dp["没拉起"]).dropna()
    print("跌得深的板块里，拉起多减没拉起，按年：" + "  ".join(f"{y} {v * 100:+.0f}%" for y, v in yr2.items()))


if __name__ == "__main__":
    main()
