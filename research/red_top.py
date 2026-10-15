"""红色 ◆S（二次冲顶）：橙色 ◆S（抱团顶预警）之后，又起一波，同时拥挤度更高、情绪亢奋、放量。

对每个指数（8 个宽基 + 通达信研究行业二级指数 + 半导体设备），在每次橙色 ◆S 之后 120 个交易日内，逐日检查：
  再起一波：收盘比橙色 ◆S 当天收盘高 ≥ RISE（同一指数自己的价格）
  拥挤度：  当天 ≥ CROWD
  情绪亢奋：近 3 天情绪最高 ≥ SENTI（DINGDI 的情绪）
  放量：    当天成交额 ≥ 20 日均额 × VOLX
满足的每一天都是红色 ◆S（可以多次出现）。比较红色 ◆S、橙色 ◆S 之后的表现，以及离这一轮最高点有多近。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dingdi  # noqa: E402
from crowding import load as load_crowd, top_zone  # noqa: E402
from point_line_surface import cfg  # noqa: E402
from tdx_reader import read_day  # noqa: E402

V = Path("D:/TDXapp/vipdoc")
BROAD = [("上证50", "sh000016"), ("沪深300", "sh000300"), ("中证500", "sh000905"), ("中证1000", "sh000852"),
         ("上证指数", "sh000001"), ("深证成指", "sz399001"), ("创业板指", "sz399006"), ("科创50", "sh000688")]
WIN = 120


RISE, CROWD, SENTI, VOLX = 0.05, 50, 85, 1.3     # 默认参数（和通达信 DINGDI 里的红色 ◆S 一致）
# 橙色 ◆S 之后前 5 个交易日不出红色 ◆S（2026-10-16，用户提议）：2015 年后三轮、全部指数，距橙 1～5 天的红色 ◆S
# 之后 20 天下跌比例只有 49%、中位 +0.3%（和随便哪天差不多），6～30 天是 81%、-11.6%。
# 两种红色 ◆S 合计，改后 20 天下跌 81%、中位 -10.9%（改前 70%、-6.1%），60 天 77%、-11.6%，60 天内回撤中位 -19%。
# 试过按指数波动率缩放"涨 5%"和"离前高 3%"（上证低一些、科创高一些）：总体结果几乎不变，
# 上证反而少了 2026-06-22 那次；上证 2021-02-18 的真顶抓不到是因为缩量（量/20 日均额 1.14）和融资在降，不是涨幅门槛。所以不分指数。
GAP_MIN = 6


def red_signal(d: pd.DataFrame, r: pd.DataFrame, crowd: pd.Series, mark: pd.Series) -> pd.Series:
    """每天是否出现红色 ◆S：最近一次橙色 ◆S 在 6～120 个交易日前，收盘比那天高 ≥5%，拥挤度 ≥50%，
    近 3 天情绪最高 ≥85，成交额 ≥ 20 日均额 1.3 倍。d 是该指数日线，r 是 dingdi.compute(d)，mark 是全市场的橙色 ◆S（0/1）。"""
    c = d.close
    mk = mark.reindex(c.index).fillna(0).astype(bool).to_numpy()
    last = np.full(len(c), -1)
    li = -1
    for i in range(len(c)):
        if mk[i]:
            li = i
        last[i] = li
    gap = np.where(last >= 0, np.arange(len(c)) - last, 10**9)
    base = np.where(last >= 0, c.to_numpy()[np.maximum(last, 0)], np.nan)
    cond = ((gap >= GAP_MIN) & (gap <= WIN) & (c.to_numpy() >= base * (1 + RISE))
            & (crowd.reindex(c.index).to_numpy() >= CROWD)
            & (r["情绪"].rolling(3).max().to_numpy() >= SENTI)
            & ((d.amount / d.amount.rolling(20).mean()).to_numpy() >= VOLX))
    return pd.Series(cond, index=c.index)


WIN2, NEAR2, SENTI2 = 30, 0.97, 70                # M 顶（第二个顶）的参数（和通达信 DINGDI 一致）


def margin_chg20_known(mg: pd.Series) -> pd.Series:
    """融资余额 20 日变化，按"收盘时已知"对齐：融资余额次日早上才公布，所以 t 日用的是 t-1 日的数。"""
    return (mg / mg.shift(20) - 1).shift(1)


def red_signal_mtop(d: pd.DataFrame, r: pd.DataFrame, crowd: pd.Series, mark: pd.Series, mg20: pd.Series) -> pd.Series:
    """红色 ◆S 的第二种情形（M 顶）：橙色 ◆S 之后 6～30 个交易日内，收盘在"橙色 ◆S 前 20 天至今"最高价的 3% 以内，
    拥挤度 ≥ max(48, 橙色 ◆S 当天的拥挤度)，5 日均额 ≥ 20 日均额，近 3 天情绪最高 ≥70，融资余额 20 日不降（用前一天的数）。"""
    c, h = d.close, d.high
    mk = mark.reindex(c.index).fillna(0).astype(bool).to_numpy()
    last = np.full(len(c), -1)
    li = -1
    for i in range(len(c)):
        if mk[i]:
            li = i
        last[i] = li
    gap = np.where(last >= 0, np.arange(len(c)) - last, 10**9)
    crv = crowd.reindex(c.index).to_numpy()
    cr_at = np.where(last >= 0, crv[np.maximum(last, 0)], np.nan)
    hv = h.to_numpy()
    hh = np.array([hv[max(0, last[i] - 20):i + 1].max() if 0 <= last[i] and gap[i] <= WIN2 else np.nan for i in range(len(c))])
    ok = ((gap >= GAP_MIN) & (gap <= WIN2) & (c.to_numpy() >= NEAR2 * hh) & (crv >= np.maximum(48, cr_at))
          & ((d.amount.rolling(5).mean() / d.amount.rolling(20).mean()).to_numpy() >= 1.0)
          & (r["情绪"].rolling(3).max().to_numpy() >= SENTI2)
          & (mg20.reindex(c.index).to_numpy() >= 0))
    return pd.Series(ok, index=c.index)


def universe():
    inds = [(f[0], "sh" + f[1]) for f in cfg() if len(f[5]) == 5]
    return BROAD + inds + [("半导体设备", "sh881321")]


def red_days(d: pd.DataFrame, r: pd.DataFrame, crowd: pd.Series, marks, rise, crowd_lv, senti, volx) -> pd.DataFrame:
    c, amt = d.close, d.amount
    cr = crowd.reindex(c.index)
    s3 = r["情绪"].rolling(3).max()
    vx = amt / amt.rolling(20).mean()
    rows = []
    for m in marks:
        if m not in c.index:
            continue
        k0 = c.index.get_loc(m)
        for k in range(k0 + 1, min(k0 + WIN + 1, len(c))):
            if c.iloc[k] >= c.iloc[k0] * (1 + rise) and cr.iloc[k] >= crowd_lv and s3.iloc[k] >= senti and vx.iloc[k] >= volx:
                rows.append({"m": m, "k": k, "k0": k0})
    return pd.DataFrame(rows)


def evaluate(rise=0.05, crowd_lv=50, senti=85, volx=1.3, show=False):
    crowd = load_crowd()["crowd"]
    hs = read_day(V / "sh" / "lday" / "sh000300.day")
    tz = top_zone(crowd, hs)
    marks = list(tz.index[(tz.mark == 1) & (tz.index >= "2015-01-01")])     # 2014-12 那次之后是大牛市，单独看
    marks_all = list(tz.index[tz.mark == 1])
    red, orange = [], []
    for name, code in universe():
        p = V / code[:2] / "lday" / f"{code}.day"
        if not p.exists():
            continue
        d = read_day(p)
        if len(d) < 300:
            continue
        r = dingdi.compute(d)
        c, lo, hi = d.close, d.low, d.high
        f = lambda k, n: c.iloc[k + n] / c.iloc[k] - 1 if k + n < len(c) else np.nan  # noqa: E731
        dd = lambda k, n: lo.iloc[k + 1:k + n + 1].min() / c.iloc[k] - 1 if k + 1 < len(c) else np.nan  # noqa: E731
        for m in marks_all:
            if m not in c.index:
                continue
            k0 = c.index.get_loc(m)
            ep_hi = hi.iloc[k0:k0 + WIN + 1].max()
            orange.append({"指数": name, "◆S": m.date(), "20天": f(k0, 20), "60天": f(k0, 60), "60天内最大回撤": dd(k0, 60),
                           "离这轮最高": c.iloc[k0] / ep_hi - 1})
        R = red_days(d, r, crowd, marks_all, rise, crowd_lv, senti, volx)
        for _, x in R.iterrows():
            k, k0 = int(x.k), int(x.k0)
            ep_hi = hi.iloc[k0:k0 + WIN + 1].max()
            red.append({"指数": name, "◆S": x.m.date(), "红◆S": c.index[k].date(), "比橙◆S涨": c.iloc[k] / c.iloc[k0] - 1,
                        "20天": f(k, 20), "60天": f(k, 60), "60天内最大回撤": dd(k, 60), "离这轮最高": c.iloc[k] / ep_hi - 1})
    O, Rd = pd.DataFrame(orange), pd.DataFrame(red)
    return O, Rd


def summary(df: pd.DataFrame, label: str):
    x = df[df["◆S"] >= pd.Timestamp("2015-01-01").date()]
    if not len(x):
        return {"组": label, "次数": 0}
    return {"组": label, "次数": len(x), "涉及指数": x["指数"].nunique(),
            "20天下跌比例": (x["20天"].dropna() < 0).mean(), "20天中位": x["20天"].median(),
            "60天下跌比例": (x["60天"].dropna() < 0).mean(), "60天中位": x["60天"].median(),
            "60天内最大回撤中位": x["60天内最大回撤"].median(), "离这轮最高中位": x["离这轮最高"].median()}


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    O, _ = evaluate()
    rows = [summary(O, "橙色 ◆S 当天（所有指数）")]
    for rise, cl, se, vx in ((0.05, 50, 85, 1.3), (0.05, 48, 85, 1.3), (0.03, 50, 80, 1.2), (0.08, 50, 85, 1.3), (0.05, 50, 90, 1.5)):
        _, Rd = evaluate(rise, cl, se, vx)
        # 同一指数同一轮只取第一个红 ◆S，另算全部红 ◆S 日
        first = Rd.groupby(["指数", "◆S"]).head(1) if len(Rd) else Rd
        rows.append(summary(Rd, f"红 ◆S 全部日（涨≥{rise:.0%} 拥挤≥{cl} 情绪≥{se} 放量≥{vx}）"))
        rows.append(summary(first, "   其中每轮第一个"))
    print("（只统计 2015 年以后的三轮：2017-11、2021-01、2026-05；2014-12 之后是大牛市，另列）")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    _, Rd = evaluate()
    print("\n默认参数下，各轮出现红 ◆S 的指数数量和日期范围：")
    for m, g in Rd.groupby("◆S"):
        print(f"  {m}: {g['指数'].nunique()} 个指数，{len(g)} 个红◆S 日，{g['红◆S'].min()} → {g['红◆S'].max()}")
    print("\n宽基和半导体设备的红 ◆S：")
    keep = [n for n, _ in BROAD] + ["半导体设备", "半导体"]
    print(Rd[Rd["指数"].isin(keep)].round(3).to_string(index=False))
