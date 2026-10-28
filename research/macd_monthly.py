"""月线 MACD 死叉之后还会不会"反转"（2026-10-19，用户转述的说法）。

说法：板块指数月线 MACD 死叉后，最多只有日线、周线级别的反弹，不会反转；除了 1998 年那次背离，月线死叉后都没有反转过。

做法：MACD(12,26,9)，与通达信一致（EMA 从第一根 K 线算起）。月末收盘确认死叉（上月 DIF≥DEA、本月 DIF<DEA），
前 24 个月不算（EMA 还没稳定）。"反转"按三种口径数：
  1. 死叉后 12 个月内，收盘超过死叉前 12 个月的最高收盘（回到前高，最严格）；
  2. 死叉后 12 个月，比死叉当月收盘涨 ≥20%；
  3. 死叉后 6 个月内重新金叉，并且金叉后 12 个月内回到前高。
数据：上证指数 1990 起、深证成指 1991 起、沪深 300 2002 起、创业板指 2010 起（新浪，存到 data/index_long/），
上证 50、中证 500、中证 1000（通达信）；通达信 125 个行业指数 2011 起（2013 年以后的死叉）。

结果：
  上证 14 次死叉，一年内回到前高 2 次：1995-12（之后一年 +65%）、1998-07（1999 年 519 行情）。
  7 个宽基 48 次：回前高 17%，一年后下跌 71%，死叉时已离前高 -23%，之后一年中位再跌 -20%。说法在宽基上大体成立，但不是"从没有"。
  行业指数 581 次：回前高 26%，一年后上涨 60%（中位 +5%），之后一年最大反弹中位 +21%。说法在板块指数上不成立。
  处在死叉状态的月份，之后一年中位收益比金叉状态低（宽基 +0.3% 对 +8.0%，行业 -1.0% 对 +9.4%），状态本身有信息。
  我们的 B 点（8 个宽基 2011-08 起）：★B 在月线死叉状态下 60 天上涨 72%、中位 +6.3%，照样有效；
  △B 稍弱（56%、+1.9% 对 60%、+2.7%），差别不大，且和"连续 5 天收在 5 日线下方 △B 不买"重叠，不另加规则。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from point_line_surface import cfg  # noqa: E402
from tdx_reader import read_day  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V = "D:/TDXapp/vipdoc"
LONG = {"上证指数": "sh000001", "深证成指": "sz399001", "沪深300": "sh000300", "创业板指": "sz399006"}
TDX = {"上证50": "sh000016", "中证500": "sh000905", "中证1000": "sh000852"}


def load_long(code: str) -> pd.Series:
    p = ROOT / "data" / "index_long" / f"{code}.csv"
    if not p.exists():
        import akshare as ak
        p.parent.mkdir(exist_ok=True)
        ak.stock_zh_index_daily(symbol=code).to_csv(p, index=False)
    d = pd.read_csv(p, parse_dates=["date"]).set_index("date")
    return d.close.astype(float)


def macd_month(c: pd.Series) -> pd.DataFrame:
    m = c.groupby(c.index.to_period("M")).agg(["last"]).rename(columns={"last": "close"})
    m["day"] = c.index.to_series().groupby(c.index.to_period("M")).max().to_numpy()
    dif = m.close.ewm(span=12, adjust=False).mean() - m.close.ewm(span=26, adjust=False).mean()
    m["dif"], m["dea"] = dif, dif.ewm(span=9, adjust=False).mean()
    m["dead"] = m.dif < m.dea
    m["death"] = m.dead & ~m.dead.shift(1, fill_value=True)
    m["golden"] = ~m.dead & m.dead.shift(1, fill_value=False)
    return m


def events(name: str, c: pd.Series, skip: int = 24) -> pd.DataFrame:
    m = macd_month(c)
    last_full = m.index[-2]                                  # 最后一个月当作没走完，不算
    rows = []
    C = c.to_numpy()
    for k in range(skip, len(m)):
        if m.index[k] > last_full:
            break
        t0 = m.day.iloc[k]
        i0 = c.index.get_loc(t0)
        lo = c.index.searchsorted(t0 - pd.DateOffset(months=12))
        hi = c.index.searchsorted(t0 + pd.DateOffset(months=12), side="right")
        complete = t0 + pd.DateOffset(months=12) <= c.index[-1]
        peak = C[lo:i0 + 1].max()
        fut = C[i0 + 1:hi]
        r = {"指数": name, "月": str(m.index[k]), "dead": bool(m.dead.iloc[k]), "death": bool(m.death.iloc[k]),
             "golden": bool(m.golden.iloc[k]), "完整": complete, "距前高": C[i0] / peak - 1,
             "回前高": bool(len(fut) and fut.max() > peak), "再跌": fut.min() / C[i0] - 1 if len(fut) else np.nan,
             "最大反弹": fut.max() / C[i0] - 1 if len(fut) else np.nan}
        for h in (3, 6, 12):
            r[f"r{h}"] = m.close.iloc[k + h] / m.close.iloc[k] - 1 if k + h < len(m) and m.index[k + h] <= last_full else np.nan
        if r["death"]:
            g = np.flatnonzero(m.golden.to_numpy()[k + 1:])
            r["几个月后金叉"] = g[0] + 1 if len(g) else np.nan
            r["金叉后回前高"] = False
            if len(g) and g[0] + 1 <= 6:
                tg = m.day.iloc[k + 1 + g[0]]
                ig = c.index.get_loc(tg)
                hg = c.index.searchsorted(tg + pd.DateOffset(months=12), side="right")
                r["金叉后回前高"] = bool(C[ig + 1:hg].max() > peak) if hg > ig + 1 else False
        rows.append(r)
    return pd.DataFrame(rows)


def universe() -> dict[str, pd.Series]:
    u = {n: load_long(c) for n, c in LONG.items()}
    for n, c in TDX.items():
        u[n] = read_day(f"{V}/{c[:2]}/lday/{c}.day").close
    return u


def industries() -> dict[str, pd.Series]:
    u = {}
    for f in cfg():
        if len(f[5]) != 5:
            continue
        try:
            d = read_day(f"{V}/sh/lday/sh{f[1]}.day")
        except Exception:  # noqa: BLE001
            continue
        if len(d) > 800:
            u[f[0]] = d.close
    return u


def summary(E: pd.DataFrame, label: str):
    d = E[E.death & E.完整]
    f = lambda s: f"{s.mean() * 100:3.0f}%"  # noqa: E731
    print(f"\n== {label}：月线死叉 {len(d)} 次（{d.月.nunique()} 个不同月份）==")
    print(f"  死叉时离前高已经跌了：中位 {d.距前高.median():+.0%}")
    print(f"  之后 12 个月：回到前高 {f(d.回前高)}，一年后涨 ≥20% {f(d.r12 >= 0.2)}，一年后下跌 {f(d.r12 < 0)}，"
          f"6 个月内重新金叉且之后回前高 {f(d.金叉后回前高)}")
    print(f"  之后 12 个月最多再跌：中位 {d.再跌.median():+.0%}；最大反弹：中位 {d.最大反弹.median():+.0%}")
    rows = []
    for k, g in (("全部月份", E), ("处在死叉状态（DIF<DEA）", E[E.dead]), ("处在金叉状态（DIF≥DEA）", E[~E.dead]),
                 ("死叉当月", E[E.death]), ("金叉当月", E[E.golden])):
        rows.append({"": k, "样本": len(g), **{f"{h}个月后上涨": f(g[f"r{h}"].dropna() > 0) for h in (3, 6, 12)},
                     **{f"{h}个月中位": f"{g[f'r{h}'].median() * 100:+.1f}%" for h in (3, 6, 12)}})
    print(pd.DataFrame(rows).set_index("").to_string())


def main():
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)
    U = universe()
    E = pd.concat([events(n, c) for n, c in U.items()], ignore_index=True)
    print("== 上证指数历次月线死叉 ==")
    sh = E[(E.指数 == "上证指数") & E.death]
    cols = ["月", "距前高", "再跌", "最大反弹", "r6", "r12", "回前高", "几个月后金叉", "金叉后回前高"]
    print(sh[cols].to_string(index=False, formatters={k: "{:+.0%}".format for k in ("距前高", "再跌", "最大反弹", "r6", "r12")}))
    summary(E, "7 个宽基")
    I = industries()
    EI = pd.concat([events(n, c) for n, c in I.items()], ignore_index=True)
    summary(EI, f"{len(I)} 个通达信行业指数（2013 年起）")
    rev = EI[EI.death & EI.完整 & EI.回前高]
    print("\n  行业指数死叉后一年内回到前高的例子（按月份数）：")
    print("  " + "，".join(f"{k} {v} 个" for k, v in rev.月.value_counts().sort_index().items()))
    now = {n: macd_month(c).iloc[-1] for n, c in {**U, **I}.items()}
    dead_now = [n for n, r in now.items() if r.dead]
    print(f"\n现在（{max(c.index[-1] for c in U.values()).date()}，按当月至今收盘）处在月线死叉状态："
          f"{len(dead_now)}/{len(now)}；宽基里：{'、'.join(n for n in U if now[n].dead) or '无'}")


if __name__ == "__main__":
    main()
