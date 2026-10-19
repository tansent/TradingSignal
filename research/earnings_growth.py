"""盈利增速和它的变化（"二阶导"）对买卖点的影响（2026-10-18）。

每个指数的滚动盈利 = 收盘价 ÷ 滚动市盈率（research/valuation.py 的数据，都是当时已公布的盈利），
  增速 g = 滚动盈利同比（250 个交易日）；加速度 a = g − 63 个交易日前的 g（增速比 3 个月前快还是慢）。
指数：上证50、沪深300、中证500、中证1000、上证指数（中证官网日度，2011 年起，增速从 2012 年起）；
  深证成指、创业板指（蛋卷周度，2016 年起）。科创50 只有市净率，不参加。
看三件事：所有交易日、B 点（△B/★B）、S 点（★S/▽S/乖S）按增速方向分组后，之后 20/60/250 天的涨跌。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dingdi  # noqa: E402
import valuation  # noqa: E402
from tdx_reader import read_day  # noqa: E402

IDX = [("上证50", "sh000016"), ("沪深300", "sh000300"), ("中证500", "sh000905"), ("中证1000", "sh000852"),
       ("上证指数", "sh000001"), ("深证成指", "sz399001"), ("创业板指", "sz399006")]
H = (20, 60, 250)


def panel() -> pd.DataFrame:
    parts = []
    for name, code in IDX:
        df = read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day")
        pe, metric = valuation.series(name)
        if metric != "市盈率":
            continue
        c = df.close
        e = (c.reindex(pe.index, method="ffill") / pe).reindex(c.index, method="ffill")
        e = e[e.index >= pe.index[0]]
        g = e / e.shift(250) - 1
        a = g - g.shift(63)
        r = dingdi.compute(df)
        x = pd.DataFrame({"指数": name, "g": g, "a": a,
                          "B": (r["底部一级"] | r["底部二级"]).reindex(e.index),
                          "S": (r["顶部"] | r["顶部一级"] | r["乖离顶"]).reindex(e.index)})
        for h in H:
            x[f"f{h}"] = (c.shift(-h) / c - 1).reindex(e.index)
        parts.append(x.dropna(subset=["g", "a"]))
    return pd.concat(parts)


def first_in_round(x: pd.DataFrame, col: str, gap: int = 20) -> pd.DataFrame:
    out = []
    for _, g in x.groupby("指数"):
        last = None
        for d in g.index[g[col].fillna(False).astype(bool)]:
            if last is None or (d - last).days > gap * 1.4:
                out.append(g.loc[[d]])
            last = d
    return pd.concat(out) if out else x.iloc[:0]


def table(x: pd.DataFrame, up: bool = True) -> pd.DataFrame:
    groups = {"增速加快": x.a > 0, "增速放缓": x.a <= 0,
              "加快且增速 > 0": (x.a > 0) & (x.g > 0), "放缓且增速 > 0（二阶导转负）": (x.a <= 0) & (x.g > 0),
              "增速 < 0": x.g <= 0}
    rows = {}
    for k, m in groups.items():
        y = x[m]
        row = {"次数": len(y)}
        for h in H:
            v = y[f"f{h}"].dropna()
            row[f"{h}天{'上涨' if up else '下跌'}"] = ((v > 0) if up else (v < 0)).mean() if len(v) else np.nan
            row[f"{h}天中位"] = v.median() if len(v) else np.nan
        rows[k] = row
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    P = panel()
    fmt = lambda t: t.apply(lambda col: col.map(lambda v: f"{v:.0f}" if col.name == "次数" else f"{v:+.1%}" if "中位" in col.name else f"{v:.0%}"))  # noqa: E731
    print("== 所有交易日（指数 × 日）==")
    print(fmt(table(P)).to_string())
    B = first_in_round(P, "B")
    print("\n== B 点（△B/★B，每轮第一个）==")
    print(fmt(table(B)).to_string())
    S = first_in_round(P, "S")
    print("\n== S 点（★S/▽S/乖S，每轮第一个），看下跌比例 ==")
    print(fmt(table(S, up=False)).to_string())
    for lo, hi in (("2011", "2018-12-31"), ("2019", "2026-12-31")):
        b = B[(B.index >= lo) & (B.index <= hi)]
        print(f"\nB 点 {lo}–{hi[:4]}：")
        print(fmt(table(b)).iloc[:2].to_string())
    now = P.groupby("指数").tail(1)[["指数", "g", "a"]]
    print("\n现在：")
    print(now.assign(g=lambda d: d.g.map("{:+.0%}".format), a=lambda d: d.a.map("{:+.0%}".format)).to_string(index=False))
