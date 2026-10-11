"""检验：月线 BBI 向上（且月收盘在月 BBI 之上）时，日线收盘跌破日 BBI 买入，是否优于其他时候买入。
BBI = (MA3 + MA6 + MA12 + MA24) / 4。月线用已走完的上一个月，避免未来数据。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lib import load  # noqa: E402

INDEXES = {"上证指数": "sh000001", "沪深300": "sh000300", "深证成指": "sz399001", "创业板指": "sz399006",
           "中证500": "sh000905", "中证1000": "sh000852"}


def bbi(c: pd.Series) -> pd.Series:
    return (c.rolling(3).mean() + c.rolling(6).mean() + c.rolling(12).mean() + c.rolling(24).mean()) / 4


def analyze(df: pd.DataFrame):
    c, l = df.close, df.low
    m = c.resample("ME").last()
    mb = bbi(m)
    up_m = (mb > mb.shift(1)) & (m > mb)                       # 月 BBI 向上且月收盘在其上方
    # 每个交易日只能看到上一个已完成月份的状态
    regime = up_m.shift(1).reindex(c.index, method="ffill").fillna(False).astype(bool)
    db = bbi(c)
    trigger = (c < db) & (c.shift(1) >= db.shift(1))           # 日线收盘首次跌破日 BBI
    out = pd.DataFrame({"regime": regime, "trigger": trigger})
    for n in (20, 60, 120):
        out[f"r{n}"] = c.shift(-n) / c - 1
    out["dd60"] = l[::-1].rolling(60, min_periods=1).min()[::-1].shift(-1) / c - 1
    return out.dropna(subset=["r120"])


if __name__ == "__main__":
    rows = []
    for name, code in INDEXES.items():
        a = analyze(load(code))
        a = a[a.index >= "2005-01-01"]
        a["index"] = name
        rows.append(a)
    A = pd.concat(rows)
    groups = {
        "全部交易日": A,
        "月BBI向上的所有交易日": A[A.regime],
        "月BBI向上 + 日线跌破BBI（你的方法）": A[A.regime & A.trigger],
        "月BBI不向上 + 日线跌破BBI（对照）": A[~A.regime & A.trigger],
    }
    print(f"{'情形':<30}{'样本':>6}{'20日均':>8}{'20日胜':>7}{'60日均':>8}{'60日胜':>7}{'120日均':>8}{'120日胜':>7}{'60日内最大跌中位':>12}")
    for k, g in groups.items():
        print(f"{k:<30}{len(g):>6}{g.r20.mean():>+8.1%}{(g.r20 > 0).mean():>7.0%}{g.r60.mean():>+8.1%}{(g.r60 > 0).mean():>7.0%}"
              f"{g.r120.mean():>+8.1%}{(g.r120 > 0).mean():>7.0%}{g.dd60.median():>+12.1%}")
    hit = A[A.regime & A.trigger]
    print("\n你的方法：按年份（6个指数合计次数 / 60日平均收益）")
    print(hit.groupby(hit.index.year).agg(n=("r60", "size"), r60=("r60", "mean")).T.round(3).to_string())
    worst = hit.nsmallest(8, "r60")
    print("\n最差的 8 次：", "; ".join(f"{d.date()} {i} {r:+.0%}" for d, i, r in zip(worst.index, worst["index"], worst.r60)))
