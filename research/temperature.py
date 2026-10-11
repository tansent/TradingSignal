"""情绪温度 v1：0（极度恐慌）~ 100（极度贪婪）。

每个分项都换算成"相对自己过去 N 个交易日的百分位"（只看过去，不用未来数据），再等权平均。
- 短线温度（窗口 500 日）：上涨占比、净涨停（涨停-跌停）、炸板率（反向）、最高连板、晋级率、
  昨日涨停今日表现、60日新高-新低、站上20日线比例、成交额（相对年均）。
- 中线温度（窗口 750 日）：20日平均上涨占比、站上60日线比例、20日平均新高-新低、
  20日成交额（相对年均）、沪市融资余额20日变化、融资买入额占沪市成交比例。
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def pct_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).rank(pct=True)


def components() -> tuple[pd.DataFrame, pd.DataFrame]:
    b = pd.read_parquet(DATA / "breadth_daily.parquet")
    a = b[b.board == "all"].set_index("date").sort_index()
    sh_amt = b[b.board.isin(["sh_main", "kcb"])].groupby("date")["amount"].sum()
    n = a["n"]
    short = pd.DataFrame({
        "上涨占比": a.up / n,
        "净涨停": (a.zt - a.dt) / n,
        "炸板率(反向)": -(a.zb / (a.zt + a.zb).replace(0, np.nan)),
        "最高连板": a.max_lb.astype(float),
        "晋级率": (a.promote / a.prev_zt).where(a.prev_zt >= 5),
        "昨日涨停今日表现": a.prev_zt_ret,
        "新高-新低": (a.nh60 - a.nl60) / n,
        "站上20日线": a.above20 / n,
        "成交额(相对年均)": a.amount / a.amount.rolling(250, min_periods=120).mean(),
    })
    m = pd.read_csv(DATA / "margin_sse.csv", parse_dates=["date"]).set_index("date")
    m = m.reindex(a.index).ffill()
    mid = pd.DataFrame({
        "20日上涨占比": short["上涨占比"].rolling(20).mean(),
        "站上60日线": a.above60 / n,
        "20日新高-新低": short["新高-新低"].rolling(20).mean(),
        "20日成交额(相对年均)": a.amount.rolling(20).mean() / a.amount.rolling(250, min_periods=120).mean(),
        "融资余额20日变化": m["融资余额"].pct_change(20),
        "融资买入占比": (m["融资买入额"] / sh_amt).rolling(5).mean(),
    })
    return short, mid


def temperature() -> pd.DataFrame:
    short, mid = components()
    ps = pd.concat({k: pct_rank(v, 500) for k, v in short.items()}, axis=1)
    pm = pd.concat({k: pct_rank(v, 750) for k, v in mid.items()}, axis=1)
    t = pd.DataFrame({
        "短线温度": ps.mean(axis=1, skipna=True) * 100,
        "中线温度": pm.mean(axis=1, skipna=True) * 100,
    })
    t["短线温度_平滑"] = t["短线温度"].ewm(span=3, adjust=False).mean()
    # 9 项平均后会被压缩到中间，再换算成"在过去 500 个交易日中的分位"，让 0~100 拉开
    t["短线温度_分位"] = (t["短线温度"].rolling(500, min_periods=250).rank(pct=True) * 100).ewm(span=3, adjust=False).mean()
    return t.join(ps.add_prefix("s_") * 100).join(pm.add_prefix("m_") * 100)


if __name__ == "__main__":
    t = temperature()
    t.to_parquet(DATA / "temperature_v1.parquet")
    pd.set_option("display.width", 200)
    print(t[["短线温度", "短线温度_平滑", "中线温度"]].dropna().tail(10).round(1))
