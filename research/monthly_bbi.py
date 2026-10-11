"""月线 BBI（MA3+MA6+MA12+MA24 的平均，按月收盘价计算），换算到每个交易日、不用未来数据。

每个交易日把当天收盘价当作"本月收盘"，前面各月用已走完月份的月末收盘，
这和通达信月线图上当月 K 线还没走完时显示的 BBI 一样，也和公式 DINGDI 里的写法一致。
"""
import pandas as pd


def asof_bbi(c: pd.Series) -> pd.Series:
    """每个交易日的月线 BBI。"""
    month = c.index.to_period("M")
    m = c.groupby(month).last()                       # 每月最后一个交易日收盘（当月为截至今天）
    prev = m.shift(1)                                 # 上一个已走完月份
    out = 0
    for n in (3, 6, 12, 24):
        s = prev.rolling(n - 1).sum() if n > 1 else 0  # 前 n-1 个已走完月份的月收盘之和
        out = out + (c + s.reindex(month).to_numpy()) / n
    return out / 4


def month_end_bbi(c: pd.Series) -> pd.DataFrame:
    """月末确认版：已走完月份的月收盘与月 BBI。"""
    m = c.resample("ME").last().dropna()
    b = (m.rolling(3).mean() + m.rolling(6).mean() + m.rolling(12).mean() + m.rolling(24).mean()) / 4
    return pd.DataFrame({"close": m, "bbi": b})
