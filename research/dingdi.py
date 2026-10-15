"""与通达信公式 DINGDI / QXZB 完全相同的计算（formulas/通达信粘贴用/）。

输入：通达信指数日线（含 adv/dec 涨跌家数列）。输出：情绪、★S / ▽S 顶部、△B / ★B 底部、月线 BBI 与跌破信号。
"""
import numpy as np
import pandas as pd

from monthly_bbi import asof_bbi


def _sma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1 / n, adjust=False).mean()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    C, O, H, L = df.close, df.open, df.high, df.low
    d = C - C.shift()
    up5, dn5 = _sma(d.clip(lower=0), 5), _sma(d.abs(), 5)
    rsv9 = (C - L.rolling(9).min()) / (H.rolling(9).max() - L.rolling(9).min()) * 100
    k9 = _sma(rsv9, 3)
    qx = (-11.59 + 0.885 * up5 / dn5 * 100 + 0.317 * k9).clip(0, 100)

    top = ((C >= H.rolling(250).max() * 0.92) & (C > C.rolling(250).mean()) & (qx.rolling(5).max() >= 84)
           & (C / O - 1 <= -0.01) & ((H - np.maximum(O, C)) / C.shift() >= 0.004)
           & ((C < C.rolling(5).mean()) | (C < (O.shift() + C.shift()) / 2)))

    # ▽S 一级顶部：情绪 5 日内曾 ≥93（过热），今天向下穿过 70，且仍在一年高点 5% 以内（与 ★S 同日时只算 ★S）
    top1 = (qx < 70) & (qx.shift() >= 70) & (qx.rolling(5).max() >= 93) & (C >= H.rolling(250).max() * 0.95) & ~top.fillna(False)

    rsi5l = (up5.shift() * 4 + (L - C.shift()).clip(lower=0)) / (dn5.shift() * 4 + (L - C.shift()).abs()) * 100
    rsv9l = (L - L.rolling(9).min()) / (H.rolling(9).max() - L.rolling(9).min()) * 100
    k9l = (k9.shift() * 2 + rsv9l) / 3
    qxl = (-11.59 + 0.885 * rsi5l + 0.317 * k9l).clip(0, 100)
    sig = (C / C.shift() - 1).rolling(250).std().shift()
    t1 = (qxl <= 10) & ((L / H.rolling(120).max() - 1) / sig <= -11)
    if "adv" in df and "dec" in df:
        panic = df.dec / (df.adv + df.dec + 0.0001) >= 0.92
    else:
        panic = pd.Series(False, index=df.index)
    since_high = H.rolling(120).apply(lambda x: len(x) - 1 - np.where(x >= x.max())[0][-1], raw=True)
    t2 = t1 & panic & (since_high >= 30)
    # 乖S 乖离顶（2026-10-16）：暴涨后天量收阴。20 日乖离 ≥ 6 倍日波动率，成交额 ≥ 前一年最大值的 90%，收盘比开盘低 ≥1%。
    # ★S 要求上影线和跌破 5 日线，抓不到"高开在最高点、一路走低"的顶（2024-10-08）。2010 年以来 30 次，之后 60 天下跌的比例 83%、中位 -4.3%；
    # 2014-12-09 那次之后继续大涨（和 ◆S 一样，那是牛市主升段）。
    if "amount" in df:
        glz = (C / C.rolling(20).mean() - 1) / sig
        gl = (glz >= 6) & (df.amount >= df.amount.rolling(250).max().shift() * 0.9) & (C / O - 1 <= -0.01)
    else:
        gl = pd.Series(False, index=df.index)
    # 破月BBI：收盘连续 2 天在月线 BBI 下方（第 2 天标记；之后要先回到上方才会再标）
    mb = asof_bbi(C)
    below = C < mb
    b2 = below & below.shift(1, fill_value=False)
    brk = b2 & ~b2.shift(1, fill_value=False)
    return pd.DataFrame({
        "情绪": qx, "情绪_盘中低": qxl,
        "顶部": top.fillna(False), "顶部一级": top1.fillna(False), "乖离顶": gl.fillna(False),
        "底部一级": (t1 & ~t2).fillna(False), "底部二级": t2.fillna(False),
        "月线BBI": mb, "破月BBI": brk,
    })
