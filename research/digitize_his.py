"""把他发的情绪指标截图（Excel 折线图）逐日读成数值。

纵轴用两条绿线定标（93 和 15），横轴按交易日等间距（每 4 个交易日一个刻度）。
每个交易日在对应列附近找红线像素：局部高点取红线上沿，局部低点取下沿，其余取中心。
被字幕遮挡、找不到红线的日子记为空值。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from tdx_reader import read_day  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIRST, LAST = "2025-07-15", "2026-06-25"
GREEN_HI, GREEN_LO = 93.0, 15.0
# (文件, 面板名, 绘图区 x 范围, 面板 y 范围, 上绿线行, 下绿线行)
PANELS = [
    ("6月25日情绪指标上证50沪深300.jpg", "上证50", (379, 2029), (15, 455), 116.0, 396.5),
    ("6月25日情绪指标上证50沪深300.jpg", "沪深300", (379, 2029), (572, 1022), 676.0, 962.0),
    ("6月25日情绪指标中证500中证1000.jpg", "中证500", (414, 1994), (18, 452), 118.0, 395.5),
    ("6月25日情绪指标中证500中证1000.jpg", "中证1000", (414, 1994), (580, 1019), 682.0, 961.0),
]


def trading_days() -> pd.DatetimeIndex:
    d = read_day("D:/TDXapp/vipdoc/sh/lday/sh000001.day").index
    return d[(d >= FIRST) & (d <= LAST)]


def digitize(path, x_rng, y_rng, g_hi, g_lo, days):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    red = (R > 150) & (G < 110) & (B < 110) & (R - G > 60)
    x0, x1 = x_rng
    y0, y1 = y_rng
    sub = red[y0:y1, x0:x1]
    cols = np.where(sub.any(axis=0))[0]
    xf, xl = cols.min() + x0, cols.max() + x0
    n = len(days)
    to_val = lambda y: GREEN_LO + (g_lo - y) * (GREEN_HI - GREEN_LO) / (g_lo - g_hi)

    def col_pixels(x):
        ys = np.where(red[y0:y1, x])[0] + y0
        return ys

    def center(x):
        ys = col_pixels(int(round(x)))
        return ys.mean() if len(ys) else np.nan

    step = (xl - xf) / (n - 1)
    vals = []
    for i in range(n):
        x = xf + i * step
        xs = [int(round(x)) + k for k in (-1, 0, 1)]
        ys = np.concatenate([col_pixels(c) for c in xs if x0 <= c < x1])
        if len(ys) == 0:
            vals.append(np.nan)
            continue
        left = center(x - 0.45 * step) if i > 0 else np.nan
        right = center(x + 0.45 * step) if i < n - 1 else np.nan
        top, bot, mid = ys.min(), ys.max(), np.median(ys)
        nb = [v for v in (left, right) if not np.isnan(v)]
        if nb and all(v > mid + 2 for v in nb):      # 两侧都更低（y 更大）→ 局部高点
            y = top + 1.5
        elif nb and all(v < mid - 2 for v in nb):    # 两侧都更高 → 局部低点
            y = bot - 1.5
        else:
            y = mid
        vals.append(to_val(y))
    return pd.Series(vals, index=days), (xf, xl, step)


if __name__ == "__main__":
    days = trading_days()
    print("交易日数", len(days), days[0].date(), "->", days[-1].date())
    out = {}
    for fn, name, xr, yr, gh, gl in PANELS:
        s, info = digitize(ROOT / "0625data" / fn, xr, yr, gh, gl, days)
        out[name] = s
        print(f"{name}: x {info[0]}..{info[1]} step {info[2]:.2f}px | 缺失 {s.isna().sum()} 天 | 最后一天 {s.iloc[-1]:.1f}")
    df = pd.DataFrame(out).round(1)
    df.to_csv(ROOT / "data" / "his_sentiment_digitized.csv")
    pd.set_option("display.width", 200)
    for d in ["2025-11-24", "2026-01-06", "2026-03-20", "2026-05-11", "2026-06-08", "2026-06-18", "2026-06-25"]:
        print(d, df.loc[d].to_dict())
