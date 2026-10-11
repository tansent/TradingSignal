"""画情绪温度 v1：上证指数 + 中线温度 + 短线温度（共用时间轴的三个小图，不用双坐标轴）。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from tdx_reader import read_day  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False,
                     "font.size": 10, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
COLD, HOT, LINE = "#2a78d6", "#e34948", "#3d3c39"
COLD_BG, HOT_BG = "#dbe8f8", "#fadede"

EVENTS = {"2015-06-12": "2015顶", "2016-01-28": "熔断底", "2018-10-18": "2018底", "2019-04-08": "2019顶",
          "2021-02-18": "2021顶", "2022-04-26": "2022底", "2024-02-05": "2024.2底", "2024-10-08": "924后顶",
          "2025-04-07": "关税底", "2026-06-26": "顶部信号", "2026-10-09": "底部信号"}


def spans(mask: pd.Series):
    """把连续为 True 的日期段转成 (起, 止) 列表。"""
    out, start, prev = [], None, None
    for d, v in mask.items():
        if v and start is None:
            start = d
        if not v and start is not None:
            out.append((start, prev)); start = None
        prev = d
    if start is not None:
        out.append((start, prev))
    return out


def draw(t: pd.DataFrame, idx: pd.Series, start: str, path: Path, title: str, label_events: bool):
    t = t.loc[start:]; idx = idx.loc[start:]
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3, 1.3]})
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE); ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax0, ax1, ax2 = axes
    mid = t["中线温度"]
    for a, b in spans((mid < 20).reindex(idx.index, fill_value=False)):
        ax0.axvspan(a, b, color=COLD_BG, lw=0)
    for a, b in spans((mid >= 80).reindex(idx.index, fill_value=False)):
        ax0.axvspan(a, b, color=HOT_BG, lw=0)
    ax0.plot(idx.index, idx.values, color=INK, lw=1.4)
    ax0.set_yscale("log")
    lo, hi = idx.min(), idx.max()
    ticks = [v for v in (2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500) if lo * 0.9 <= v <= hi * 1.1]
    ax0.set_yticks(ticks); ax0.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax0.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax0.set_ylim(lo * 0.95, hi * 1.12)
    ax0.set_title("上证指数（背景：蓝 = 中线温度 < 20 恐慌区，红 = 中线温度 ≥ 80 过热区）", loc="left", color=INK, fontsize=11)
    for ax, col, name in ((ax1, "中线温度", "中线温度（约 1 个月视角）"), (ax2, "短线温度_平滑", "短线温度（逐日，3 日平滑）")):
        s = t[col]
        ax.axhspan(0, 20, color=COLD_BG, lw=0); ax.axhspan(80, 100, color=HOT_BG, lw=0)
        ax.plot(s.index, s.values, color=LINE, lw=1.1)
        ax.set_ylim(0, 100); ax.set_yticks([0, 20, 50, 80, 100])
        ax.set_title(name, loc="left", color=INK, fontsize=11)
        last = s.dropna()
        ax.annotate(f"{last.iloc[-1]:.0f}", (last.index[-1], last.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=10, fontweight="bold")
    for d, name in EVENTS.items():
        d = pd.Timestamp(d)
        if d < idx.index[0] or d not in idx.index:
            continue
        if label_events or name in ("顶部信号", "底部信号"):
            ax0.annotate(name, (d, idx[d]), xytext=(0, 12 if "顶" in name else -20), textcoords="offset points",
                         ha="center", fontsize=8.5, color=INK2,
                         arrowprops=dict(arrowstyle="-", color="#a9a8a2", lw=0.8))
    ax2.xaxis.set_major_locator(mdates.YearLocator() if label_events else mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y" if label_events else "%Y-%m"))
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    t = pd.read_parquet(ROOT / "data" / "temperature_v1.parquet")
    idx = read_day("D:/TDXapp/vipdoc/sh/lday/sh000001.day")["close"]
    draw(t, idx, "2012-01-01", ROOT / "reports" / "temperature_v1_full.png", "A股情绪温度 v1：2012–2026", True)
    draw(t, idx, "2025-03-01", ROOT / "reports" / "temperature_v1_recent.png", "A股情绪温度 v1：最近 19 个月", False)
    print("saved")
