"""顶部预警 v1（R1'）：高位 + 近 5 日情绪曾超买 + 带上影的实体阴线 + 收盘跌破 5 日线或前日实体中点。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from lib import load  # noqa: E402
from replica_sentiment import rsi, kdj_k  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HIS_TOPS = pd.to_datetime(["2015-04-28", "2015-05-28", "2015-06-16", "2026-06-23"])


def top_signal(df: pd.DataFrame) -> pd.Series:
    c, o, h = df.close, df.open, df.high
    senti = (-11.59 + 0.885 * rsi(c, 5) + 0.317 * kdj_k(df)).clip(0, 100)
    high_zone = (c >= 0.92 * h.rolling(250).max()) & (c > c.rolling(250).mean())
    overbought = senti.rolling(5).max() >= 84
    bear_body = c / o - 1 <= -0.010
    upper_shadow = (h - pd.concat([o, c], axis=1).max(axis=1)) / c.shift(1) >= 0.004
    breakdown = (c < c.rolling(5).mean()) | (c < (o.shift(1) + c.shift(1)) / 2)
    return (high_zone & overbought & bear_body & upper_shadow & breakdown).fillna(False)


if __name__ == "__main__":
    df = load("sh000001")
    sig = top_signal(df)
    hits = sig[sig & (sig.index >= "2014-01-01")].index
    c, l, h = df.close, df.low, df.high
    print("2014 年以来触发：")
    for d in hits:
        i = c.index.get_loc(d)
        dd = l.iloc[i + 1:i + 61].min() / c.iloc[i] - 1 if i + 1 < len(c) else float("nan")
        up = h.iloc[i + 1:i + 61].max() / c.iloc[i] - 1 if i + 1 < len(c) else float("nan")
        tag = " ← 他的信号" if d in HIS_TOPS else ""
        print(f"  {d.date()} 收{c[d]:.0f}  之后60日内 最低{dd:+.1%} 最高{up:+.1%}{tag}")
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), gridspec_kw={"height_ratios": [1, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (a, b, title) in zip(axes, [("2014-06-01", "2015-09-30", "2014–2015 牛市"), ("2024-08-01", "2026-10-09", "2024–2026")]):
        s = c.loc[a:b]
        ax.set_facecolor("#fcfcfb")
        ax.plot(s.index, s.values, color="#0b0b0b", lw=1.3)
        mine = [d for d in hits if a <= str(d.date()) <= b]
        ax.scatter(mine, h.loc[mine] * 1.015, marker="v", s=55, color="#e34948", zorder=3, label="顶部预警 v1")
        his = [d for d in HIS_TOPS if pd.Timestamp(a) <= d <= pd.Timestamp(b)]
        ax.scatter(his, h.loc[his] * 1.045, marker="*", s=130, color="#eda100", edgecolor="#0b0b0b", lw=0.5, zorder=4, label="他的顶部信号")
        for sname in ("top", "right"):
            ax.spines[sname].set_visible(False)
        ax.grid(axis="y", color="#e7e6e2"); ax.set_axisbelow(True)
        ax.set_title(f"上证指数 {title}", loc="left", fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("顶部预警 v1 与他的顶部信号", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ROOT / "reports" / "top_signal_v1.png", dpi=125, facecolor="#fcfcfb")
