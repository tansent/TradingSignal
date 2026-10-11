"""画底部信号 v1（一级/二级）在上证指数、沪深300 上的位置，并标出他的底部信号。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from bottom_features import all_features, load_index  # noqa: E402
from bottom_signal_v1 import tiers  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HIS = pd.to_datetime(["2025-04-07", "2026-10-09"])
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False, "font.size": 10})
F = all_features()
panels = [("上证指数", "sh000001", "2015-06-01", "2016-06-30"), ("上证指数", "sh000001", "2018-01-01", "2019-06-30"),
          ("沪深300", "sh000300", "2023-06-01", "2026-10-09"), ("创业板指", "sz399006", "2023-06-01", "2026-10-09")]
fig, axes = plt.subplots(len(panels), 1, figsize=(13, 13))
fig.patch.set_facecolor("#fcfcfb")
for ax, (name, code, a, b) in zip(axes, panels):
    df = load_index(code).loc[a:b]
    f = F[name].loc[a:b]
    t1, t2 = tiers(f)
    t2 = t2 & (f["距120日高天数"] >= 30)
    t1_only = t1 & ~t2
    ax.set_facecolor("#fcfcfb")
    ax.plot(df.index, df.close, color="#0b0b0b", lw=1.2)
    ax.scatter(t1_only[t1_only].index, df.low[t1_only] * 0.985, marker="^", s=28, color="#eda100", zorder=3, label="一级（关注区）")
    ax.scatter(t2[t2].index, df.low[t2] * 0.97, marker="^", s=60, color="#e34948", zorder=4, label="二级（较强）")
    his = [d for d in HIS if pd.Timestamp(a) <= d <= pd.Timestamp(b)]
    if his:
        ax.scatter(his, df.low.loc[his] * 0.945, marker="*", s=140, color="#2a78d6", edgecolor="#0b0b0b", lw=0.5, zorder=5, label="他的底部信号")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e7e6e2"); ax.set_axisbelow(True)
    ax.set_title(f"{name}  {a[:7]} ~ {b[:7]}", loc="left", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[2].legend(loc="upper left", frameon=False, ncol=3)
fig.suptitle("底部信号 v1（按收盘数据回测；盘中最低价近似午间状态）", x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(ROOT / "reports" / "bottom_signal_v1.png", dpi=120, facecolor="#fcfcfb")
print("saved")
