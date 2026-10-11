"""上证指数 + 过热分数（融资买入占比、成交额相对年均、融资余额60日增速 的 3 年分位平均）。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from overheat_eval import heat  # noqa: E402
from plot_temperature import spans  # noqa: E402
from tdx_reader import read_day  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False, "font.size": 10})
h = heat()
c = read_day("D:/TDXapp/vipdoc/sh/lday/sh000001.day").close.loc["2014-01-01":]
H = h.H.reindex(c.index)
fig, (a0, a1) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
fig.patch.set_facecolor("#fcfcfb")
for ax in (a0, a1):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e7e6e2"); ax.set_axisbelow(True)
for a, b in spans((H >= 0.9).fillna(False)):
    a0.axvspan(a, b, color="#fadede", lw=0)
a0.plot(c.index, c.values, color="#0b0b0b", lw=1.2)
a0.set_yscale("log")
a0.set_yticks([2000, 2500, 3000, 3500, 4000, 5000]); a0.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
a0.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a0.set_title("上证指数（红色背景 = 过热分数 ≥ 0.9，融资和成交额都处于近 3 年最热的区间）", loc="left", fontsize=11)
a1.axhspan(0.9, 1.0, color="#fadede", lw=0)
a1.plot(H.index, H.values, color="#3d3c39", lw=1)
a1.set_ylim(0, 1); a1.set_yticks([0, 0.5, 0.9, 1])
a1.set_title("过热分数（成交额相对年均、融资买入占成交额、融资余额 60 日增速，三者的近 3 年分位平均）", loc="left", fontsize=11)
a1.annotate(f"{H.dropna().iloc[-1]:.2f}", (H.dropna().index[-1], H.dropna().iloc[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontweight="bold")
a1.xaxis.set_major_locator(mdates.YearLocator()); a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.suptitle("融资 + 成交额过热分数：2014–2026", x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(ROOT / "reports" / "overheat.png", dpi=125, facecolor="#fcfcfb")
print("saved")
