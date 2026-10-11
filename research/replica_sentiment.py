"""复刻他的分指数情绪指标：情绪 = a + b×RSI5 + c×K(9,3,3)，截断到 0~100。系数用他 2025-07~2026-06 的读数拟合。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from tdx_reader import read_day  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V = "D:/TDXapp/vipdoc/sh/lday/"
IDX = {"上证50": "sh000016", "沪深300": "sh000300", "中证500": "sh000905", "中证1000": "sh000852"}


def rsi(c, n):
    d = c.diff()
    return d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean() / d.abs().ewm(alpha=1 / n, adjust=False).mean() * 100


def kdj_k(df, n=9):
    ll, hh = df.low.rolling(n).min(), df.high.rolling(n).max()
    return ((df.close - ll) / (hh - ll) * 100).ewm(alpha=1 / 3, adjust=False).mean()


def features(code):
    d = read_day(V + code + ".day")
    return pd.DataFrame({"RSI5": rsi(d.close, 5), "K": kdj_k(d)})


def fit():
    his = pd.read_csv(ROOT / "data" / "his_sentiment_digitized.csv", index_col=0, parse_dates=True)
    rows = [features(code).join(his[k].rename("y"), how="inner").dropna() for k, code in IDX.items()]
    tr = pd.concat(rows)
    X = np.c_[np.ones(len(tr)), tr[["RSI5", "K"]].to_numpy()]
    b, *_ = np.linalg.lstsq(X, tr.y.to_numpy(), rcond=None)
    return b, his


def replica(code, b):
    f = features(code)
    return (b[0] + b[1] * f.RSI5 + b[2] * f.K).clip(0, 100)


if __name__ == "__main__":
    b, his = fit()
    print(f"系数：情绪 = {b[0]:.2f} + {b[1]:.3f} × RSI5 + {b[2]:.3f} × K")
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    latest = {}
    for ax, (k, code) in zip(axes, IDX.items()):
        r = replica(code, b)
        latest[k] = r.loc["2026-09-24":].round(0)
        h = his[k]
        j = pd.concat([r.rename("r"), h.rename("h")], axis=1).dropna()
        ax.set_facecolor("#fcfcfb")
        ax.axhline(93, color="#1baf7a", lw=1); ax.axhline(15, color="#1baf7a", lw=1)
        ax.plot(h.index, h.values, color="#e34948", lw=1.6, label="他的读数（从截图读取）")
        rr = r.loc[h.index[0]:]
        ax.plot(rr.index, rr.values, color="#2a78d6", lw=1.2, label="复刻版")
        ax.set_ylim(-3, 108); ax.set_yticks([0, 15, 50, 93])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"{k}情绪指标   相关系数 {j.r.corr(j.h):.3f}，平均误差 {(j.r - j.h).abs().mean():.1f}", loc="left", fontsize=11)
        ax.axvline(pd.Timestamp("2026-06-25"), color="#a9a8a2", lw=0.8, ls="--")
    axes[0].legend(loc="upper left", ncol=2, frameon=False)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator()); axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("他的分指数情绪指标 vs 复刻版（虚线右侧为他未公开的区间）", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ROOT / "reports" / "replica_sentiment.png", dpi=120, facecolor="#fcfcfb")
    print(pd.DataFrame(latest).to_string())
