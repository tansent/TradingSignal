"""三种仓位方法的最终回测：8 个指数（科创50 从 2021 年起），全区间和前后两段，外加净值图。"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from strategy_backtest import load_market, run, signals, stats, strategies  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IDX = [("上证50", "sh000016"), ("沪深300", "sh000300"), ("中证500", "sh000905"), ("中证1000", "sh000852"),
       ("上证指数", "sh000001"), ("深证成指", "sz399001"), ("创业板指", "sz399006"), ("科创50", "sh000688")]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def table(S, strat, a, b):
    rows = []
    for name, s0 in S.items():
        s = s0[(s0.index >= max(a, "2021-01-04" if name == "科创50" else a)) & (s0.index <= b)]
        if len(s) < 400:
            continue
        rows.append({"指数": name, "方法": "买入持有", **stats(s.close / s.close.iloc[0], pd.Series(1.0, index=s.index), 1)})
        for k, (bs, ss, ex) in strat.items():
            r = run(s, bs, ss, ex)
            rows.append({"指数": name, "方法": k, **stats(r["equity"], r["pos"], int(r.trades.iloc[0]))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    mkt = load_market()
    S = {name: signals(code, mkt, 200.0) for name, code in IDX}
    strat = strategies(etf_ok=True)
    order = ["买入持有"] + list(strat)
    for a, b in (("2011-08-01", "2026-12-31"), ("2011-08-01", "2018-12-31"), ("2019-01-01", "2026-12-31")):
        T = table(S, strat, a, b)
        avg = T.groupby("方法")[["年化收益", "最大回撤", "平均仓位", "买卖次数"]].mean().reindex(order)
        print(f"\n===== {a[:4]}–{b[:4]}：{T['指数'].nunique()} 个指数平均 =====")
        print(avg.assign(年化收益=lambda d: d["年化收益"].map("{:+.1%}".format), 最大回撤=lambda d: d["最大回撤"].map("{:.1%}".format),
                         平均仓位=lambda d: d["平均仓位"].map("{:.0%}".format), 买卖次数=lambda d: d["买卖次数"].round(0)).to_string())
        if a == "2011-08-01" and b.startswith("2026"):
            for col in ("年化收益", "最大回撤"):
                print(f"\n-- 各指数{col}（%）")
                print(T.pivot(index="指数", columns="方法", values=col).reindex(columns=order, index=[n for n, _ in IDX]).mul(100).round(1).to_string())
    # 净值图：沪深300、创业板指
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = {"买入持有": "#888888", "A 对称（买1/3、卖1/3）": "#4C78A8", "B 买重卖轻（买1/2、卖1/5）": "#F58518",
              "C0 分档（不设离场）": "#B279A2", "C 分档 + 破月BBI离场": "#E45756"}
    for ax, name in zip(axes, ("沪深300", "创业板指")):
        s = S[name][S[name].index >= "2011-08-01"]
        ax.plot(s.index, s.close / s.close.iloc[0], color=colors["买入持有"], lw=1.2, label="买入持有")
        for k, (bs, ss, ex) in strat.items():
            r = run(s, bs, ss, ex)
            ax.plot(r.index, r["equity"], color=colors[k], lw=1.4 if k.startswith("C ") else 1.0, label=k)
        ax.set_title(f"{name}：各方法净值（2011-08 起，起点 = 1）")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = ROOT / "reports" / "strategy_backtest.png"
    fig.savefig(out, dpi=130)
    print("\n图：", out)
