"""底部信号规则回测：6 个指数，2011-06 起。信号日收盘买入，看 20/60 日表现。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bottom_features import INDEXES, all_features, load_index  # noqa: E402
from backtest import first_trigger  # noqa: E402

START = "2011-06-01"


def rules(f: pd.DataFrame) -> dict[str, pd.Series]:
    base = (f["情绪_盘中低"] <= 10) & (f["距120日高σ_盘中低"] <= -11)
    return {
        "B0 盘中情绪≤10 + 深回撤": base,
        "B1 B0 + 20日急跌(≤-1σ)": base & (f["20日跌幅σ_盘中低"] <= -1.0),
        "B2 B0 + 中线温度≤30": base & (f["中线温度"] <= 30),
        "B3 B0 + 宽度恐慌(下跌≥80%或新低≥30%)": base & ((f["下跌占比"] >= 0.8) | (f["60日新低占比"] >= 0.3)),
        "B4 B0 + 下跌已持续≥40天": base & (f["距120日高天数"] >= 40),
        "B5 B1 + 下跌≥40天": base & (f["20日跌幅σ_盘中低"] <= -1.0) & (f["距120日高天数"] >= 40),
    }


def fwd(df, dates):
    c, h, l = df.close, df.high, df.low
    rows = []
    for d in dates:
        i = c.index.get_loc(d)
        r = {"date": d}
        if i + 60 < len(c):
            r.update(r20=c.iloc[i + 20] / c.iloc[i] - 1, r60=c.iloc[i + 60] / c.iloc[i] - 1,
                     dd60=l.iloc[i + 1:i + 61].min() / c.iloc[i] - 1, up60=h.iloc[i + 1:i + 61].max() / c.iloc[i] - 1)
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    F = all_features()
    res, base = {}, []
    for name, code in INDEXES.items():
        df = load_index(code)
        f = F[name]
        base.append(fwd(df, df.index[df.index >= START][::5]).assign(index=name))
        for rn, cond in rules(f).items():
            s = first_trigger(cond.fillna(False), 20)
            s = s[s.index >= START]
            res.setdefault(rn, []).append(fwd(df, s[s].index).assign(index=name, env=f.loc[s[s].index, "环境"].to_numpy()))
    years = (pd.Timestamp("2026-10-09") - pd.Timestamp(START)).days / 365 * len(INDEXES)

    def line(name, r):
        r = r.dropna(subset=["r60"])
        return (f"{name:<36}{len(r):>4}{len(r) / years:>7.2f}{r.r20.mean():>+8.1%}{(r.r20 > 0).mean():>7.0%}"
                f"{r.r60.mean():>+8.1%}{(r.r60 > 0).mean():>7.0%}{r.dd60.median():>+9.1%}{r.up60.median():>+9.1%}")
    print(f"{'规则':<36}{'次数':>4}{'每指数每年':>7}{'20日均':>8}{'20日胜':>7}{'60日均':>8}{'60日胜':>7}{'60日内最大跌':>9}{'60日内最大涨':>9}")
    print(line("基准（任意一天）", pd.concat(base)))
    for rn, lst in res.items():
        print(line(rn, pd.concat(lst)))
    for rn in ("B1 B0 + 20日急跌(≤-1σ)", "B5 B1 + 下跌≥40天"):
        r = pd.concat(res[rn])
        print(f"\n{rn} 按环境：", {k: f"{len(v)}次 60日胜率{(v.r60.dropna() > 0).mean():.0%} 均{v.r60.mean():+.1%}" for k, v in r.groupby("env")})
        piv = r.assign(d=r.date.dt.date.astype(str)).groupby("d")["index"].apply(lambda x: "/".join(x)).reset_index()
        print("   触发日（指数）:", "; ".join(f"{a}({b})" for a, b in zip(piv.d, piv["index"])))
