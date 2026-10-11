"""底部信号规则回测。

用法：python research/backtest.py [close|intraday]
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research")
from lib import NAMES, features, forward_stats, load  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 500)

INDEXES = ["sh000001", "sh000300", "sz399001", "sz399006", "sh000905", "sh000852", "sh000688"]
START = "2005-01-01"
COOLDOWN = 20  # 信号后 N 根内不重复


def first_trigger(cond: pd.Series, cooldown: int = COOLDOWN) -> pd.Series:
    """条件成立，且之前 cooldown 根内没有发出过信号。"""
    out = np.zeros(len(cond), dtype=bool)
    last = -10 ** 9
    for i, v in enumerate(cond.fillna(False).to_numpy()):
        if v and i - last > cooldown:
            out[i] = True
            last = i
    return pd.Series(out, index=cond.index)


def rule_bottom(f: pd.DataFrame, k: dict) -> pd.Series:
    """多周期同时极端超卖：短（5日）、中（20日）、长（60日/距高点回撤）、RSI。"""
    conds = [
        f["z5"] <= k["z5"],
        f["z20"] <= k["z20"],
        f["zbias60"] <= k["zbias60"],
        f["z_dd120"] <= k["zdd"],
        f["rsi14"] <= k["rsi14"],
    ]
    hit = sum(c.astype(int) for c in conds)
    return hit >= k.get("need", len(conds))


DEFAULT = dict(z5=-1.5, z20=-1.3, zbias60=-0.85, zdd=-12, rsi14=30)


def run(mode: str, k: dict, verbose: bool = True):
    all_rows = []
    for code in INDEXES:
        df = load(code)
        f = features(df, mode)
        sig = first_trigger(rule_bottom(f, k))
        sig = sig[sig.index >= START]
        dates = sig[sig].index
        st = forward_stats(df, dates)
        st["index"] = NAMES[code]
        st["regime"] = f.loc[dates, "regime"].to_numpy()
        all_rows.append(st)
    res = pd.concat(all_rows, ignore_index=True)
    return res


def baseline(mode: str):
    rows = []
    for code in INDEXES:
        df = load(code)
        d = df.index[df.index >= START]
        st = forward_stats(df, d[::5])  # 每 5 天取一个样本
        rows.append(st)
    b = pd.concat(rows)
    return b


def summarize(res: pd.DataFrame, label: str = ""):
    years = {c: (load(c).index[-1] - max(load(c).index[0], pd.Timestamp(START))).days / 365 for c in INDEXES}
    g = res.groupby("index")
    print(f"\n--- {label} 信号数 {len(res)} ---")
    for h in (5, 20, 60):
        r = res[f"ret{h}"].dropna()
        print(f"  {h:>2}日后: 平均 {r.mean():+.2%}  中位 {r.median():+.2%}  上涨概率 {(r > 0).mean():.0%}  "
              f"期间最大浮亏(中位) {res[f'mae{h}'].median():+.2%}")
    print("  按环境:", {k: f"{len(v)}次 20日胜率{(v['ret20'] > 0).mean():.0%} 均值{v['ret20'].mean():+.1%}"
                         for k, v in res.groupby("regime")})


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "close"
    b = baseline(mode)
    print("=== 基准（任意一天买入）===")
    for h in (5, 20, 60):
        r = b[f"ret{h}"].dropna()
        print(f"  {h:>2}日后: 平均 {r.mean():+.2%}  上涨概率 {(r > 0).mean():.0%}")
    res = run(mode, DEFAULT)
    summarize(res, f"{mode} 默认参数")
    print(res[["index", "date", "regime", "ret5", "ret20", "ret60", "mae20", "mae60"]]
          .assign(date=lambda x: x["date"].dt.date).to_string(index=False, float_format=lambda v: f"{v:+.1%}"))
