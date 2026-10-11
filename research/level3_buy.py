"""三级买入信号研究：★B 之后的几天内，再叠加 中线温度 ≤10 和/或 国家队宽基 ETF 大额净申购。

每次 ★B 算一轮（与上一次 ★B 间隔 ≥20 个交易日才算新一轮），在这一轮的 ★B 当天到之后 5 个交易日内，
条件第一次凑齐的那天就是"确认日"（ETF 份额次日才公布，所以只用前一天及更早的申购数据）。
统计确认日收盘买入后 20/60/120 天的涨跌，以及之后 60 天内还会再跌多少。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "daily"))
import dingdi  # noqa: E402
from tdx_reader import read_day  # noqa: E402

WINDOW = 5
GAP = 20
ETF_YI = 200.0


def level3(b2: pd.Series, mid: pd.Series, etf: pd.Series, etf_yi: float = ETF_YI) -> pd.Series:
    """★★B 三级底部：近 6 个交易日（含今天）出现过 ★B，且同期 中线温度 ≤10 或 宽基 ETF 单日净申购 ≥etf_yi 亿
    （申购数据次日才公布，只用前一天及更早的）。同一轮（前 20 个交易日内已标过）只标第一次。"""
    idx = b2.index
    win = lambda s: s.astype(float).rolling(WINDOW + 1, min_periods=1).max().fillna(0).astype(bool)  # noqa: E731
    cond = win(b2.fillna(False)) & (win(mid.reindex(idx) <= 10) | win(etf.reindex(idx).shift(1) >= etf_yi))
    seen = cond.astype(float).shift(1).rolling(GAP, min_periods=1).max().fillna(0).astype(bool)
    return cond & ~seen


def episodes(code: str, mid: pd.Series, etf: pd.Series, etf_yi: float) -> pd.DataFrame:
    df = read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day")
    r = dingdi.compute(df)
    c, low = df.close, df.low
    idx = df.index
    m = mid.reindex(idx)
    e_known = etf.reindex(idx).shift(1)               # 前一天的净申购今天才知道
    b2 = r["底部二级"].to_numpy()
    out, last = [], -10**9
    for i in np.flatnonzero(b2):
        if idx[i] < pd.Timestamp("2011-08-01") or i - last < GAP:
            if i - last < GAP:
                last = i
            continue
        last = i
        j_end = min(i + WINDOW, len(idx) - 1)
        rec = {"date": idx[i]}
        for key, cond in (("冷", lambda k: (m.iloc[i:k + 1] <= 10).any()),
                          ("国家队", lambda k: (e_known.iloc[i:k + 1] >= etf_yi).any())):
            rec[key] = next((k for k in range(i, j_end + 1) if cond(k)), None)
        for name, ks in (("★B", [i]), ("★B+冷", [rec["冷"]]), ("★B+国家队", [rec["国家队"]]),
                         ("★B+两者", [rec["冷"], rec["国家队"]]), ("★B+任一", [rec["冷"], rec["国家队"]])):
            ks2 = [k for k in ks if k is not None]
            if name == "★B+两者" and len(ks2) < 2:
                continue
            if name != "★B" and not ks2:
                continue
            k = min(ks2) if name == "★B+任一" else max(ks2)
            row = {"组": name, "指数": code, "★B日": idx[i].date(), "确认日": idx[k].date(), "中线温度": m.iloc[i]}
            for n in (20, 60, 120):
                row[f"r{n}"] = c.iloc[k + n] / c.iloc[k] - 1 if k + n < len(c) else np.nan
            row["再跌"] = low.iloc[k + 1:k + 61].min() / c.iloc[k] - 1 if k + 1 < len(c) else np.nan
            out.append(row)
    return pd.DataFrame(out)


def main(etf_yi=200.0):
    from update import INDEXES, etf_net_creation  # 放在函数里，避免和 daily/update.py 互相导入
    mid = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "temperature_v1.parquet")["中线温度"]
    etf = etf_net_creation()["net_yi"]
    names = {code: name for name, code, _ in INDEXES}
    E = pd.concat([episodes(code, mid, etf, etf_yi) for _, code, _ in INDEXES], ignore_index=True)
    E["指数"] = E["指数"].map(names)
    return E


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    for thr in (100.0, 200.0):
        E = main(thr)
        print(f"\n===== 国家队门槛：单日净申购 ≥{thr:.0f} 亿 =====")
        g = E.groupby("组", sort=False).agg(轮数=("r60", "size"), r20均=("r20", "mean"), r60均=("r60", "mean"), r60胜=("r60", lambda x: (x.dropna() > 0).mean()),
                                            r120均=("r120", "mean"), 再跌中位=("再跌", "median"), 再跌最差=("再跌", "min"))
        print(g.round(3).to_string())
        for grp in ("★B+冷", "★B+国家队", "★B+两者"):
            s = E[E["组"] == grp].sort_values("确认日")
            print(f"\n{grp}：", "; ".join(f"{a} {n} 60天{r:+.0%}" for a, n, r in zip(s["确认日"], s["指数"], s["r60"].fillna(np.nan))))
