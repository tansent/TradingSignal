"""检验"假破位 / 假突破"：
假破位：收盘跌破此前 60 日最低收盘价，且 5 个交易日内收盘重新站回该价位 → 在站回当天记为信号。
真破位：跌破后 5 天内没有站回 → 在第 5 天记为对照。
假突破 / 真突破同理（60 日最高收盘价）。
个股收益扣除同期全市场等权平均收益（超额收益），以去掉大盘涨跌的影响。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from index_sentiment import adj_close  # noqa: E402
from lib import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V = Path("D:/TDXapp/vipdoc")
N, K = 60, 5


def events(c: pd.Series) -> pd.DataFrame:
    """返回每个事件：类型、信号日、对应的关键价位。"""
    lo = c.shift(1).rolling(N, min_periods=N).min()
    hi = c.shift(1).rolling(N, min_periods=N).max()
    cv = c.to_numpy()
    out = []
    for kind, level, broke in (("破位", lo, c < lo), ("突破", hi, c > hi)):
        b = broke.to_numpy()
        lv = level.to_numpy()
        i, n = N + 1, len(c)
        while i < n - K:
            if b[i] and not b[i - 1]:
                L = lv[i]
                back = None
                for j in range(i + 1, i + K + 1):
                    if (kind == "破位" and cv[j] > L) or (kind == "突破" and cv[j] < L):
                        back = j
                        break
                if back is not None:
                    out.append(("假" + kind, c.index[back]))
                    i = back + 1
                else:
                    out.append(("真" + kind, c.index[i + K]))
                    i = i + K + 1
                continue
            i += 1
    return pd.DataFrame(out, columns=["type", "date"])


def fwd(c: pd.Series, dates, horizons=(20, 60)) -> pd.DataFrame:
    idx = c.index.get_indexer(dates)
    res = {}
    for h in horizons:
        j = np.minimum(idx + h, len(c) - 1)
        r = c.to_numpy()[j] / c.to_numpy()[idx] - 1
        r[idx + h >= len(c)] = np.nan
        res[f"r{h}"] = r
    return pd.DataFrame(res, index=dates)


def market_index() -> pd.Series:
    b = pd.read_parquet(ROOT / "data" / "breadth_daily.parquet")
    a = b[b.board == "all"].set_index("date").sort_index()
    return (1 + a["ret_mean"].fillna(0)).cumprod()


def stocks():
    mkt = market_index()
    rows = []
    files = [f for m in ("sh", "sz") for f in (V / m / "lday").glob("*.day")
             if f.stem[2:5] in ("600", "601", "603", "605", "688", "000", "001", "002", "003", "300", "301")]
    for f in files:
        c = adj_close(f.stem[2:])
        if c is None or len(c) < N + 80:
            continue
        c = c[c.index >= "2010-01-01"]
        ev = events(c)
        if ev.empty:
            continue
        r = fwd(c, pd.DatetimeIndex(ev.date))
        m = fwd(mkt.reindex(c.index).ffill(), pd.DatetimeIndex(ev.date))
        ev = ev.assign(r20=r.r20.to_numpy() - m.r20.to_numpy(), r60=r.r60.to_numpy() - m.r60.to_numpy())
        rows.append(ev)
    return pd.concat(rows)


def indexes():
    rows = []
    for code in ("sh000001", "sh000300", "sz399001", "sz399006", "sh000905", "sh000852"):
        c = load(code).close
        c = c[c.index >= "2005-01-01"]
        ev = events(c)
        r = fwd(c, pd.DatetimeIndex(ev.date))
        rows.append(ev.assign(r20=r.r20.to_numpy(), r60=r.r60.to_numpy(), code=code))
    return pd.concat(rows)


def summarize(df, label, absolute=False):
    print(f"\n== {label} ==")
    print(f"{'类型':<6}{'次数':>8}{'20日均':>9}{'20日>0':>8}{'60日均':>9}{'60日>0':>8}{'60日中位':>9}")
    for t in ("假破位", "真破位", "假突破", "真突破"):
        g = df[df.type == t].dropna(subset=["r60"])
        print(f"{t:<6}{len(g):>8}{g.r20.mean():>+9.2%}{(g.r20 > 0).mean():>8.0%}{g.r60.mean():>+9.2%}{(g.r60 > 0).mean():>8.0%}{g.r60.median():>+9.2%}")


if __name__ == "__main__":
    summarize(indexes(), "6 个指数，2005 起（绝对收益）")
    S = stocks()
    S.to_parquet(ROOT / "data" / "false_break_stocks.parquet")
    summarize(S, "全部 A 股个股，2010 起（相对全市场等权的超额收益）")
