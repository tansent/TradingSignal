"""按指数成分股计算情绪指标的候选算法，并与他的读数（digitize_his.py 的结果）比对。

成分股用中证指数公司当前名单（近似；历史调样的影响在近一年内较小）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tdx_reader import read_day  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V = Path("D:/TDXapp/vipdoc")
INDEX = {"上证50": ("000016", "sh000016"), "沪深300": ("000300", "sh000300"),
         "中证500": ("000905", "sh000905"), "中证1000": ("000852", "sh000852")}


def stock_path(code: str) -> Path:
    mk = "sh" if code.startswith(("6", "9")) else ("bj" if code.startswith(("4", "8", "92")) else "sz")
    return V / mk / "lday" / f"{mk}{code}.day"


def adj_close(code: str) -> pd.Series | None:
    """近似复权收盘：除权日（涨跌幅超出 22%/12% 等不可能值）用当日 收/开 代替。"""
    p = stock_path(code)
    if not p.exists():
        return None
    df = read_day(p)
    df = df[(df.close > 0) & (df.volume > 0)]
    c, o = df.close, df.open
    r = c.pct_change()
    lim = 0.30 if code.startswith(("4", "8", "92")) else (0.20 if code.startswith(("30", "68")) else 0.10)
    ex = r.abs() > lim + 0.02
    r = r.where(~ex, c / o - 1).fillna(0)
    return (1 + r).cumprod()


def panel(index_code: str) -> pd.DataFrame:
    cons = pd.read_csv(ROOT / "data" / "constituents" / f"{index_code}_current.csv", dtype={"成分券代码": str})
    cols = {}
    for code in cons["成分券代码"]:
        s = adj_close(code.zfill(6))
        if s is not None:
            cols[code] = s
    return pd.DataFrame(cols).sort_index()


def rsi_sma(c: pd.DataFrame | pd.Series, n: int):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    ab = d.abs().ewm(alpha=1 / n, adjust=False).mean()
    return up / ab * 100


def kdj(df: pd.DataFrame, n=9):
    ll, hh = df.low.rolling(n).min(), df.high.rolling(n).max()
    rsv = (df.close - ll) / (hh - ll) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    return rsv, k, d, 3 * k - 2 * d


def candidates(name: str) -> pd.DataFrame:
    ic, tdx = INDEX[name]
    P = panel(ic)
    idx = read_day(V / "sh" / "lday" / f"{tdx}.day")
    traded = P.notna() & P.shift(1).notna()
    ret = P.pct_change()
    out = pd.DataFrame(index=P.index)
    out["上涨占比"] = (ret > 0).sum(axis=1) / traded.sum(axis=1) * 100
    for n in (5, 10, 20):
        out[f"站上MA{n}占比"] = (P > P.rolling(n).mean()).sum(axis=1) / P.rolling(n).mean().notna().sum(axis=1) * 100
    out["成分股RSI6均值"] = rsi_sma(P, 6).mean(axis=1)
    out["成分股RSI14均值"] = rsi_sma(P, 14).mean(axis=1)
    out["5日上涨占比"] = (P.pct_change(5) > 0).sum(axis=1) / P.shift(5).notna().sum(axis=1) * 100
    rsv, k, d, j = kdj(idx)
    out["指数RSV9"], out["指数K"], out["指数D"] = rsv, k, d
    out["指数RSI6"] = rsi_sma(idx.close, 6)
    return out


if __name__ == "__main__":
    his = pd.read_csv(ROOT / "data" / "his_sentiment_digitized.csv", index_col=0, parse_dates=True)
    rows = []
    allc = {}
    for name in INDEX:
        c = candidates(name)
        allc[name] = c
        j = c.join(his[name].rename("他"), how="inner").dropna(subset=["他"])
        for col in c.columns:
            x = j[[col, "他"]].dropna()
            rows.append({"指数": name, "候选": col, "相关系数": x[col].corr(x["他"]),
                         "平均绝对误差": (x[col] - x["他"]).abs().mean()})
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(r.pivot(index="候选", columns="指数", values="相关系数").round(3).assign(平均=lambda d: d.mean(axis=1)).sort_values("平均", ascending=False))
    print()
    print(r.pivot(index="候选", columns="指数", values="平均绝对误差").round(1))
    pd.concat(allc, axis=1).to_parquet(ROOT / "data" / "index_sentiment_candidates.parquet")
