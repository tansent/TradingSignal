"""从通达信全部 A 股日线计算每日市场宽度与短线情绪指标。

输出 data/breadth_daily.parquet（按日期 × 板块），列见 aggregate()。

处理要点：
- 涨跌停价 = round(昨收 × (1 ± 幅度), 2)。幅度：主板 10%，创业板 2020-08-24 起 20%（之前 10%），
  科创板 20%，北交所 30%。拿不到历史 ST 状态，ST 的 5% 涨停不计入。
- 新股上市前 5 个交易日不计入（无涨跌幅限制，会干扰统计）。
- 日线不复权：若按昨收算的涨跌幅超出涨跌停幅度 2% 以上，视为除权日，当天剔除，
  复权序列用当天"收盘/开盘"近似当天真实涨跌。
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tdx_reader import read_day  # noqa: E402

V = Path("D:/TDXapp/vipdoc")
OUT = Path(__file__).resolve().parent.parent / "data"
CYB_REFORM = pd.Timestamp("2020-08-24")
BOARD_PREFIX = {
    "sh_main": ("sh600", "sh601", "sh603", "sh605"),
    "kcb": ("sh688", "sh689"),
    "sz_main": ("sz000", "sz001", "sz002", "sz003"),
    "cyb": ("sz300", "sz301"),
    "bj": ("bj920", "bj43", "bj83", "bj87"),
}


def board_of(code: str) -> str | None:
    for b, pres in BOARD_PREFIX.items():
        if code.startswith(pres):
            return b
    return None


def limit_pct(board: str, dates: pd.DatetimeIndex) -> np.ndarray:
    if board in ("sh_main", "sz_main"):
        return np.full(len(dates), 0.10)
    if board == "kcb":
        return np.full(len(dates), 0.20)
    if board == "cyb":
        return np.where(dates >= CYB_REFORM, 0.20, 0.10)
    return np.full(len(dates), 0.30)


def round2(x: np.ndarray) -> np.ndarray:
    """交易所四舍五入到分。"""
    return np.floor(x * 100 + 0.5 + 1e-9) / 100


def per_stock(code: str, board: str) -> pd.DataFrame | None:
    df = read_day(V / code[:2] / "lday" / f"{code}.day")
    df = df[(df["close"] > 0) & (df["volume"] > 0)]
    if len(df) < 2:
        return None
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    pc = np.r_[np.nan, c[:-1]]
    L = limit_pct(board, df.index)
    ret = c / pc - 1
    age = np.arange(len(df))
    exday = np.abs(ret) > L + 0.02                       # 除权（或数据异常）
    valid = (age >= 5) & ~exday & ~np.isnan(ret)

    up_px, dn_px = round2(pc * (1 + L)), round2(pc * (1 - L))
    zt = valid & (np.abs(c - up_px) < 0.0051)            # 收盘涨停
    dt = valid & (np.abs(c - dn_px) < 0.0051)            # 收盘跌停
    touch = valid & (h >= up_px - 0.0051)                # 盘中摸过涨停
    zb = touch & ~zt                                     # 炸板

    # 连板数
    lb = np.zeros(len(df), dtype=np.int16)
    for i in range(len(df)):
        lb[i] = lb[i - 1] + 1 if zt[i] and i > 0 else (1 if zt[i] else 0)

    # 复权收盘（除权日用当日 收/开 近似真实涨跌）
    adj_ret = np.where(exday, c / o - 1, ret)
    adj_ret = np.nan_to_num(adj_ret)
    adj = np.cumprod(1 + adj_ret)
    s = pd.Series(adj, index=df.index)
    hh60 = s.rolling(60, min_periods=60).max().shift(1).to_numpy()
    ll60 = s.rolling(60, min_periods=60).min().shift(1).to_numpy()
    ma20 = s.rolling(20).mean().to_numpy()
    ma60 = s.rolling(60).mean().to_numpy()

    out = pd.DataFrame({
        "valid": valid,
        "up": valid & (ret > 0), "down": valid & (ret < 0),
        "ret": np.where(valid, ret, np.nan),
        "zt": zt, "dt": dt, "zb": zb, "touch": touch, "lb": lb,
        "prev_zt": np.r_[False, zt[:-1]], "prev_lb": np.r_[0, lb[:-1]],
        "nh60": valid & (adj > hh60), "nl60": valid & (adj < ll60),
        "above20": valid & (adj > ma20), "above60": valid & (adj > ma60),
        "amount": df["amount"].to_numpy(),
    }, index=df.index)
    out["board"] = board
    return out


def aggregate(frames: list[pd.DataFrame]) -> pd.DataFrame:
    big = pd.concat(frames)
    big.index.name = "date"
    big = big.reset_index()
    big["prev_zt_ret"] = np.where(big["prev_zt"] & big["valid"], big["ret"], np.nan)
    big["prev_lb_up"] = big["prev_zt"] & big["zt"]         # 昨日涨停今日继续涨停（晋级）

    def agg(g: pd.DataFrame) -> pd.DataFrame:
        return g.groupby("date").agg(
            n=("valid", "sum"), up=("up", "sum"), down=("down", "sum"),
            zt=("zt", "sum"), dt=("dt", "sum"), zb=("zb", "sum"), touch=("touch", "sum"),
            max_lb=("lb", "max"), lb2=("lb", lambda x: int((x >= 2).sum())),
            prev_zt=("prev_zt", "sum"), promote=("prev_lb_up", "sum"),
            prev_zt_ret=("prev_zt_ret", "mean"), ret_mean=("ret", "mean"), ret_median=("ret", "median"),
            nh60=("nh60", "sum"), nl60=("nl60", "sum"),
            above20=("above20", "sum"), above60=("above60", "sum"), amount=("amount", "sum"),
        )

    parts = [agg(big).assign(board="all")]
    for b, g in big.groupby("board"):
        parts.append(agg(g).assign(board=b))
    return pd.concat(parts).reset_index()


def main():
    t0 = time.time()
    frames, n = [], 0
    for m in ("sh", "sz", "bj"):
        for f in sorted((V / m / "lday").glob("*.day")):
            code = f.stem
            b = board_of(code)
            if b is None:
                continue
            r = per_stock(code, b)
            if r is not None:
                frames.append(r)
                n += 1
    print(f"stocks {n}, read+compute {time.time() - t0:.0f}s")
    res = aggregate(frames)
    res.to_parquet(OUT / "breadth_daily.parquet", index=False)
    print(f"saved {len(res)} rows, total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
