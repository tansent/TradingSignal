"""通达信没有写入最近的交易日日线时，用新浪数据补上（不改动通达信文件）。

补出的数据存 data/tdx_overlay.parquet，research/tdx_reader.read_day 读取时自动合并
（只合并晚于通达信文件最后一天的日期，通达信有数据时以通达信为准）。
- 最近一个交易日：个股和指数都用新浪收盘快照。指数的涨跌家数按成分股（沪深300/上证50/中证500/中证1000）
  或板块（上证/深证/创业板/科创板）从个股快照统计，口径与通达信一致。
- 中间缺的交易日（好几天没运行）：8 个指数用新浪指数日线补齐（没有涨跌家数）；个股没有可批量下载的
  历史来源，不补，并且最近一天的个股快照也不写入（否则"昨收"接到几天前，涨跌停统计会错），
  全市场宽度和温度停在个股数据连续的最后一天，等通达信「盘后数据下载」补齐。
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["NO_PROXY"] = "*"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
from tdx_reader import OVERLAY, VIPDOC, is_index  # noqa: E402

INDEX_UNIVERSE = {
    "sh000001": lambda c: c.str.startswith(("sh60", "sh68")),
    "sz399001": lambda c: c.str.startswith(("sz00", "sz30")),
    "sz399006": lambda c: c.str.startswith("sz30"),
    "sh000688": lambda c: c.str.startswith("sh68"),
}
CONSTITUENTS = {"sh000300": "000300", "sh000016": "000016", "sh000905": "000905", "sh000852": "000852"}


def trading_calendar() -> pd.DatetimeIndex:
    import akshare as ak
    return pd.DatetimeIndex(pd.to_datetime(ak.tool_trade_date_hist_sina()["trade_date"])).sort_values()


def last_closed_trading_day(cal: pd.DatetimeIndex | None = None) -> pd.Timestamp:
    cal = trading_calendar() if cal is None else cal
    now = pd.Timestamp.now()
    today = now.normalize()
    closed = cal[(cal < today) | ((cal == today) & (now.hour * 60 + now.minute >= 15 * 60 + 5))]
    return closed.max()


def tdx_last_date() -> pd.Timestamp:
    import numpy as np
    b = (VIPDOC / "sh" / "lday" / "sh000001.day").read_bytes()
    return pd.to_datetime(str(np.frombuffer(b[-32:-28], dtype="<u4")[0]))


def constituents(code6: str) -> set[str]:
    f = ROOT / "data" / "constituents" / f"{code6}_current.csv"
    df = pd.read_csv(f, dtype={"成分券代码": str})
    return {("sh" if x.startswith(("6", "9")) else "sz") + x.zfill(6) for x in df["成分券代码"]}


def index_history(days: list[pd.Timestamp]) -> list[dict]:
    """新浪指数日线，补中间缺的交易日（只有价格和成交量，没有涨跌家数）。"""
    import akshare as ak
    rows = []
    for code in list(INDEX_UNIVERSE) + list(CONSTITUENTS):
        h = ak.stock_zh_index_daily(symbol=code)
        h["date"] = pd.to_datetime(h["date"])
        for _, r in h[h["date"].isin(days)].iterrows():
            rows.append({"code": code, "date": r["date"], "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"],
                         "amount": np.nan, "volume": float(r["volume"]), "adv": np.nan, "dec": np.nan})
    return rows


def fill(force: bool = False) -> str:
    """返回说明文字。force=True 时即使通达信已有该日数据也重新生成（用于测试）。"""
    import akshare as ak
    cal = trading_calendar()
    target = last_closed_trading_day(cal)
    have = tdx_last_date()
    if have >= target and not force:
        return f"通达信已有 {have.date()} 的数据，无需补充"
    old = pd.read_parquet(OVERLAY) if OVERLAY.exists() else pd.DataFrame(columns=["code", "date"])
    old = old[old["date"] > have]  # 通达信已有的日期不再需要
    middle = [d for d in cal[(cal > have) & (cal < target)]]
    is_idx = old["code"].map(is_index).astype(bool)
    idx_have = set(old.loc[old["code"] == "sh000001", "date"])
    stock_n = old.loc[~is_idx, "date"].value_counts()
    stock_have = set(stock_n[stock_n >= 1000].index)
    idx_missing = [d for d in middle if d not in idx_have]
    stock_gap = [d for d in middle if d not in stock_have]

    s = ak.stock_zh_a_spot()
    s = s[(s["成交量"] > 0) & (s["今开"] > 0)].copy()
    rows = pd.DataFrame({
        "code": s["代码"], "date": target, "open": s["今开"], "high": s["最高"], "low": s["最低"],
        "close": s["最新价"], "amount": s["成交额"].astype(float), "volume": s["成交量"].astype(float),
    })
    up = s.set_index("代码")["涨跌额"] > 0
    dn = s.set_index("代码")["涨跌额"] < 0
    ix = ak.stock_zh_index_spot_sina().set_index("代码")
    idx_rows = []
    for code in list(INDEX_UNIVERSE) + list(CONSTITUENTS):
        if code not in ix.index:
            continue
        r = ix.loc[code]
        if code in INDEX_UNIVERSE:
            mask = INDEX_UNIVERSE[code](up.index.to_series())
            uni = set(up.index[mask.to_numpy()])
        else:
            uni = constituents(CONSTITUENTS[code])
        idx_rows.append({"code": code, "date": target, "open": r["今开"], "high": r["最高"], "low": r["最低"], "close": r["最新价"],
                         "amount": float(r["成交额"]), "volume": float(r["成交量"]),
                         "adv": float(up[up.index.isin(uni)].sum()), "dec": float(dn[dn.index.isin(uni)].sum())})
    if stock_gap:
        rows = rows.iloc[0:0]
    hist = index_history(idx_missing) if idx_missing else []
    rows = pd.concat([df for df in (rows, pd.DataFrame(idx_rows), pd.DataFrame(hist)) if len(df)], ignore_index=True)
    new = (pd.concat([old, rows], ignore_index=True) if len(old) else rows).drop_duplicates(["code", "date"], keep="last")
    new.to_parquet(OVERLAY, index=False)
    msg = f"通达信最新为 {have.date()}，已用新浪收盘快照补充 {target.date()}：指数 {len(idx_rows)} 个"
    if hist:
        msg += f"；中间缺的 {len(idx_missing)} 个交易日已用新浪指数日线补齐 8 个指数"
    if stock_gap:
        msg += (f"。警告：个股缺 {stock_gap[0].date()} 起 {len(stock_gap)} 个交易日，没有补个股，全市场宽度和温度停在"
                f" {cal[cal < stock_gap[0]].max().date()}；请打开通达信 → 盘后数据下载 补齐后再运行")
    else:
        msg += f"、个股 {len(s)} 只"
    return msg


if __name__ == "__main__":
    print(fill(force="--force" in sys.argv))
