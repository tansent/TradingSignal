"""读取通达信本地日线文件（vipdoc/{sh,sz,bj}/lday/*.day）。

每条记录 32 字节：日期(YYYYMMDD) 开 高 低 收(整数, 股票/指数 ×100, 基金/债券 ×1000)
成交额(float32) 成交量(uint32) 保留(uint32)。
"""
import struct
from pathlib import Path

import numpy as np
import pandas as pd

REC = struct.Struct("<5IfII")
VIPDOC = Path("D:/TDXapp/vipdoc")
OVERLAY = Path(__file__).resolve().parent.parent / "data" / "tdx_overlay.parquet"
_overlay_cache: dict = {}


def _overlay() -> pd.DataFrame | None:
    """新浪补充的日线（daily/sina_fill.py 写入）；按文件修改时间缓存。"""
    if not OVERLAY.exists():
        return None
    mt = OVERLAY.stat().st_mtime
    if _overlay_cache.get("mtime") != mt:
        _overlay_cache["df"] = pd.read_parquet(OVERLAY)
        _overlay_cache["mtime"] = mt
    return _overlay_cache["df"]

# 基金、ETF、债券价格是 ×1000 存的
_FUND_PREFIX = ("15", "16", "18", "50", "51", "52", "56", "58", "11", "12", "13")


def price_scale(code: str) -> int:
    num = code[2:] if code[:2] in ("sh", "sz", "bj") else code
    return 1000 if num.startswith(_FUND_PREFIX) else 100


def is_index(code: str) -> bool:
    return code.startswith(("sh000", "sh880", "sh881", "sz399", "bj899"))


def read_day(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    b = path.read_bytes()
    n = len(b) // REC.size
    arr = np.frombuffer(b[: n * REC.size], dtype=np.dtype([
        ("date", "<u4"), ("open", "<u4"), ("high", "<u4"), ("low", "<u4"), ("close", "<u4"),
        ("amount", "<f4"), ("volume", "<u4"), ("_r", "<u4")]))
    s = price_scale(path.stem)
    df = pd.DataFrame({
        "date": pd.to_datetime(arr["date"].astype(str), format="%Y%m%d", errors="coerce"),
        "open": arr["open"] / s, "high": arr["high"] / s, "low": arr["low"] / s, "close": arr["close"] / s,
        "amount": arr["amount"].astype(float), "volume": arr["volume"].astype(float),
    })
    if is_index(path.stem):
        # 指数记录的最后 4 字节：低 16 位 = 上涨家数，高 16 位 = 下跌家数（即公式里的 ADVANCE/DECLINE）
        df["adv"] = (arr["_r"] & 0xFFFF).astype(float)
        df["dec"] = (arr["_r"] >> 16).astype(float)
    df = df.dropna(subset=["date"]).set_index("date")
    ov = _overlay()
    if ov is not None and len(df):
        extra = ov[(ov["code"] == path.stem) & (ov["date"] > df.index.max())]
        if len(extra):
            cols = [c for c in df.columns if c in extra.columns]
            df = pd.concat([df, extra.set_index("date")[cols]]).sort_index()
    return df


def list_files(vipdoc: str | Path, market: str) -> list[Path]:
    return sorted((Path(vipdoc) / market / "lday").glob("*.day"))


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 200)
    df = read_day(sys.argv[1])
    print(len(df), df.index.min().date(), "->", df.index.max().date())
    print(df.tail(5))
