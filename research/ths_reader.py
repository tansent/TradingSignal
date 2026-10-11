"""读取同花顺本地 hd1.0 格式数据文件（history/*/day/*.day、extra/*.ext）。"""
import struct
from pathlib import Path

import pandas as pd


def decode(u: int) -> float | None:
    """同花顺 4 字节数值：高 4 位是 10 的指数（最高位为 1 表示负指数），低 28 位是整数。"""
    if u in (0xFFFFFFFF, 0x7FFFFFFF, 0x0FFFFFFF):
        return None
    hi = u >> 28
    m = u & 0x0FFFFFFF
    exp = hi & 0x7
    return m * (10 ** -exp if hi & 0x8 else 10 ** exp)


def read(path: str | Path, raw: bool = False) -> pd.DataFrame:
    b = Path(path).read_bytes()
    assert b[:6] == b"hd1.0\x00", "not hd1.0"
    n, hlen, rlen, ncol = struct.unpack_from("<IHHH", b, 6)
    cols = [struct.unpack_from("<HBB", b, 16 + 4 * i) for i in range(ncol)]
    ids = [f"{c[0]:04x}" for c in cols]
    rows = []
    for k in range(n):
        off = hlen + k * rlen
        vals = struct.unpack_from(f"<{ncol}I", b, off)
        rows.append([vals[0]] + [v if raw else decode(v) for v in vals[1:]])
    df = pd.DataFrame(rows, columns=ids)
    df.insert(0, "date", pd.to_datetime(df.pop(ids[0]).astype(str), format="%Y%m%d", errors="coerce"))
    return df


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    df = read(sys.argv[1])
    print(df.tail(int(sys.argv[2]) if len(sys.argv) > 2 else 5).to_string())
