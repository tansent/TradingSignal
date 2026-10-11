"""把上交所宽基 ETF 每日份额补到 2015 年（daily/etf_shares.py 只从 2023-10 开始）。

单线程、每天一个请求、间隔 0.4 秒；每 40 天写一次盘，中断后重跑会跳过已有日期。
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd

os.environ["NO_PROXY"] = "*"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "daily"))
from etf_shares import KEY, OUT, trading_days  # noqa: E402


def save(old: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    new = pd.concat([old, pd.DataFrame(rows)]).drop_duplicates(["date", "code"]).sort_values(["date", "code"])
    new.to_csv(OUT, index=False)
    return new


def main(start="2015-01-05", end="2023-10-08"):
    import akshare as ak
    old = pd.read_csv(OUT, dtype={"code": str, "date": str})
    done = set(old["date"])
    days = [d for d in trading_days(start) if d <= end.replace("-", "") and d not in done]
    print("to fetch:", len(days), flush=True)
    rows, fails = [], []
    for i, day in enumerate(days, 1):
        try:
            df = ak.fund_etf_scale_sse(date=day)
            df = df[df["基金代码"].astype(str).isin(KEY)]
            rows += [{"date": day, "code": str(r["基金代码"]), "shares": float(r["基金份额"])} for _, r in df.iterrows()]
        except Exception as e:  # noqa: BLE001
            fails.append(day)
            print(day, "ERR", repr(e)[:80], flush=True)
            time.sleep(3)
        if i % 40 == 0:
            old = save(old, rows); rows = []
            print(i, "/", len(days), day, flush=True)
        time.sleep(0.4)
    save(old, rows)
    print("done; failed days:", len(fails), fails[:20], flush=True)


if __name__ == "__main__":
    main()
