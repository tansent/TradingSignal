"""上交所宽基 ETF 每日份额（增量下载，存 data/etf_shares_sse.csv）。"""
import os
import sys
import time
from pathlib import Path

import pandas as pd

os.environ["NO_PROXY"] = "*"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "etf_shares_sse.csv"
KEY = {"510300": "华泰柏瑞沪深300", "510310": "易方达沪深300", "510330": "华夏沪深300", "510050": "华夏上证50",
       "510500": "南方中证500", "512100": "南方中证1000", "588000": "华夏科创50", "510180": "华安上证180"}


def trading_days(start: str) -> list[str]:
    sys.path.insert(0, str(ROOT / "research"))
    from tdx_reader import read_day
    d = read_day("D:/TDXapp/vipdoc/sh/lday/sh000001.day").index
    return [x.strftime("%Y%m%d") for x in d[d >= start]]


def update(start: str = "2023-10-01"):
    import akshare as ak
    old = pd.read_csv(OUT, dtype={"code": str}) if OUT.exists() else pd.DataFrame(columns=["date", "code", "shares"])
    done = set(old["date"].astype(str))
    rows = []
    for day in trading_days(start):
        if day in done:
            continue
        try:
            df = ak.fund_etf_scale_sse(date=day)
        except Exception as e:  # noqa: BLE001
            print(day, "ERR", repr(e)[:80])
            time.sleep(2)
            continue
        df = df[df["基金代码"].astype(str).isin(KEY)]
        rows += [{"date": day, "code": str(r["基金代码"]), "shares": float(r["基金份额"])} for _, r in df.iterrows()]
        time.sleep(0.3)
    if rows:
        new = pd.concat([old, pd.DataFrame(rows)]).drop_duplicates(["date", "code"]).sort_values(["date", "code"])
        new.to_csv(OUT, index=False)
        print("added days:", len({r['date'] for r in rows}), "total rows:", len(new))
    else:
        print("no new days")


if __name__ == "__main__":
    update(sys.argv[1] if len(sys.argv) > 1 else "2023-10-01")
