"""韩国集成电路出口（HS 8542，存储 + 逻辑芯片），月度，来自联合国贸易数据库 UN Comtrade 公开接口（免费版每次一个月）。
韩国产业通商资源部每月 1 日公布上月出口，所以 t 月的数据在 t+1 月月初就能用。存 data/korea_semi_exports.csv（美元）。
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "korea_semi_exports.csv"
URL = "https://comtradeapi.un.org/public/v1/preview/C/M/HS"


def fetch(start="2010-01", end=None) -> pd.DataFrame:
    old = pd.read_csv(OUT, dtype={"period": str}) if OUT.exists() else pd.DataFrame(columns=["period", "usd"])
    have = set(old.dropna(subset=["usd"]).period)          # 空值的月份重新下载
    months = pd.period_range(start, end or (pd.Timestamp.now() - pd.DateOffset(months=1)).strftime("%Y-%m"), freq="M")
    rows = []
    for m in months:
        per = m.strftime("%Y%m")
        if per in have:
            continue
        for attempt in range(3):
            try:
                r = requests.get(URL, params={"reporterCode": "410", "period": per, "partnerCode": "0",
                                              "cmdCode": "8542", "flowCode": "X"}, timeout=30)
                if r.status_code == 429:
                    time.sleep(10 * (attempt + 1)); continue
                j = r.json()
                d = j.get("data") or []
                v = sum(x.get("primaryValue") or 0 for x in d if x.get("partnerCode") == 0)
                rows.append({"period": per, "usd": v if v else None})
                print(per, r.status_code, v if v else f"空 {str(j)[:120]}", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(per, "ERR", repr(e)[:80], flush=True); time.sleep(5)
        time.sleep(1.2)
    df = pd.concat([old, pd.DataFrame(rows)]).drop_duplicates("period", keep="last").sort_values("period")
    df.to_csv(OUT, index=False)
    return df


if __name__ == "__main__":
    fetch(*(sys.argv[1:] or []))
