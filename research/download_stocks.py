"""用 baostock 下载 2010 年以来全部 A 股（含已退市）日线，不复权。

每只股票存一个 parquet 到 data/stocks/，已存在的跳过（可断点续传）。
用法：python research/download_stocks.py [进程数]
"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

import baostock as bs
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "stocks"
LOG = ROOT / "data" / "download.log"
START, END = "2010-01-01", "2026-10-09"
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST"
NUM = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]


def stock_list() -> pd.DataFrame:
    bs.login()
    rs = bs.query_stock_basic()
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    df = pd.DataFrame(rows, columns=rs.fields)
    df = df[df["type"] == "1"]  # 股票
    df = df[(df["ipoDate"] <= END) & ((df["outDate"] == "") | (df["outDate"] >= START))]
    df.to_csv(ROOT / "data" / "stock_basic.csv", index=False)
    return df


def fetch(code: str, start: str, end: str) -> pd.DataFrame:
    for attempt in range(3):
        rs = bs.query_history_k_data_plus(code, FIELDS, start_date=start, end_date=end,
                                          frequency="d", adjustflag="3")
        if rs.error_code != "0":
            time.sleep(2 + attempt * 3)
            bs.login()
            continue
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        for c in NUM:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce").astype("Int8")
        df["isST"] = pd.to_numeric(df["isST"], errors="coerce").astype("Int8")
        return df
    raise RuntimeError(f"{code}: {rs.error_code} {rs.error_msg}")


def worker(tasks, q):
    bs.login()
    for code, start, end in tasks:
        f = OUT / f"{code}.parquet"
        if f.exists():
            q.put((code, "skip", 0))
            continue
        try:
            df = fetch(code, start, end)
            df.to_parquet(f, index=False)
            q.put((code, "ok", len(df)))
        except Exception as e:  # noqa: BLE001
            q.put((code, f"err {e!r}"[:200], 0))
    bs.logout()
    q.put(None)


def main(nproc: int):
    OUT.mkdir(parents=True, exist_ok=True)
    lst = stock_list()
    tasks = []
    for r in lst.itertuples():
        s = max(START, r.ipoDate)
        e = min(END, r.outDate) if r.outDate else END
        tasks.append((r.code, s, e))
    chunks = [tasks[i::nproc] for i in range(nproc)]
    q = mp.Queue()
    procs = [mp.Process(target=worker, args=(c, q)) for c in chunks]
    for p in procs:
        p.start()
    done = finished = rows = 0
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as log:
        log.write(f"start {time.ctime()} total {len(tasks)} nproc {nproc}\n")
        while finished < nproc:
            m = q.get()
            if m is None:
                finished += 1
                continue
            done += 1
            rows += m[2]
            if m[1] not in ("ok", "skip"):
                log.write(f"{m[0]} {m[1]}\n")
            if done % 100 == 0 or done == len(tasks):
                el = time.time() - t0
                log.write(f"{done}/{len(tasks)} rows {rows} elapsed {el / 60:.1f}min "
                          f"eta {el / done * (len(tasks) - done) / 60:.0f}min\n")
                log.flush()
        log.write(f"done {time.ctime()}\n")
    for p in procs:
        p.join()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
