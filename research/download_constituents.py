"""用 baostock 下载 沪深300 / 中证500 / 上证50 的历史成分股（每月月末一个快照）。"""
from pathlib import Path
import baostock as bs
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "constituents"
QUERIES = {"hs300": bs.query_hs300_stocks, "zz500": bs.query_zz500_stocks, "sz50": bs.query_sz50_stocks}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bs.login()
    dates = pd.date_range("2010-01-31", "2026-09-30", freq="ME").strftime("%Y-%m-%d").tolist() + ["2026-10-09"]
    for name, fn in QUERIES.items():
        rows = []
        for d in dates:
            rs = fn(date=d)
            while rs.error_code == "0" and rs.next():
                r = rs.get_row_data()
                rows.append((d, r[0], r[1]))
        df = pd.DataFrame(rows, columns=["snapshot", "updateDate", "code"])
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(name, len(df), "rows,", df.updateDate.nunique(), "distinct versions,", df.groupby("snapshot").size().describe()[["min", "max"]].to_dict())
    bs.logout()

if __name__ == "__main__":
    main()
