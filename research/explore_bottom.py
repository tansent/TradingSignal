import sys
import pandas as pd
sys.path.insert(0, "research")
from lib import load, features, NAMES

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
COLS = ["dd120", "z_dd120", "days_from_high", "z5", "z10", "z20", "z60", "zbias20", "zbias60", "rsi6", "rsi14", "down20", "vr20", "regime"]
BOTTOMS = ["2005-06-06", "2008-10-28", "2012-12-04", "2013-06-25", "2015-07-08", "2015-08-26", "2016-01-28",
           "2018-10-19", "2019-01-04", "2020-02-04", "2020-03-23", "2022-03-15", "2022-04-26", "2022-10-31",
           "2024-02-05", "2024-09-18", "2025-04-07", "2026-10-09"]

for code in ["sh000300", "sz399001", "sz399006"]:
    df = load(code)
    f = features(df, "intraday")
    num = f[COLS[:-1]]
    pct = num.rank(pct=True)
    print(f"\n===== {NAMES[code]} (盘中口径, 括号内为历史分位) =====")
    rows = []
    for d in BOTTOMS:
        d = pd.Timestamp(d)
        if d not in f.index or pd.isna(f.loc[d, "vol250"]):
            continue
        r = {"date": d.date()}
        for k in COLS[:-1]:
            r[k] = f"{f.loc[d, k]:.2f}({pct.loc[d, k]*100:.0f})"
        r["regime"] = f.loc[d, "regime"]
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
