"""拉取主要指数日线（新浪源），保存为 CSV。"""
import sys
import akshare as ak

INDEXES = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
    "sh000300": "沪深300",
    "sh000688": "科创50",
    "sh000905": "中证500",
    "sh000852": "中证1000",
}

out_dir = sys.argv[1]
for code, name in INDEXES.items():
    try:
        df = ak.stock_zh_index_daily(symbol=code)
        df.to_csv(f"{out_dir}/{code}.csv", index=False)
        print(code, name, len(df), df["date"].iloc[0], "->", df["date"].iloc[-1])
    except Exception as e:
        print(code, name, "FAILED", repr(e)[:200])
