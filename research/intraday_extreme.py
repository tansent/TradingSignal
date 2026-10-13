"""检验"ETF 急跌是买点、急涨是卖点，但一路下跌不能买"。

用指数日线近似 ETF：
- 急跌：盘中最低价比昨收低 ≥ X（固定 3%，或按近 20 日波动率的 2 倍）。假设挂单在阈值价成交。
- 急涨：盘中最高价比昨收高 ≥ X，假设在阈值价卖出。
- 一路下跌：收盘在 60 日线下方且 60 日线比 20 天前低（下跌趋势中）。
看成交后到第 1/5/20 个交易日收盘的涨跌，和所有交易日的平均比较。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tdx_reader import read_day  # noqa: E402

IDX = [("沪深300", "sh000300"), ("中证1000", "sh000852"), ("国证2000", "sz399303"), ("创业板指", "sz399006")]


def events(code: str, start="2012-01-01", sigma_k=None, fixed=0.03) -> pd.DataFrame:
    d = read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day")
    pc = d.close.shift()
    sig = (d.close / pc - 1).rolling(20).std().shift()
    thr = sig * sigma_k if sigma_k else pd.Series(fixed, index=d.index)
    ma60 = d.close.rolling(60).mean()
    down = (d.close.shift() < ma60.shift()) & (ma60.shift() < ma60.shift(21))   # 前一天已处于下跌趋势
    out = pd.DataFrame({"low_hit": d.low <= pc * (1 - thr), "high_hit": d.high >= pc * (1 + thr),
                        "buy_px": pc * (1 - thr), "sell_px": pc * (1 + thr), "down": down}, index=d.index)
    for n in (1, 5, 20):
        fut = d.close.shift(-(n - 1))
        out[f"买{n}"] = fut / out.buy_px - 1          # 急跌挂单买入后
        out[f"卖{n}"] = fut / out.sell_px - 1         # 急涨挂单卖出后（正数 = 卖早了）
        out[f"基准{n}"] = d.close.shift(-n) / d.close - 1
    out["yr"] = out.index.year
    return out[out.index >= start].dropna(subset=["buy_px", "基准20"])


def summary(sigma_k=None, fixed=0.03):
    rows = []
    for name, code in IDX:
        e = events(code, sigma_k=sigma_k, fixed=fixed)
        yrs = (e.index[-1] - e.index[0]).days / 365.25
        b, s = e[e.low_hit], e[e.high_hit]
        row = {"指数": name, "急跌天数/年": len(b) / yrs, "急涨天数/年": len(s) / yrs}
        for lab, g in (("急跌买-全部", b), ("急跌买-非下跌趋势", b[~b.down]), ("急跌买-下跌趋势中", b[b.down])):
            row[f"{lab} 次数"] = len(g)
            for n in (1, 5, 20):
                row[f"{lab} {n}日"] = g[f"买{n}"].mean()
            row[f"{lab} 20日胜率"] = (g["买20"] > 0).mean()
        for n in (1, 5, 20):
            row[f"急涨卖后 {n}日"] = s[f"卖{n}"].mean()
            row[f"基准 {n}日"] = e[f"基准{n}"].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("指数")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    for lab, kw in (("固定阈值 3%", {"fixed": 0.03}), ("按波动率 2 倍", {"sigma_k": 2.0})):
        T = summary(**kw)
        print(f"\n===== {lab}（2012 年以来）=====")
        print(T[["急跌天数/年", "急涨天数/年"]].round(1).to_string())
        cols = [c for c in T.columns if c.startswith("急跌买") and not c.endswith("次数")]
        print((T[cols] * 100).round(1).to_string())
        print(T[[c for c in T.columns if c.endswith("次数")]].to_string())
        print((T[[c for c in T.columns if c.startswith("急涨卖后") or c.startswith("基准")]] * 100).round(2).to_string())
