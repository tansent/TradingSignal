"""B 点（△B/★B）买入后要不要设止损：比较不同离场方式（2026-10-17）。

8 个宽基指数，2011-08 起，每轮取第一个 △B/★B（与上一笔间隔 ≥20 个交易日），信号当天收盘买入：
  持有 60 天不止损；收盘跌 5%/8%/12% 止损；按 1.5 倍波动幅度止损（日波动率 × √20 × 1.5）；
  收盘跌破信号当天最低价 1% 止损；破月线 BBI 离场（最长持有 120 天）。
结论：价格止损截断了最大亏损，但把平均收益从 +2.1% 压到 +0.6～1.0%（恐慌底常再探一次底，止损正好洗在反弹前）；
"跌破信号日最低"最差（赚钱比例 21%，平均 -0.2%）；破月 BBI 离场最好（平均 +2.9%、赚钱 61%）。
极端亏损（最差 -43%）靠仓位承受：△B 只买到总资金 20% 时，对总资金的影响约 -8.6%。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dingdi  # noqa: E402
from tdx_reader import read_day  # noqa: E402

IDX = [("上证", "sh000001"), ("沪深300", "sh000300"), ("上证50", "sh000016"), ("中证500", "sh000905"),
       ("中证1000", "sh000852"), ("深成", "sz399001"), ("创业板", "sz399006"), ("科创50", "sh000688")]
HOLD = 60


def trades() -> dict[str, list[float]]:
    rules = ["持有60天不止损", "跌5%止损", "跌8%止损", "跌12%止损", "1.5倍波动止损", "收盘跌破信号日最低1%", "破月BBI离场"]
    res = {k: [] for k in rules}
    for _, code in IDX:
        df = read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day")
        r = dingdi.compute(df)
        c, low = df.close.to_numpy(), df.low.to_numpy()
        sd = pd.Series(c).pct_change().rolling(250).std().to_numpy()
        brk = r["破月BBI"].to_numpy()
        last = -999
        for i in np.flatnonzero((r["底部一级"] | r["底部二级"]).to_numpy()):
            if df.index[i] < pd.Timestamp("2011-08-01") or i - last < 20 or i + HOLD >= len(c):
                continue
            last = i
            path = c[i + 1:i + HOLD + 1] / c[i] - 1

            def stop(level):
                hit = np.flatnonzero(path <= -level)
                return path[hit[0]] if len(hit) else path[-1]
            res["持有60天不止损"].append(path[-1])
            for s, k in ((0.05, "跌5%止损"), (0.08, "跌8%止损"), (0.12, "跌12%止损")):
                res[k].append(stop(s))
            res["1.5倍波动止损"].append(stop(1.5 * sd[i] * np.sqrt(20)))
            inv = np.flatnonzero(c[i + 1:i + HOLD + 1] < low[i] * 0.99)
            res["收盘跌破信号日最低1%"].append(path[inv[0]] if len(inv) else path[-1])
            bb = np.flatnonzero(brk[i + 1:i + 121])
            res["破月BBI离场"].append(c[i + 1 + bb[0]] / c[i] - 1 if len(bb) else c[min(i + 120, len(c) - 1)] / c[i] - 1)
    return res


if __name__ == "__main__":
    res = trades()
    print(f"共 {len(res['持有60天不止损'])} 笔")
    for k, v in res.items():
        v = np.array(v)
        print(f"{k:16s} 平均 {v.mean():+.1%}  中位 {np.median(v):+.1%}  赚钱 {(v > 0).mean():.0%}  "
              f"亏超10% {(v < -0.10).mean():.0%}  最差 {v.min():+.1%}")
