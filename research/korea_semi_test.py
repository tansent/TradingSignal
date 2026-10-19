"""韩国集成电路出口增速的"二阶导"能不能提前预警 A 股半导体板块的顶部（2026-10-18）。

数据：research/korea_semi.py 下载的韩国 HS 8542 月度出口（美元）。t 月数据在 t+1 月月初公布，所以从 t+1 月第一个交易日起才用。
  增速 g = 最近 3 个月出口合计 ÷ 去年同期 3 个月合计 − 1（3 个月合计去掉单月噪声）；
  加速度 a = g − 3 个月前的 g。
  四种状态：扩张加速（g>0, a>0）、扩张放缓（g>0, a≤0，即"二阶导转负"）、收缩加深（g≤0, a≤0）、收缩改善（g≤0, a>0）。
  见顶信号：增速仍为正，但比最近 12 个月的最高增速低 15 个百分点以上，每轮只取第一次（与上一次间隔 ≥6 个月）。
看 A 股半导体（881319）、半导体设备（881321）、创业板指之后 3/6/12 个月的涨跌。

结果（UN Comtrade 只有 2013-01 ~ 2025-12，状态从 2014-03 起；月度样本互相重叠，实际只有 3~4 轮芯片周期）：
  创业板指：扩张放缓时之后 3/6/12 个月上涨只有 40%/42%/44%，中位 -4.7%/-6.5%/-4.7%，是四种状态里最差的；
    扩张加速时 53%/60%/53%，收缩加深时 62%/58%/54%。
  半导体：扩张放缓时 3 个月后上涨 47%、中位 -2.4%（扩张加速 60%、+2.4%；收缩加深 75%、+8.4%，周期谷底反而最好）。
  半导体设备：四种状态都偏涨（国产替代的长期趋势盖过周期），看不出规律。
  "见顶信号"（增速比 12 个月高点低 15 个百分点）只有 3 次（2018-03、2022-06、2025-02），太少，不能下结论。
2026 年的数据联合国数据库还没有；韩国产业通商资源部公布的半导体出口同比：上半年 +163%、7 月 +179%、8 月 +209%、9 月 +263%，
仍在"扩张加速"（金额主要来自存储涨价）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tdx_reader import read_day  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TARGETS = [("半导体", "sh881319"), ("半导体设备", "sh881321"), ("创业板指", "sz399006")]
H = {"3个月": 63, "6个月": 126, "12个月": 250}


def monthly() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "korea_semi_exports.csv", dtype={"period": str}).dropna()
    d["month"] = pd.PeriodIndex(d.period, freq="M")
    s = d.set_index("month").usd.sort_index()
    s3 = s.rolling(3).sum()
    g = s3 / s3.shift(12) - 1
    a = g - g.shift(3)
    peak = g.rolling(12, min_periods=6).max()
    m = pd.DataFrame({"usd": s, "g": g, "a": a, "peak12": peak}).dropna(subset=["g"])
    m["state"] = np.select([(m.g > 0) & (m.a > 0), (m.g > 0) & (m.a <= 0), (m.g <= 0) & (m.a <= 0)],
                           ["扩张加速", "扩张放缓", "收缩加深"], "收缩改善")
    m["avail"] = (m.index + 1).to_timestamp()           # 下个月月初才能用
    return m


def signals(m: pd.DataFrame) -> pd.DatetimeIndex:
    cand = m[(m.g > 0) & (m.g <= m.peak12 - 0.15)]
    out, last = [], None
    for t, r in cand.iterrows():
        if last is None or (t - last).n >= 6:
            out.append(r.avail)
        last = t
    return pd.DatetimeIndex(out)


def forward(c: pd.Series, when: pd.Timestamp) -> dict:
    i = c.index.searchsorted(when)
    if i >= len(c):
        return {}
    out = {"entry": c.index[i]}
    for k, n in H.items():
        out[k] = c.iloc[i + n] / c.iloc[i] - 1 if i + n < len(c) else np.nan
    return out


def main():
    m = monthly()
    print(f"韩国集成电路出口：{m.index[0]} → {m.index[-1]}，最近 6 个月增速（3 个月合计同比）与加速度：")
    print(m[["g", "a", "state"]].tail(6).assign(g=lambda d: d.g.map("{:+.0%}".format), a=lambda d: d.a.map("{:+.0%}".format)).to_string())
    for name, code in TARGETS:
        c = read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day").close
        rows = [dict(state=r.state, **forward(c, r.avail)) for t, r in m.iterrows() if r.avail >= c.index[0]]
        R = pd.DataFrame(rows).dropna(subset=["entry"])
        print(f"\n== {name}：按出口状态分（每月一个样本）==")
        for st in ("扩张加速", "扩张放缓", "收缩加深", "收缩改善"):
            g = R[R.state == st]
            print(f"  {st}  {len(g):3d} 个月  " + "  ".join(
                f"{k}后涨 {(g[k].dropna() > 0).mean() * 100:3.0f}% 中位 {g[k].median() * 100:+5.1f}%" for k in H))
        b = pd.DataFrame([forward(c, t) for t in signals(m) if t >= c.index[0]])
        print(f"  见顶信号（增速比 12 个月高点低 15 个百分点以上）{len(b)} 次：" + "  ".join(
            f"{k}后跌 {(b[k].dropna() < 0).mean() * 100:3.0f}% 中位 {b[k].median() * 100:+5.1f}%" for k in H))
        print("    日期：", [d.strftime("%Y-%m") for d in b.entry])


if __name__ == "__main__":
    main()
