"""国家队托底是不是阶段大底，还是只延缓下跌；下跌趋势里和牛市里有没有区别（2026-10-19，用户转述的说法）。

入场轮次沿用 research/national_team.py：上交所宽基 ETF 单日净申购 ≥ 阈值亿元，且沪深300 当天涨幅 < 2%，
相隔不超过 20 个交易日算同一轮。数据 2015 年起（份额数据）。主口径 ≥100 亿（13 轮），宽口径 ≥50 亿（25 轮，
早年的 50 亿不一定是国家队）。
每一轮从第一天收盘算起，看之后 120 个交易日：
  再跌 = 最低收盘 / 第一天收盘 - 1，几天后见低；"是阶段底"= 再跌不超过 3%；
  延缓 = 入场前 20 天的涨跌，对比入场后 20 天的涨跌；只看下跌中入场（入场前 20 天下跌）的轮次，排除牛市里的追涨申购；
  趋势 = 第一天沪深300 在年线上方还是下方。

结果（≥100 亿，走完 120 天的 10 轮）：
  只有 2024-01、2024-08、2025-04 三轮在 20 天内见底、之后最多再跌 3～4%；其余 7 轮沪深300 又跌了 5～32%，见低要 40～106 天。
  三次接近底部的都在年线下方（跌了很久之后）；年线上方（刚从高位下来）的轮次没有一次是阶段底：
  2015-07 -26%、2023-01 -8%、2024-12 -10%、2025-10 -5%，以及还没走完的 2026-06 -12%、2026-07 -11%。
  下跌中入场、年线下方的 5 轮，入场后 20 天全部比入场前 20 天好（中位 -4.5% → +6.6%），60 天后中位 +7.1%。
  之后 120 天再跌中位：上证50 -6%、沪深300 -9%、创业板指 -12%、中证1000 -14%——托的是大盘股，小盘、成长跌得更多。
  样本很少（年线上方 2 轮、下方 5 轮），只能当作方向。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "daily"))
import national_team as nt  # noqa: E402
from tdx_reader import read_day  # noqa: E402
from update import etf_net_creation  # noqa: E402

V = "D:/TDXapp/vipdoc"
IDX = {"沪深300": "sh000300", "上证50": "sh000016", "中证1000": "sh000852", "创业板指": "sz399006"}


def outcome(c: pd.Series, t0: pd.Timestamp, n: int = 120) -> dict:
    C = c.to_numpy()
    i = c.index.get_loc(t0)
    fut = C[i + 1:i + 1 + n]
    j = int(np.argmin(fut)) if len(fut) else 0
    low = fut[j] / C[i] - 1 if len(fut) else np.nan
    pre = (C[i] / C[i - 20]) ** (1 / 20) - 1
    post = (fut[j] / C[i]) ** (1 / (j + 1)) - 1 if len(fut) and low < 0 else 0.0
    r = {"再跌": min(low, 0.0), "几天见低": j + 1 if low < 0 else 0, "入场前20天": C[i] / C[i - 20] - 1,
         "前日均": pre, "后日均": post, "完整": len(fut) == n}
    for h in (20, 60, 120):
        r[f"r{h}"] = C[i + h] / C[i] - 1 if i + h < len(C) else np.nan
    return r


def run(thr: float) -> pd.DataFrame:
    nt.MIN_YI = thr
    net = etf_net_creation().net_yi
    data = {n: read_day(f"{V}/{c[:2]}/lday/{c}.day").close for n, c in IDX.items()}
    h = read_day(f"{V}/sh/lday/sh000300.day")
    ma250 = h.close.rolling(250).mean()
    rows = []
    for ep in nt.episodes(net, h):
        t0 = ep.index[0]
        r = {"开始": t0.date(), "天数": len(ep), "合计亿": round(ep.net.sum()),
             "趋势": "年线下方" if h.close[t0] < ma250[t0] else "年线上方"}
        r.update(outcome(h.close, t0))
        for n in ("上证50", "中证1000", "创业板指"):
            r[f"{n}再跌"] = outcome(data[n], t0)["再跌"]
        rows.append(r)
    return pd.DataFrame(rows)


def report(R: pd.DataFrame, label: str):
    pct = lambda v: f"{v:+.0%}" if pd.notna(v) else "  —"  # noqa: E731
    print(f"\n== {label}：{len(R)} 轮 ==")
    cols = ["开始", "天数", "合计亿", "趋势", "入场前20天", "r20", "再跌", "几天见低", "r60", "r120", "上证50再跌", "中证1000再跌", "创业板指再跌"]
    print(R[cols].to_string(index=False, formatters={k: pct for k in cols[4:] if k != "几天见低"}))
    T = R[R.完整 & (R.入场前20天 < 0)]
    print(f"  只看下跌中入场（入场前 20 天下跌）且走完 120 天的 {len(T)} 轮：")
    for k, g in T.groupby("趋势"):
        print(f"    {k} {len(g)} 轮：是阶段底（之后最多再跌 3%）{(g.再跌 > -0.03).sum()} 轮；再跌中位 {g.再跌.median():+.0%}；"
              f"入场前 20 天中位 {g.入场前20天.median():+.1%} → 入场后 20 天中位 {g.r20.median():+.1%}（{(g.r20 > g.入场前20天).sum()} 轮变好）；"
              f"60 天后中位 {g.r60.median():+.1%}")
    print("    之后 120 天再跌中位：" + "，".join(f"{n} {T[c].median():+.0%}" for n, c in
                                         (("沪深300", "再跌"), ("上证50", "上证50再跌"), ("中证1000", "中证1000再跌"), ("创业板指", "创业板指再跌"))))


def main():
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 250)
    report(run(100), "≥100 亿")
    report(run(50), "≥50 亿（宽口径）")


if __name__ == "__main__":
    main()
