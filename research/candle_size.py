"""顶部 K 线大、底部 K 线小？大涨之后的大跌能不能接？大涨大跌期间该不该买？（2026-10-20，用户的观察）

K 线大小 = 当天振幅（最高 - 最低）/ 前一天收盘，取 20 天平均。
一、顶和底：收盘价按 20% 的折返划波段（zigzag），比较每个顶、每个底之前 20 天的 K 线大小，以及该指数全部日子的中位数。
    上证 1993 年起（新浪），8 个宽基 2011-08 起（通达信）。
二、大跌日：当天跌幅超过 2 倍日波动率（前 250 天）。分三类：
    高位大跌 = 离一年高点 10% 以内，且之前 60 天涨了 15% 以上（"大涨后的大跌"）；
    低位大跌 = 离一年高点跌了 20% 以上；其他大跌。看收盘买入后 20/60/120 天。
三、每天按"K 线大小在过去 500 天的分位"（高三分之一 / 低三分之一）×"位置"（离一年高点 10% 以内 / 跌了 20% 以上）分组，
    看之后 60 天的涨跌和"60 天内再跌 10% 以上"的概率。8 个宽基 2011-08 起。

结果：
  一、顶和底之前 20 天的 K 线都比平时大：上证 1993 起顶 1.5 倍、底 1.4 倍（80%、84% 的顶/底大于平时）。
     底部不是"天天小 K 线"：恐慌底（2008-11、2015-08、2016-01、2024-02）K 线很大；只有阴跌见底（2012-12、2019-01）是小 K 线。
  二、大涨后的高位大跌接进去：8 个宽基 2011 起 96 次，60 天上涨 47%、120 天中位 -3.3%，46% 在 60 天内再跌 10% 以上（三类里最高）；
     但上证 1993 起 37 次是 62%、+17.7%（1996-97、2006-07 那种真牛市里大跌是回调）。近十几年的结构市里，大涨后的大跌多半还在顶部。
  三、离一年高点 10% 以内：K 线大的时候 60 天上涨 44%、中位 -1.3%、34% 再跌 10% 以上；K 线小的时候 63%、+3.3%、14%。
     跌了 20% 以上：K 线大（恐慌）56%、+1.9%；K 线小（阴跌）48%、-0.4%。
  我们的 B 点要求离 120 日高点回撤 ≥11 倍日波动率，1456 个里只有 22 个（1.5%，如 2018-02、2024-11）在离一年高点 10% 以内，
  本来就基本避开了高位大涨大跌的阶段。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from macd_monthly import ROOT  # noqa: E402
from stop_loss import IDX  # noqa: E402
from tdx_reader import read_day  # noqa: E402


def long_sh() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "index_long" / "sh000001.csv", parse_dates=["date"]).set_index("date")
    return d[d.index >= "1993-01-01"][["open", "high", "low", "close"]].astype(float)


def candle(d: pd.DataFrame) -> pd.Series:
    return ((d.high - d.low) / d.close.shift()).rolling(20).mean()


def zigzag(c: pd.Series, th: float = 0.20):
    C, pts = c.to_numpy(), []
    mode, hi_i, lo_i, ext_i = 0, 0, 0, 0
    for i in range(1, len(C)):
        if mode == 0:
            hi_i = i if C[i] > C[hi_i] else hi_i
            lo_i = i if C[i] < C[lo_i] else lo_i
            if C[i] >= C[lo_i] * (1 + th):
                pts.append(("底", lo_i)); mode, ext_i = 1, i
            elif C[i] <= C[hi_i] * (1 - th):
                pts.append(("顶", hi_i)); mode, ext_i = -1, i
        elif mode == 1:
            if C[i] > C[ext_i]:
                ext_i = i
            elif C[i] <= C[ext_i] * (1 - th):
                pts.append(("顶", ext_i)); mode, ext_i = -1, i
        else:
            if C[i] < C[ext_i]:
                ext_i = i
            elif C[i] >= C[ext_i] * (1 + th):
                pts.append(("底", ext_i)); mode, ext_i = 1, i
    return [(k, c.index[i]) for k, i in pts]


def part1(name: str, d: pd.DataFrame):
    k = candle(d)
    med = k.median()
    rows = []
    for kind, t in zigzag(d.close):
        v = k.loc[:t].iloc[-1]
        if pd.notna(v):
            rows.append({"类": kind, "日": t.date(), "K线/中位": v / med})
    R = pd.DataFrame(rows)
    g = R.groupby("类")["K线/中位"]
    print(f"  {name}：顶 {g.size().get('顶', 0)} 个，之前 20 天 K 线是平时的 {g.median().get('顶', np.nan):.1f} 倍"
          f"（{(R[R.类 == '顶']['K线/中位'] > 1).mean():.0%} 大于平时）；"
          f"底 {g.size().get('底', 0)} 个，{g.median().get('底', np.nan):.1f} 倍（{(R[R.类 == '底']['K线/中位'] > 1).mean():.0%} 大于平时）")
    return R.assign(指数=name)


def fwd(c: np.ndarray, i: int, h: int):
    return c[i + h] / c[i] - 1 if i + h < len(c) else np.nan


def part2_3(data: dict):
    ev, pan = [], []
    for name, d in data.items():
        c = d.close
        C = c.to_numpy()
        r = c.pct_change()
        sd = r.rolling(250).std().shift()
        hi = c.rolling(250, min_periods=200).max()
        dd = c / hi - 1
        up60 = c / c.shift(60) - 1
        k = candle(d)
        kp = k.rolling(500, min_periods=250).rank(pct=True)
        start = np.flatnonzero(c.index >= pd.Timestamp("2011-08-01"))[0] if name != "上证(1993起)" else 260
        for i in range(start, len(C) - 1):
            f20, f60, f120 = fwd(C, i, 20), fwd(C, i, 60), fwd(C, i, 120)
            lo60 = C[i + 1:i + 61].min() / C[i] - 1 if i + 60 < len(C) else np.nan
            if r.iloc[i] <= -2 * sd.iloc[i]:
                kind = ("高位大跌（大涨后）" if dd.iloc[i] >= -0.10 and up60.iloc[i] >= 0.15 else
                        "低位大跌" if dd.iloc[i] <= -0.20 else "其他大跌")
                ev.append({"指数": name, "类": kind, "f20": f20, "f60": f60, "f120": f120, "lo60": lo60})
            if name != "上证(1993起)" and pd.notna(kp.iloc[i]):
                pos = "离高点10%内" if dd.iloc[i] >= -0.10 else "跌了20%以上" if dd.iloc[i] <= -0.20 else None
                vol = "K线大" if kp.iloc[i] >= 2 / 3 else "K线小" if kp.iloc[i] <= 1 / 3 else None
                if pos and vol:
                    pan.append({"组": f"{pos}·{vol}", "f60": f60, "lo60": lo60})
    E = pd.DataFrame(ev)
    agg = lambda g: pd.Series({"次数": len(g), "20天涨": f"{(g.f20.dropna() > 0).mean():.0%}", "60天涨": f"{(g.f60.dropna() > 0).mean():.0%}",  # noqa: E731
                               "60天中位": f"{g.f60.median():+.1%}", "120天中位": f"{g.f120.median():+.1%}",
                               "60天内再跌10%以上": f"{(g.lo60.dropna() <= -0.10).mean():.0%}"})
    for scope, sub in (("8 个宽基 2011-08 起", E[E.指数 != "上证(1993起)"]), ("上证 1993 起", E[E.指数 == "上证(1993起)"])):
        print(f"\n== 二、大跌日收盘买入（{scope}）==")
        print(sub.groupby("类").apply(agg, include_groups=False).to_string())
    P = pd.DataFrame(pan)
    print("\n== 三、按 K 线大小 × 位置，之后 60 天（8 个宽基 2011-08 起，每天一个样本）==")
    print(P.groupby("组").apply(lambda g: pd.Series({"天数": len(g), "60天涨": f"{(g.f60.dropna() > 0).mean():.0%}",
                                                    "60天中位": f"{g.f60.median():+.1%}", "60天平均": f"{g.f60.mean():+.1%}",
                                                    "60天内再跌10%以上": f"{(g.lo60.dropna() <= -0.10).mean():.0%}"}),
                                include_groups=False).to_string())


def main():
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)
    data = {"上证(1993起)": long_sh()}
    for name, code in IDX:
        data[name] = read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day")[["open", "high", "low", "close"]]
    print("== 一、20% 波段的顶和底，之前 20 天的 K 线大小（和该指数所有日子的中位数比）==")
    R = pd.concat([part1(n, d[d.index >= ("1993-01-01" if n.startswith("上证(") else "2011-01-01")]) for n, d in data.items()])
    print(R[R.指数 == "上证(1993起)"].pivot_table(index="日", columns="类", values="K线/中位").map(
        lambda v: f"{v:.1f}" if pd.notna(v) else "").to_string())
    part2_3(data)


if __name__ == "__main__":
    main()
