"""牛市主线能不能在早期认出来（2026-10-20，用户提问）。

做法：125 个通达信行业指数。以上证 ★B 每轮第一天为"大盘见底"的时点 T（当时就知道，不用事后挑底），
在 T 之后 40 个交易日（约两个月，"牛市初期"）用当时已知的信息挑板块，看之后 250 个交易日（T+40 → T+290）的表现。
"领涨板块"= 之后 250 天涨幅排前 15% 的板块（约 19 个），随便挑命中率就是 15%。
候选迹象（各挑前 15%）：
  早期领涨：T → T+40 涨幅最大；
  率先创新高：T → T+40 之间收盘创一年新高（全部入选，不限 15%）；
  底部抗跌：T 当天离一年高点跌得最少；
  熊市相对强度：T 之前 120 天涨幅最大；
  资金流入：T → T+40 成交额占全部行业的比例 ÷ T 之前一年的占比，最大。
另外做月度面板：每个月末按 3 个月、6 个月涨幅，近 20 天是否创一年新高，成交占比变化分三档，看之后 12 个月超额。

结果（11 次见底，2012-09 到 2025-04）：
  五种迹象挑中之后一年领涨板块的比例 11%～15%，和随便挑（15%）一样——认不出主线。
  底部抗跌、熊市相对强度、资金流入挑的板块之后一年平均超额 +3.6%、+3.9%、+2.9%（各 7/11 次为正），只是偏强一点；
  早期领涨（-0.2%）、率先创新高（+0.5%）没有用。2018-10 之后的半导体、消费电子，2022 年的游戏，事前都没被挑出来。
  月度面板：6 个月涨幅高的三分之一之后 12 个月超额 +0.9%、低的 -1.5%；近 20 天创新高 +1.0% 对 -0.6%；
  成交占比上升的反而略差（-0.4% 对 +0.4%，资金涌入常是拥挤）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from point_line_surface import cfg  # noqa: E402
from strategy_backtest import load_market, signals  # noqa: E402
from tdx_reader import read_day  # noqa: E402

V = "D:/TDXapp/vipdoc"
EARLY, AFTER, TOP = 40, 250, 0.15


def load():
    cs, am = {}, {}
    for f in cfg():
        if len(f[5]) != 5:
            continue
        try:
            d = read_day(f"{V}/sh/lday/sh{f[1]}.day")
        except Exception:  # noqa: BLE001
            continue
        if len(d) > 800:
            cs[f[0]], am[f[0]] = d.close, d.amount
    C = pd.DataFrame(cs).sort_index()
    A = pd.DataFrame(am).sort_index().reindex(C.index)
    return C[C.index >= "2011-01-01"], A[A.index >= "2011-01-01"]


def bottoms():
    s = signals("sh000001", load_market(), 200.0)
    out, last = [], None
    for t in s.index[s.b2.to_numpy()]:
        if t >= pd.Timestamp("2012-01-01") and (last is None or s.index.get_loc(t) - s.index.get_loc(last) > 20):
            out.append(t)
        last = t
    return out


def events(C, A):
    share = A.div(A.sum(axis=1), axis=0)
    hi250 = C.rolling(250, min_periods=200).max()
    rows, examples = [], []
    for T in bottoms():
        i = C.index.get_loc(T)
        if i + EARLY + AFTER >= len(C):
            continue
        t1, t2 = C.index[i + EARLY], C.index[i + EARLY + AFTER]
        out = C.loc[t2] / C.loc[t1] - 1
        out = (out - out.median()).dropna()
        lead = set(out.nlargest(int(len(out) * TOP)).index)
        k = int(len(out) * TOP)
        picks = {
            "早期领涨": (C.loc[t1] / C.loc[T] - 1).dropna().nlargest(k).index,
            "率先创新高": C.columns[(C.iloc[i + 1:i + EARLY + 1] >= hi250.shift(1).iloc[i + 1:i + EARLY + 1]).any()],
            "底部抗跌": (C.loc[T] / hi250.loc[T] - 1).dropna().nlargest(k).index,
            "熊市相对强度": (C.loc[T] / C.iloc[i - 120] - 1).dropna().nlargest(k).index,
            "资金流入": (share.iloc[i + 1:i + EARLY + 1].mean() / share.iloc[i - 250:i].mean()).dropna().nlargest(k).index,
        }
        for name, idx in picks.items():
            idx = [x for x in idx if x in out.index]
            rows.append({"见底": T.date(), "迹象": name, "入选": len(idx), "命中": len(lead & set(idx)),
                         "超额": out[idx].mean() if idx else np.nan})
        top5 = out.nlargest(5).index
        examples.append(f"  {T.date()} 之后一年领涨：" + "、".join(
            f"{n}{'(' + ','.join(k[:2] for k, v in picks.items() if n in v) + ')' if any(n in v for v in picks.values()) else ''}" for n in top5))
    return pd.DataFrame(rows), examples


def panel(C, A):
    share = A.div(A.sum(axis=1), axis=0)
    hi250 = C.rolling(250, min_periods=200).max()
    newhi = (C >= hi250).rolling(20).max()
    flow = share.rolling(60).mean() / share.rolling(250).mean()
    ends = C.groupby(C.index.to_period("M")).apply(lambda x: x.index.max())
    ends = ends[ends >= pd.Timestamp("2012-01-01")].iloc[:-1].tolist()      # 最后一个月没走完
    feats = {"3个月涨幅": lambda t, k: C.loc[t] / C.iloc[k - 60] - 1, "6个月涨幅": lambda t, k: C.loc[t] / C.iloc[k - 120] - 1,
             "成交占比变化": lambda t, k: flow.loc[t]}
    rows = []
    for t, t12 in zip(ends[:-12], ends[12:]):
        k = C.index.get_loc(t)
        f = C.loc[t12] / C.loc[t] - 1
        x = (f - f.mean()).dropna()
        for name, fn in feats.items():
            v = fn(t, k).reindex(x.index).dropna()
            q = pd.qcut(v.rank(method="first"), 3, labels=["低", "中", "高"])
            for lab in ("低", "高"):
                rows.append({"迹象": name, "档": lab, "超额": x[q[q == lab].index].mean()})
        nh = newhi.loc[t].reindex(x.index)
        rows.append({"迹象": "近20天创一年新高", "档": "高", "超额": x[nh == 1].mean()})
        rows.append({"迹象": "近20天创一年新高", "档": "低", "超额": x[nh == 0].mean()})
    R = pd.DataFrame(rows)
    return R.groupby(["迹象", "档"]).超额.agg(["mean", lambda s: (s > 0).mean()])


def main():
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 220)
    C, A = load()
    R, ex = events(C, A)
    print(f"== 一、上证 ★B 见底后 {EARLY} 天挑板块，看之后 {AFTER} 天（{R.见底.nunique()} 次见底；随便挑命中率 {TOP:.0%}）==")
    g = R.groupby("迹象", sort=False)
    print(pd.DataFrame({"平均入选": g.入选.mean().round(1), "命中率": (g.命中.sum() / g.入选.sum()).map("{:.0%}".format),
                        "之后一年超额": g.超额.mean().map("{:+.1%}".format),
                        "超额为正的次数": g.超额.apply(lambda s: f"{(s > 0).sum()}/{s.notna().sum()}")}).to_string())
    print("\n".join(ex))
    print("\n== 二、月度面板：按迹象分档，之后 12 个月超额（平均 / 为正的月份比例）==")
    P = panel(C, A)
    print(P.rename(columns={"mean": "平均超额", "<lambda_0>": "为正的月份"}).to_string(
        formatters={"平均超额": "{:+.1%}".format, "为正的月份": "{:.0%}".format}))


if __name__ == "__main__":
    main()
