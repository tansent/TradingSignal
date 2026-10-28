"""按已有利润抬高止损线（移动止损、保本止损）能不能"永远不亏"（2026-10-20，用户转述的"防守"方法）。

沿用 research/stop_loss.py 的入场：8 个宽基 2011-08 起，每轮第一个 △B/★B（与上一笔间隔 ≥20 个交易日），信号当天收盘买入。
所有方法最长持有 250 天；止损按收盘触发、第二天开盘卖出（把跳空和滑点算进去）。
  不止损：持有 250 天；破月 BBI：现行规则；
  移动止损 X%：收盘从买入后最高收盘回落 X%；
  保本止损：浮盈曾到 5% 以后，收盘跌回买入价就卖；
  利润回吐一半：浮盈曾到 6% 以后，收盘回到最高浮盈的一半以下就卖。
"被洗出去"：卖出后 60 天内收盘又比卖出价高 5% 以上。

结果（290 笔）：
  破月 BBI（现行）平均 +5.0%、中位 +3.3%、赚钱 67%。
  移动止损 5/8/12%：平均 +1.3/+1.6/+2.3%，中位都是负的，三分之二的笔亏钱，止损后 65～70% 又涨回 5% 以上——
  最大亏损从 -39% 压到 -14～-19%，代价是大量小亏。
  保本止损：68% 的笔亏钱（保本线触发后 90% 第二天开盘已经低于买入价），浮盈没到 5% 的笔完全没保护，最差仍是 -39%。
  利润回吐一半：赚钱 73%、中位 +3.9%，平均 +4.3%，最差同样 -39%。
  滑点不是主要问题（指数第二天开盘比触发价平均差 0～0.4%），主要成本是止损线落在正常波动范围里被洗出去。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dingdi  # noqa: E402
from stop_loss import IDX  # noqa: E402
from tdx_reader import read_day  # noqa: E402

HOLD = 250


def exit_at(i, j, c, o):
    """第 j 天收盘触发，第 j+1 天开盘卖出；j 为 None 表示持有到期。返回 (收益, 卖出日, 滑点)。"""
    if j is None:
        k = min(i + HOLD, len(c) - 1)
        return c[k] / c[i] - 1, k, 0.0
    k = min(j + 1, len(c) - 1)
    return o[k] / c[i] - 1, k, o[k] / c[j] - 1


def trades() -> pd.DataFrame:
    rows = []
    for _, code in IDX:
        df = read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day")
        r = dingdi.compute(df)
        c, o = df.close.to_numpy(), df.open.to_numpy()
        brk = r["破月BBI"].to_numpy()
        last = -999
        for i in np.flatnonzero((r["底部一级"] | r["底部二级"]).to_numpy()):
            if df.index[i] < pd.Timestamp("2011-08-01") or i - last < 20 or i + HOLD >= len(c):
                continue
            last = i
            seg = c[i + 1:i + HOLD + 1]
            peak = np.maximum.accumulate(seg)
            gain = seg / c[i] - 1
            maxg = np.maximum.accumulate(gain)
            first = lambda m: i + 1 + int(np.flatnonzero(m)[0]) if m.any() else None  # noqa: E731
            rules = {"不止损，持有250天": None, "破月BBI离场（现行）": first(brk[i + 1:i + HOLD + 1])}
            for x in (0.05, 0.08, 0.12):
                rules[f"移动止损 {x:.0%}"] = first(seg <= peak * (1 - x))
            rules["保本止损（浮盈到5%后）"] = first((maxg >= 0.05) & (gain <= 0))
            rules["利润回吐一半（浮盈到6%后）"] = first((maxg >= 0.06) & (gain <= maxg / 2))
            for k, j in rules.items():
                ret, out, slip = exit_at(i, j, c, o)
                after = c[out + 1:out + 61]
                rows.append({"方法": k, "收益": ret, "天数": out - i, "滑点": slip, "止损": j is not None,
                             "被洗": j is not None and len(after) > 0 and after.max() >= o[out] * 1.05})
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)
    T = trades()
    g = T.groupby("方法", sort=False)
    out = pd.DataFrame({
        "平均": g.收益.mean().map("{:+.1%}".format), "中位": g.收益.median().map("{:+.1%}".format),
        "赚钱": g.收益.apply(lambda s: f"{(s > 0).mean():.0%}"), "亏钱": g.收益.apply(lambda s: f"{(s < 0).mean():.0%}"),
        "最差": g.收益.min().map("{:+.1%}".format), "平均持有天数": g.天数.mean().round(0).astype(int),
        "被洗出去": g.apply(lambda d: f"{d.被洗.sum() / max(d.止损.sum(), 1):.0%}", include_groups=False),
        "第二天开盘比触发价": g.apply(lambda d: f"{d[d.止损].滑点.mean():+.2%}" if d.止损.any() else "—", include_groups=False),
    })
    print(f"共 {T.groupby('方法').size().iloc[0]} 笔（每种方法同一批入场）")
    print(out.to_string())
    be = T[T.方法 == "保本止损（浮盈到5%后）"]
    hit = be[be.止损]
    print(f"\n保本止损触发 {len(hit)} 次，其中卖出价低于买入价（跳空跌穿保本线）{(hit.收益 < 0).mean():.0%}；"
          f"没触发保本的 {len(be) - len(hit)} 笔里亏钱的 {(be[~be.止损].收益 < 0).mean():.0%}（浮盈没到 5% 就一路跌，保本线从没生效）")


if __name__ == "__main__":
    main()
