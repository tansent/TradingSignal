"""反弹大阳线那天的"最强板块"之后还强不强？它破位是不是这波反弹结束的信号？（2026-10-28，用户的盘面想法）

起涨日 T：上证当天涨 ≥2%，且前一天收盘比 60 日最高收盘低 ≥7%（"反弹大阳线"，如 2026-10-23），相隔 ≥20 个交易日。
T 当天收盘给每个板块打分（只在收盘站上 5 日线的板块里排）：当天涨幅、阳线实体（收/开-1）、成交额 / 前 20 日均额（资金）、
上涨家数比例（情绪），四项横向排名取平均，第一名 = 最强板块，前 5 名 = 最强组。
一、持续性：最强组之后 5/20/60 天比全部板块中位数多涨多少，跑赢一半板块的比例。
二、风向标：最强板块"破位"（收盘跌破 T 当天最低价，或连续 2 天收在 20 日线下方）之后，上证 20 天怎么走；
    和上证自己同样破位之后比；最强板块破位是否早于上证。
板块：125 个通达信研究行业（2012 起，主测）；269 个概念板块（2019 起，概念板块多在题材火了以后才建，只作参考）。
研究行业 2022 年以前没有涨跌家数，那时只用前三项排名。

结果（行业 37 个起涨日；概念 16 个）：
  持续性：最强 1 个之后 5/20/60 天比中位板块多涨 +0.8% / +0.0% / +3.5%，跑赢一半板块 57% / 41% / 57%；
  最强 5 个 +0.5% / +1.0% / +1.2%（49% / 54% / 50%）。概念：最强 1 个 -0.3% / +0.3% / -1.1%。基本没有持续性。
  风向标：最强板块破位反而比上证晚 1～2 天（中位）；它连续 2 天收在 20 日线下方后，上证 20 天中位 -0.8%、下跌 58%（36 次），
  上证自己破位后 +1.6%、43%；跌破起涨日最低价后 +0.5% 对 +2.8%。概念上没有差别。样本小，不能当规律。
  2026-10-23（上证收盘只涨 0.9%，但从最低点拉起 116 点）：行业排名前两位电机制造、自动化设备，概念里减速器第 2、外骨骼机器人第 3、
  人形机器人第 7；到 10-27 这些板块都跌了 2～5%。
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tdx_reader import read_day  # noqa: E402

V = "D:/TDXapp/vipdoc"


def boards(kind: str) -> dict[str, pd.DataFrame]:
    txt = subprocess.run(["iconv", "-f", "GBK", "-t", "UTF-8", "D:/TDXapp/T0002/hq_cache/tdxzs3.cfg"],
                         capture_output=True, text=True, encoding="utf-8").stdout
    rows = [f for f in (l.split("|") for l in txt.splitlines()) if len(f) >= 6]
    pick = [f for f in rows if (f[2] == "12" and len(f[5]) == 5)] if kind == "行业" else [f for f in rows if f[2] == "4"]
    out = {}
    for f in pick:
        try:
            d = read_day(f"{V}/sh/lday/sh{f[1]}.day")
        except Exception:  # noqa: BLE001
            continue
        if len(d) > 120:
            out[f[0]] = d
    return out


def events(sh: pd.DataFrame, start="2012-01-01"):
    c = sh.close
    r = c.pct_change()
    dd = c.shift() / c.shift().rolling(60).max() - 1
    out, last = [], None
    for t in sh.index[(r >= 0.02) & (dd <= -0.07) & (sh.index >= start)]:
        if last is None or sh.index.get_loc(t) - sh.index.get_loc(last) >= 20:
            out.append(t)
            last = t
    return out


def metrics(B: dict, t) -> pd.DataFrame:
    """t 当天收盘站上 5 日线的板块：涨幅、阳线实体、成交额 / 前 20 日均额、上涨家数比例。"""
    rows = {}
    for n, d in B.items():
        if t not in d.index:
            continue
        i = d.index.get_loc(t)
        if i < 25:
            continue
        c, o, a = d.close.iloc[i], d.open.iloc[i], d.amount
        if c <= d.close.iloc[i - 4:i + 1].mean():
            continue
        adv, dec = d.adv.iloc[i], d.dec.iloc[i]
        rows[n] = {"涨幅": c / d.close.iloc[i - 1] - 1, "实体": c / o - 1, "量比": a.iloc[i] / a.iloc[i - 20:i].mean(),
                   "上涨比例": adv / (adv + dec) if adv + dec > 0 else np.nan}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=["涨幅", "实体", "量比", "上涨比例"])


def score(B: dict, t, S: pd.DataFrame | None = None) -> pd.Series:
    S = (metrics(B, t) if S is None else S).dropna(subset=["涨幅", "实体", "量比"])
    if S["上涨比例"].isna().mean() > 0.5:          # 2022 年以前研究行业没有涨跌家数，只用前三项
        S = S.drop(columns="上涨比例")
    return S.astype(float).rank(pct=True).mean(axis=1).sort_values(ascending=False)


def today(kind: str, k: int = 3) -> dict:
    """日报用：最近一个交易日的综合排名前 k 个板块（只是描述，不预测）。"""
    B = boards(kind)
    t = max(d.index[-1] for d in B.values())
    S = metrics(B, t)
    sc = score(B, t, S) if len(S) else pd.Series(dtype=float)
    top = [{"name": n, "pct": float(S.loc[n, "涨幅"]) * 100, "vol": float(S.loc[n, "量比"]),
            "up": None if pd.isna(S.loc[n, "上涨比例"]) else float(S.loc[n, "上涨比例"]) * 100} for n in sc.index[:k]]
    return {"date": t, "top": top, "above": len(S), "total": sum(1 for d in B.values() if t in d.index)}


def fwd(B, names, t, h):
    out = {}
    for n in names:
        d = B[n]
        i = d.index.get_loc(t) if t in d.index else None
        if i is not None and i + h < len(d):
            out[n] = d.close.iloc[i + h] / d.close.iloc[i] - 1
    return pd.Series(out)


def broke(d: pd.DataFrame, t, n=60):
    """T 之后 n 天内第一次破位的日期：收盘跌破 T 当天最低价；连续 2 天收在 20 日线下方。"""
    i = d.index.get_loc(t)
    seg = d.iloc[i + 1:i + n + 1]
    low = seg.index[seg.close < d.low.iloc[i]]
    ma20 = d.close.rolling(20).mean()
    below = (d.close < ma20).iloc[i + 1:i + n + 1]
    two = below & below.shift(1, fill_value=False)
    return {"跌破起涨日最低": low[0] if len(low) else None, "连续2天在20日线下": two.index[two][0] if two.any() else None}


def main(kind="行业", start="2012-01-01"):
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 220)
    sh = read_day(f"{V}/sh/lday/sh000001.day")
    B = boards(kind)
    ev = events(sh, start)
    print(f"== {kind}（{len(B)} 个），起涨日 {len(ev)} 个：{'、'.join(t.strftime('%Y-%m-%d') for t in ev)} ==")
    pers, rows = [], []
    C = sh.close
    for t in ev:
        sc = score(B, t)
        if len(sc) < 10:
            continue
        top1, top5 = sc.index[0], list(sc.index[:5])
        rec = {"起涨日": t.date(), "最强": top1, "前5": "、".join(top5)}
        for h in (5, 20, 60):
            f = fwd(B, list(B), t, h)
            if len(f) < 10:
                continue
            ex = f - f.median()
            pers.append({"h": h, "组": "最强1", "超额": ex.get(top1, np.nan), "跑赢": float(ex.get(top1, np.nan) > 0) if top1 in ex else np.nan})
            for n in top5:
                if n in ex:
                    pers.append({"h": h, "组": "最强5", "超额": ex[n], "跑赢": float(ex[n] > 0)})
            rest = [n for n in sc.index[-int(len(sc) * 0.2):] if n in ex]
            for n in rest:
                pers.append({"h": h, "组": "站上5日线里最弱20%", "超额": ex[n], "跑赢": float(ex[n] > 0)})
        it = sh.index.get_loc(t)
        lb, sb = broke(B[top1], t), broke(sh, t)
        for k in lb:
            d0 = lb[k]
            if d0 is not None and sh.index.get_loc(d0) + 20 < len(sh):
                j = sh.index.get_loc(d0)
                rec[f"最强{k}"] = f"{d0.date()} 上证20天{C.iloc[j + 20] / C.iloc[j] - 1:+.1%}"
            else:
                rec[f"最强{k}"] = "没破" if d0 is None else f"{d0.date()}"
            s0 = sb[k]
            rec[f"上证{k}"] = "没破" if s0 is None else f"{s0.date()}"
            if d0 is not None and sh.index.get_loc(d0) + 20 < len(sh):
                j = sh.index.get_loc(d0)
                rows.append({"破位": k, "谁": "最强板块", "上证20天": C.iloc[j + 20] / C.iloc[j] - 1,
                             "早于上证天数": (sh.index.get_loc(s0) - j) if s0 is not None else np.nan})
            if s0 is not None and sh.index.get_loc(s0) + 20 < len(sh):
                j = sh.index.get_loc(s0)
                rows.append({"破位": k, "谁": "上证自己", "上证20天": C.iloc[j + 20] / C.iloc[j] - 1, "早于上证天数": np.nan})
        rec["上证T后60天"] = f"{C.iloc[it + 60] / C.iloc[it] - 1:+.1%}" if it + 60 < len(C) else "—"
        print("  " + "  ".join(f"{k}:{v}" for k, v in rec.items()))
    P = pd.DataFrame(pers)
    print("\n一、持续性（比全部板块中位数多涨，平均 / 跑赢一半板块的比例）")
    print(P.groupby(["组", "h"]).agg(样本=("超额", "size"), 平均超额=("超额", lambda x: f"{x.mean():+.1%}"),
                                     跑赢=("跑赢", lambda x: f"{x.mean():.0%}")).unstack("h").to_string())
    R = pd.DataFrame(rows)
    print("\n二、破位之后上证 20 天（中位 / 下跌比例），最强板块破位比上证早几天（中位）")
    print(R.groupby(["破位", "谁"]).agg(次数=("上证20天", "size"), 中位=("上证20天", lambda x: f"{x.median():+.1%}"),
                                        下跌=("上证20天", lambda x: f"{(x < 0).mean():.0%}"),
                                        早几天=("早于上证天数", lambda x: f"{x.median():.0f}" if x.notna().any() else "—")).to_string())


if __name__ == "__main__":
    main("行业", "2012-01-01")
    main("概念", "2019-01-01")
