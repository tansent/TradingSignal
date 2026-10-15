"""跌破整理箱体下沿之后会怎样？（宽基指数、通达信研究行业指数、半导体）

箱体：信号日之前 N=60 个交易日的最高价（箱顶）和最低价（箱底）；只有箱体高度不超过
1.6 × 日波动率 × √N（即比同期随机波动的正常幅度还窄）才算"整理箱体"，单边下跌不算。
跌破：收盘价低于箱底；突破：收盘价高于箱顶。同一指数 20 个交易日内只算第一次。

检验：
1. 跌破/突破之后 20/60/120 天的涨跌，对比同一批指数所有交易日（基准）；
2. 跌破之后多久收回箱底（假跌破）、120 天内能否收盘站上箱顶；
3. 跌破后反弹回箱体内（收盘重回箱底之上）那天算起，之后的表现——箱体是否成了"套牢盘压力"；
4. 跌破后 30 天内出现 △B/★B，从 B 那天算起的表现，对比其他时候的 B。
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dingdi  # noqa: E402
from tdx_reader import read_day  # noqa: E402

N, K, GAP = 60, 1.6, 20
BROAD = [("上证50", "sh000016"), ("沪深300", "sh000300"), ("中证500", "sh000905"), ("中证1000", "sh000852"),
         ("上证指数", "sh000001"), ("深证成指", "sz399001"), ("创业板指", "sz399006"), ("科创50", "sh000688")]
SEMI = [("半导体(研究行业)", "sh881319"), ("半导体设备(研究行业)", "sh881321"), ("半导体(通达信行业)", "sh880491"),
        ("半导体设备ETF 159516", "sz159516"), ("半导体设备ETF 561980", "sh561980")]


def industries() -> list[tuple[str, str]]:
    """通达信研究行业二级指数（约 128 个，2011 年起，带涨跌家数）。"""
    txt = subprocess.run(["iconv", "-f", "GBK", "-t", "UTF-8", "D:/TDXapp/T0002/hq_cache/tdxzs3.cfg"],
                         capture_output=True, text=True, encoding="utf-8").stdout
    out = []
    for line in txt.splitlines():
        f = line.split("|")
        if len(f) >= 6 and f[2] == "12" and len(f[5]) == 5:
            out.append((f[0], "sh" + f[1]))
    return out


def split_adjust(d: pd.DataFrame) -> pd.DataFrame:
    """ETF 份额拆分（如 159516 在 2026-07-10 一拆二）在通达信日线里是未复权的价格跳空，按整数比例前复权。"""
    d = d.copy()
    ratio = d.close.shift() / d.open
    for t in ratio.index[(ratio > 1.6) | (ratio < 0.6)]:
        f = round(ratio[t]) if ratio[t] > 1 else 1 / round(1 / ratio[t])
        d.loc[d.index < t, ["open", "high", "low", "close"]] /= f
    return d


def load(code: str) -> pd.DataFrame | None:
    p = Path(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day")
    if not p.exists():
        return None
    d = read_day(str(p))
    if code[2:4] in ("15", "51", "56", "58"):
        d = split_adjust(d)
    return d if len(d) > 400 else None


def fwd(c: pd.Series, k: int, n: int) -> float:
    return c.iloc[k + n] / c.iloc[k] - 1 if k + n < len(c) else np.nan


def analyze(name: str, d: pd.DataFrame, start="2012-01-01"):
    c, h, l = d.close, d.high, d.low
    top, low = h.rolling(N).max().shift(1), l.rolling(N).min().shift(1)
    vol = np.log(c).diff().rolling(250).std().shift(1)
    isbox = (top / low - 1) <= K * vol * np.sqrt(N)
    r = dingdi.compute(d)
    bsig = (r["底部一级"] | r["底部二级"]).to_numpy()
    star = r["底部二级"].to_numpy()
    ok = (c.index >= start)
    ev, base = [], []
    for kind, m in (("跌破箱底", (c < low) & isbox & ok), ("突破箱顶", (c > top) & isbox & ok)):
        last = -10**9
        for k in np.flatnonzero(m.to_numpy()):
            if k - last <= GAP:
                continue
            last = k
            L, T = low.iloc[k], top.iloc[k]
            e = {"名称": name, "日期": c.index[k], "事件": kind, "箱底": L, "箱顶": T, "箱高": T / L - 1,
                 "20天": fwd(c, k, 20), "60天": fwd(c, k, 60), "120天": fwd(c, k, 120),
                 "60天内再跌": l.iloc[k + 1:k + 61].min() / c.iloc[k] - 1 if k + 1 < len(c) else np.nan}
            if kind == "跌破箱底" and k + 120 < len(c):
                after = c.iloc[k + 1:k + 121]
                e["5天内收回箱底"] = bool((after.iloc[:5] > L).any())
                e["20天内收回箱底"] = bool((after.iloc[:20] > L).any())
                e["120天内站上箱顶"] = bool((after > T).any())
                # 反弹回箱体内：60 天内第一次收盘重回箱底之上
                back = np.flatnonzero((after.iloc[:60] > L).to_numpy())
                if len(back):
                    j = k + 1 + back[0]
                    e["回箱日"] = c.index[j]
                    e["回箱后60天"] = fwd(c, j, 60)
                    e["回箱后20天内又跌破"] = bool((c.iloc[j + 1:j + 21] < L).any())
                    e["回箱后120天内站上箱顶"] = bool((c.iloc[j + 1:j + 121] > T).any())
                # 跌破后 30 天内第一次出现 B（且仍在箱顶下方）
                bb = [j for j in range(k, min(k + 31, len(c))) if bsig[j] and c.iloc[j] < T]
                if bb:
                    j = bb[0]
                    e.update({"B日": c.index[j], "B是★B": bool(star[j]), "B后20天": fwd(c, j, 20), "B后60天": fwd(c, j, 60),
                              "B后120天": fwd(c, j, 120), "B后60天内再跌": l.iloc[j + 1:j + 61].min() / c.iloc[j] - 1,
                              "B后120天内站上箱顶": bool((c.iloc[j + 1:j + 121] > T).any())})
            ev.append(e)
    # 基准：所有交易日；以及"其他时候的 B"（不在箱体跌破后 30 天内），同类 20 天内只算第一次
    down_days = [c.index.get_loc(e["日期"]) for e in ev if e["事件"] == "跌破箱底"]
    near = np.zeros(len(c), bool)
    for k in down_days:
        near[k:k + 31] = True
    last = -10**9
    for k in np.flatnonzero(bsig & ok):
        first, last = k - last > GAP, k
        if first and not near[k]:
            base.append({"名称": name, "类型": "其他时候的B", "20天": fwd(c, k, 20), "60天": fwd(c, k, 60), "120天": fwd(c, k, 120),
                         "60天内再跌": l.iloc[k + 1:k + 61].min() / c.iloc[k] - 1 if k + 1 < len(c) else np.nan})
    all_days = pd.DataFrame({"20天": c.shift(-20) / c - 1, "60天": c.shift(-60) / c - 1, "120天": c.shift(-120) / c - 1})[ok]
    return ev, base, all_days.assign(名称=name, 类型="所有交易日")


def run(universe):
    E, B, A = [], [], []
    for name, code in universe:
        d = load(code)
        if d is None:
            continue
        ev, base, alld = analyze(name, d)
        E += ev
        B += base
        A.append(alld)
    return pd.DataFrame(E), pd.DataFrame(B), pd.concat(A)


def rate(x):
    x = x.dropna()
    return (x > 0).mean() if len(x) else np.nan


def ret_row(x: pd.DataFrame, cols=("20天", "60天", "120天"), dd="60天内再跌") -> dict:
    out = {"次数": len(x)}
    for col in cols:
        out[f"{col}上涨比例"] = rate(x[col])
        out[f"{col}中位"] = x[col].median()
    if dd in x:
        out["60天内再跌(中位)"] = x[dd].median()
    return out


def report(label, universe):
    E, B, A = run(universe)
    dn, up = E[E["事件"] == "跌破箱底"], E[E["事件"] == "突破箱顶"]
    print(f"\n######## {label}（{E['名称'].nunique()} 个，2012 年起；箱高中位 {dn['箱高'].median():.1%}）")
    t = pd.DataFrame([ret_row(A.assign(**{"60天内再跌": np.nan})) | {"次数": len(A)},
                      ret_row(dn), ret_row(up)], index=["所有交易日（基准）", "跌破箱底", "突破箱顶"])
    print(t.drop(columns=["60天内再跌(中位)"]).round(3).to_string())
    dn2 = dn.dropna(subset=["120天内站上箱顶"])
    print(f"跌破后：5 天内收回箱底 {dn2['5天内收回箱底'].mean():.0%}，20 天内收回 {dn2['20天内收回箱底'].mean():.0%}，"
          f"120 天内收盘站上箱顶 {dn2['120天内站上箱顶'].mean():.0%}；60 天内再跌中位 {dn['60天内再跌'].median():.1%}")
    rb = dn2.dropna(subset=["回箱日"])
    print(f"反弹回箱体内（{len(rb)} 次）：之后 20 天内又跌破箱底 {rb['回箱后20天内又跌破'].mean():.0%}，"
          f"120 天内站上箱顶 {rb['回箱后120天内站上箱顶'].mean():.0%}，回箱后 60 天上涨比例 {rate(rb['回箱后60天']):.0%}、中位 {rb['回箱后60天'].median():+.1%}")
    bb = dn.dropna(subset=["B日"])
    if len(bb):
        ren = {"B后20天": "20天", "B后60天": "60天", "B后120天": "120天", "B后60天内再跌": "60天内再跌"}
        fromb = lambda x: x[list(ren)].rename(columns=ren)  # noqa: E731
        rows = {"跌破后 30 天内的 B": ret_row(fromb(bb))}
        st = bb[bb["B是★B"].astype(bool)]
        if len(st):
            rows["  其中 ★B"] = ret_row(fromb(st))
        rows["其他时候的 B"] = ret_row(B)
        print(pd.DataFrame(rows).T.round(3).to_string())
        x = bb.dropna(subset=["B后120天"])
        print(f"跌破后的 B：120 天内收盘站上箱顶 {x['B后120天内站上箱顶'].mean():.0%}")
    return E


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    report("宽基指数", BROAD)
    report("通达信研究行业二级指数", industries())
    E = report("半导体（行业指数和半导体设备 ETF）", SEMI)
    print("\n半导体设备(研究行业) 历次跌破箱底：")
    x = E[(E["名称"] == "半导体设备(研究行业)") & (E["事件"] == "跌破箱底")]
    print(x[["日期", "箱底", "箱顶", "箱高", "20天", "60天", "120天", "60天内再跌", "120天内站上箱顶"]].round(3).to_string(index=False))
