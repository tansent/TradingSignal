"""点线面：相关指数（面=大盘、线=所属行业/板块）已跌破整理箱体、而目标（点）还在自己的箱体内时，目标之后会怎样？

箱体和跌破的定义与 box_break.py 相同（前 60 个交易日的高低点，高度不超过 1.6×日波动率×√60；收盘跌破箱底）。
- 相关指数"刚跌破"：5 个交易日内出现过跌破，且收盘仍在那次的箱底下方。同一相关指数 20 个交易日内只取第一天。
  变体"创60日新低"：不要求箱体，收盘低于前 60 日最低价就算（像现在的科创50，是下跌趋势而不是箱体）。
- 目标"还没跌破"：目标处在整理箱体中、收盘在箱底之上，且 20 个交易日内没有跌破过。
对比组：相关指数正常（20 天内没跌破、收盘在自己的 60 日低点之上）时，同样还在箱体内的目标（逐日）。
结果看目标之后 20/60 天的涨跌、20/60 天内跟着跌破自己箱底的比例、60 天内最大回撤、相对相关指数的 60 天超额。
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from box_break import GAP, K, N, load  # noqa: E402

START = "2012-01-01"


def cfg() -> list[list[str]]:
    txt = subprocess.run(["iconv", "-f", "GBK", "-t", "UTF-8", "D:/TDXapp/T0002/hq_cache/tdxzs3.cfg"],
                         capture_output=True, text=True, encoding="utf-8").stdout
    return [f for f in (line.split("|") for line in txt.splitlines()) if len(f) >= 6 and f[2] == "12"]


def state(d: pd.DataFrame) -> pd.DataFrame:
    c, h, l = d.close, d.high, d.low
    top, low = h.rolling(N).max().shift(1), l.rolling(N).min().shift(1)
    vol = np.log(c).diff().rolling(250).std().shift(1)
    isbox = (top / low - 1) <= K * vol * np.sqrt(N)
    return pd.DataFrame({"c": c, "low": low, "isbox": isbox, "brk": (c < low) & isbox, "new_low": c < low})


def related_events(s: pd.DataFrame, need_box: bool = True) -> tuple[pd.DatetimeIndex, pd.Series]:
    """相关指数"刚跌破"的事件日（去重）和"正常"状态（逐日布尔）。"""
    b = s["brk"] if need_box else s["new_low"]
    ref = s["low"].where(b).ffill(limit=4)
    broken = b.astype(int).rolling(5, min_periods=1).max().astype(bool) & (s["c"] < ref)
    ev, last = [], -10**9
    for k in np.flatnonzero(broken.to_numpy()):
        if k - last > GAP and s.index[k] >= pd.Timestamp(START):
            ev.append(s.index[k])
        last = k
    normal = (b.astype(int).rolling(GAP, min_periods=1).max() == 0) & (s["c"] >= s["low"])
    return pd.DatetimeIndex(ev), normal


def outcomes(d: pd.DataFrame, s: pd.DataFrame, rel_close: pd.Series) -> pd.DataFrame:
    c, l = d.close, d.low
    rev = lambda x, n: x[::-1].rolling(n, min_periods=1).min()[::-1].shift(-1)  # noqa: E731  t+1..t+n 内的最小值
    rr60 = rel_close.reindex(c.index).shift(-60) / rel_close.reindex(c.index) - 1
    out = pd.DataFrame({
        "20天": c.shift(-20) / c - 1, "60天": c.shift(-60) / c - 1,
        "20天内跌破": (rev(c, 20) < s["low"]).astype(float), "60天内跌破": (rev(c, 60) < s["low"]).astype(float),
        "60天最大回撤": rev(l, 60) / c - 1,
    })
    out["相对60天"] = out["60天"] - rr60
    out.loc[c.index[-60:], ["60天", "60天内跌破", "60天最大回撤", "相对60天"]] = np.nan
    out.loc[c.index[-20:], ["20天", "20天内跌破"]] = np.nan
    inside = s["isbox"] & (s["c"] >= s["low"]) & (s["brk"].astype(int).rolling(GAP, min_periods=1).max() == 0)
    also_broken = s["brk"].astype(int).rolling(5, min_periods=1).max().astype(bool) | s["new_low"]
    return out.assign(inside=inside, also_broken=also_broken)


def compare(pairs: list[tuple[str, str]], related: str, need_box=True) -> tuple[dict, int]:
    """pairs: [(目标名, 目标代码)]，related: 相关指数代码。"""
    R = load(related)
    rs = state(R)
    ev, normal = related_events(rs, need_box)
    A, B, C = [], [], []
    for _, code in pairs:
        d = load(code)
        if d is None:
            continue
        d = d[d.index.isin(R.index)]
        o = outcomes(d, state(d), R.close)
        o = o[o.index >= START]
        e = o.reindex(ev).dropna(subset=["inside"])
        A.append(e[e["inside"].astype(bool)])
        C.append(e[e["also_broken"].astype(bool)])
        B.append(o[o["inside"] & normal.reindex(o.index).fillna(False)])
    return {"相关指数刚跌破，目标还在箱体内": pd.concat(A), "相关指数正常，目标在箱体内（基准）": pd.concat(B),
            "相关指数刚跌破，目标也已跌破": pd.concat(C)}, len(ev)


def table(groups: dict) -> pd.DataFrame:
    rows = {}
    for k, x in groups.items():
        up = lambda s: (s.dropna() > 0).mean()  # noqa: E731
        rows[k] = {"样本": len(x), "20天上涨比例": up(x["20天"]), "20天中位": x["20天"].median(),
                   "60天上涨比例": up(x["60天"]), "60天中位": x["60天"].median(),
                   "20天内跟着跌破": x["20天内跌破"].dropna().astype(float).mean(),
                   "60天内跟着跌破": x["60天内跌破"].dropna().astype(float).mean(),
                   "60天最大回撤中位": x["60天最大回撤"].median(), "相对相关指数60天": x["相对60天"].median()}
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    C = cfg()
    lv2 = [(f[0], "sh" + f[1], f[5][:3]) for f in C if len(f[5]) == 5]
    lv1 = {f[5]: "sh" + f[1] for f in C if len(f[5]) == 3}
    inds = [(n, c) for n, c, _ in lv2]

    def show(title, t, n_ev):
        print(f"\n== {title}（相关指数跌破事件 {n_ev} 次）==")
        print(t.round(3).to_string())

    for need_box, tag in ((True, "跌破整理箱体"), (False, "创60日新低")):
        print(f"\n################ 相关指数：{tag}")
        for name, code in (("沪深300", "sh000300"), ("中证1000", "sh000852"), ("上证指数", "sh000001")):
            g, n = compare(inds, code, need_box)
            show(f"面={name}，点=127 个二级行业", table(g), n)
        # 线：每个二级行业对应自己的一级行业，样本合并后统计
        merged, n_all = {}, 0
        for tag1, pcode in lv1.items():
            kids = [(n, c) for n, c, p in lv2 if p == tag1]
            if kids:
                g, n = compare(kids, pcode, need_box)
                n_all += n
                for k, v in g.items():
                    merged.setdefault(k, []).append(v)
        show("线=所属一级行业，点=二级行业", table({k: pd.concat(v) for k, v in merged.items()}), n_all)
        semi = [("半导体材料", "sh881320"), ("半导体设备", "sh881321"), ("集成电路设计", "sh881322"), ("功率半导体", "sh881323"),
                ("半导体制造", "sh881324"), ("半导体封测", "sh881325")]
        for name, code in (("科创50", "sh000688"), ("创业板指", "sz399006"), ("半导体(二级)", "sh881319"), ("上证指数", "sh000001")):
            g, n = compare(semi, code, need_box)
            show(f"相关={name}，点=半导体 6 个三级行业", table(g), n)
