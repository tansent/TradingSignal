"""把 DINGDI 用在周线上：列出全部周线信号和之后的表现，并回测只按周线信号交易。

周线由日线合成（周五或该周最后一个交易日为一根）。涨跌家数取该周最后一个交易日的值
（通达信周线上 ADVANCE/DECLINE 的口径未核实，只影响 ★B）。
注意：公式里的窗口在周线上含义不同，HHV(H,250) 变成约 5 年高点，HHV(H,120) 约 2.3 年。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dingdi  # noqa: E402
from tdx_reader import read_day  # noqa: E402

IDX = [("上证指数", "sh000001"), ("深证成指", "sz399001"), ("创业板指", "sz399006"), ("沪深300", "sh000300"),
       ("上证50", "sh000016"), ("中证500", "sh000905"), ("中证1000", "sh000852"), ("科创50", "sh000688")]


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(df.index.to_period("W-FRI"))
    w = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last(),
                      "volume": g.volume.sum(), "amount": g.amount.sum()})
    if "adv" in df:
        w["adv"], w["dec"] = g.adv.last(), g.dec.last()
    w.index = g.apply(lambda x: x.index[-1])                 # 用该周最后一个交易日作日期
    return w


def events(name: str, code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day")
    w = weekly(d)
    r = dingdi.compute(w)
    c, h, low = w.close, w.high, w.low
    rows = []
    for col, lab in (("顶部", "★S"), ("顶部一级", "▽S"), ("底部二级", "★B"), ("底部一级", "△B"), ("破月BBI", "破")):
        for t in r.index[r[col]]:
            k = w.index.get_loc(t)
            row = {"指数": name, "日期": t.date(), "信号": lab, "收盘": c.iloc[k]}
            for n in (13, 26, 52):
                row[f"{n}周后"] = c.iloc[k + n] / c.iloc[k] - 1 if k + n < len(c) else np.nan
            row["26周内最高"] = h.iloc[k + 1:k + 27].max() / c.iloc[k] - 1 if k + 1 < len(c) else np.nan
            row["26周内最低"] = low.iloc[k + 1:k + 27].min() / c.iloc[k] - 1 if k + 1 < len(c) else np.nan
            rows.append(row)
    return pd.DataFrame(rows), w.join(r)


def run(w: pd.DataFrame, buy: float, sell: float, exit_bbi: bool, b2_extra: float = 0.0, cash=0.02, cost=0.0005):
    """周线回测，规则同日线 C 方法：同一轮（4 周内）不重复加仓；信号当周收盘成交。"""
    p, pos, turn = 0.0, [], []
    ep_end, ep_add, sell_end = -1, 0.0, -1
    for i, (_, r) in enumerate(w.iterrows()):
        q = p
        if r["底部一级"] or r["底部二级"]:
            want = buy + (b2_extra if r["底部二级"] else 0)
            if i > ep_end:
                ep_end, ep_add = i + 4, 0.0
            add = min(max(0.0, want - ep_add), 1 - q)
            q += add
            ep_add += add
        if (r["顶部"] or r["顶部一级"]) and i > sell_end and q > 0:
            sell_end = i + 4
            q = max(0.0, q - sell)
        if exit_bbi and r["破月BBI"]:
            q = 0.0
        turn.append(abs(q - p))
        p = q
        pos.append(p)
    c = w.close.to_numpy()
    ret = np.r_[0, c[1:] / c[:-1] - 1]
    held = np.r_[0, pos[:-1]]
    eq = np.cumprod(1 + held * ret + (1 - held) * cash / 52 - np.array(turn) * cost)
    eq = pd.Series(eq, index=w.index)
    yrs = (w.index[-1] - w.index[0]).days / 365.25
    return {"年化": eq.iloc[-1] ** (1 / yrs) - 1, "最大回撤": (eq / eq.cummax() - 1).min(), "平均仓位": float(np.mean(pos))}


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    allev, W = [], {}
    for name, code in IDX:
        e, w = events(name, code)
        allev.append(e)
        W[name] = w
    E = pd.concat(allev, ignore_index=True)
    E.to_csv(Path(__file__).resolve().parent.parent / "data" / "weekly_signal_events.csv", index=False, encoding="utf-8-sig")
    print("== 每类周线信号：次数和之后表现（8 个指数合计，各指数有数据以来）==")
    g = E.groupby("信号").agg(次数=("日期", "size"), 涨13周=("13周后", lambda x: (x.dropna() > 0).mean()), 平均13周=("13周后", "mean"),
                              涨52周=("52周后", lambda x: (x.dropna() > 0).mean()), 平均52周=("52周后", "mean"),
                              之后26周最高中位=("26周内最高", "median"), 之后26周最低中位=("26周内最低", "median"))
    print(g.round(3).to_string())
    for nm in ("创业板指", "上证指数"):
        s = E[E["指数"] == nm].sort_values("日期")
        print(f"\n== {nm}：全部周线顶部信号 ==")
        print(s[s["信号"].isin(["★S", "▽S"])][["日期", "信号", "收盘", "13周后", "52周后", "26周内最高", "26周内最低"]].round(3).to_string(index=False))
        b = s[s["信号"].isin(["★B", "△B"])]
        print(f"{nm}：周线底部信号 {len(b)} 次，按年：", b.groupby(pd.to_datetime(b["日期"]).dt.year).size().to_dict())
        print(f"{nm}：破月BBI 周线 {int((s['信号'] == '破').sum())} 次")
    print("\n== 创业板指：只按周线信号交易（2015-01 起，空仓资金 2%/年）==")
    w = W["创业板指"]
    w = w[w.index >= "2015-01-01"]
    bh = w.close / w.close.iloc[0]
    yrs = (w.index[-1] - w.index[0]).days / 365.25
    print("买入持有", f"年化 {bh.iloc[-1] ** (1 / yrs) - 1:+.1%}，最大回撤 {(bh / bh.cummax() - 1).min():.1%}")
    for lab, args in (("全进全出：底部信号全仓买、顶部信号全部卖", (1.0, 1.0, False)),
                      ("全进全出 + 破月BBI 清仓", (1.0, 1.0, True)),
                      ("分档：△B 买 20%、★B 买 40%、顶部卖 20%、破月BBI 清仓", (0.2, 0.2, True, 0.2)),
                      ("分档、不设 BBI 离场", (0.2, 0.2, False, 0.2))):
        st = run(w, *args)
        print(f"{lab}：年化 {st['年化']:+.1%}，最大回撤 {st['最大回撤']:.1%}，平均仓位 {st['平均仓位']:.0%}")
