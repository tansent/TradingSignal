"""牛熊线选哪条，以及一位趋势博主的"5/12/72 日线"买法（2026-10-19）。

一、牛熊线：8 个宽基，2011-08 起。同一套规则：收盘连续 2 天在线上方就满仓，连续 2 天在下方就空仓（空仓年化 2%，
    每次换仓 0.05% 成本）。比较月线 BBI、250 日线（年线）、120 日线（半年线）、72 日线。
二、博主买法：牛市（收盘在 72 日线上方）里，K 线回踩 5 日线或 12 日线不破（最低价碰到或跌破均线、收盘仍在均线上方），
    最好缩量（成交量低于 5 日均量）、K 线短小（振幅低于 20 日平均振幅的 70%）。
    127 个行业指数 + 8 个宽基，2013 年起，看之后 5/10/20 天的涨跌；再模拟"收盘跌破 12 日线就卖（最长 60 天）"的每笔收益。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dingdi  # noqa: E402
from point_line_surface import cfg  # noqa: E402
from tdx_reader import read_day  # noqa: E402

V = "D:/TDXapp/vipdoc"
BROAD = [("上证50", "sh000016"), ("沪深300", "sh000300"), ("中证500", "sh000905"), ("中证1000", "sh000852"),
         ("上证指数", "sh000001"), ("深证成指", "sz399001"), ("创业板指", "sz399006"), ("科创50", "sh000688")]


def switch(c: pd.Series, line: pd.Series) -> pd.Series:
    above = (c > line).to_numpy()
    below = (c < line).to_numpy()
    pos, p = np.zeros(len(c)), 0.0
    for i in range(1, len(c)):
        if above[i] and above[i - 1]:
            p = 1.0
        elif below[i] and below[i - 1]:
            p = 0.0
        pos[i] = p
    return pd.Series(pos, index=c.index)


def perf(c: pd.Series, pos: pd.Series) -> dict:
    ret = c.pct_change().fillna(0).to_numpy()
    held = np.r_[0, pos.to_numpy()[:-1]]
    turn = np.abs(np.diff(np.r_[0, pos.to_numpy()]))
    eq = np.cumprod(1 + held * ret + (1 - held) * 0.02 / 252 - turn * 0.0005)
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    e = pd.Series(eq, index=c.index)
    return {"年化": e.iloc[-1] ** (1 / yrs) - 1, "最大回撤": (e / e.cummax() - 1).min(), "持仓时间": pos.mean(), "换仓次数/年": turn.sum() / yrs}


def part1():
    rows = []
    for name, code in BROAD:
        d = read_day(f"{V}/{code[:2]}/lday/{code}.day")
        c = d.close
        lines = {"月线BBI": dingdi.compute(d)["月线BBI"], "250日线": c.rolling(250).mean(), "120日线": c.rolling(120).mean(), "72日线": c.rolling(72).mean()}
        for lo, hi, per in (("2011-08-01", "2026-12-31", "全期"), ("2011-08-01", "2018-12-31", "2011-2018"), ("2019-01-01", "2026-12-31", "2019-2026")):
            m = (c.index >= lo) & (c.index <= hi)
            if m.sum() < 250:
                continue
            rows.append({"区间": per, "方法": "买入持有", **perf(c[m], pd.Series(1.0, index=c[m].index))})
            for k, ln in lines.items():
                rows.append({"区间": per, "方法": k, **perf(c[m], switch(c, ln)[m])})
    R = pd.DataFrame(rows)
    out = R.groupby(["区间", "方法"], sort=False).mean(numeric_only=True)
    print("== 一、牛熊线（8 个宽基平均）==")
    print(out.apply(lambda col: col.map(lambda v: f"{v:.1f}" if col.name == "换仓次数/年" else f"{v:+.1%}" if col.name != "持仓时间" else f"{v:.0%}")).to_string())


def part2():
    universe = [(f[0], "sh" + f[1]) for f in cfg() if len(f[5]) == 5] + BROAD
    rows = []
    for name, code in universe:
        try:
            d = read_day(f"{V}/{code[:2]}/lday/{code}.day")
        except Exception:  # noqa: BLE001
            continue
        if len(d) < 400:
            continue
        c, h, lo, vol = d.close, d.high, d.low, d.volume
        ma5, ma12, ma72 = c.rolling(5).mean(), c.rolling(12).mean(), c.rolling(72).mean()
        rng = (h - lo) / c.shift()
        bull = c > ma72
        pb5 = bull & (lo <= ma5) & (c >= ma5)
        pb12 = bull & (lo <= ma12) & (c >= ma12)
        quiet = (vol < vol.rolling(5).mean()) & (rng <= rng.rolling(20).mean() * 0.7)
        C, M12 = c.to_numpy(), ma12.to_numpy()
        idx = np.flatnonzero((d.index >= pd.Timestamp("2013-01-01")).astype(bool))
        flags = {"牛市里所有交易日（基准）": bull, "回踩 5 日线不破": pb5, "回踩 12 日线不破": pb12,
                 "回踩 5 日线不破 + 缩量小K线": pb5 & quiet, "回踩 12 日线不破 + 缩量小K线": pb12 & quiet}
        for k, f in flags.items():
            fv = f.to_numpy()
            for i in idx[fv[idx]]:
                if i + 60 >= len(C):
                    continue
                out = np.flatnonzero(C[i + 1:i + 61] < M12[i + 1:i + 61])
                j = i + 1 + out[0] if len(out) else i + 60
                rows.append({"组": k, "f5": C[i + 5] / C[i] - 1, "f10": C[i + 10] / C[i] - 1, "f20": C[i + 20] / C[i] - 1,
                             "trade": C[j] / C[i] - 1, "days": j - i})
    R = pd.DataFrame(rows)
    print("\n== 二、回踩买法（127 个行业 + 8 个宽基，2013 年起）==")
    for k, g in R.groupby("组", sort=False):
        print(f"  {k:22s} {len(g):7d} 次  " + "  ".join(f"{h}天涨 {(g[f'f{h}'] > 0).mean() * 100:3.0f}% 中位 {g[f'f{h}'].median() * 100:+.1f}%" for h in (5, 10, 20))
              + f"  |  跌破12日线就卖：赚钱 {(g.trade > 0).mean() * 100:3.0f}% 平均 {g.trade.mean() * 100:+.2f}% 平均持有 {g.days.mean():.0f} 天")


if __name__ == "__main__":
    part1()
    part2()
