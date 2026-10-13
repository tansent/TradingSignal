"""纳指ETF(513100)、标普ETF(513500) 按美股信号操作，和红利低波比较。

- ETF 价格取通达信日线（人民币，含 QDII 溢价的影响），单日涨跌超过 30% 视为份额拆分。
- 美股信号（research/us_market.py）在美国收盘后才知道，所以在下一个 A 股交易日收盘成交。
  纳指ETF 用纳斯达克综指的信号（纳指100 的开高低收只从 2014 年开始），标普ETF 用标普500 的信号。
- 红利低波用全收益指数 H20269 再减去 0.5%/年近似 ETF 费率，信号用价格指数 H30269（与 sector_backtest 一致）。
- 空仓资金年化 2%，每次买卖成本 0.05%。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import sector_backtest as sbt  # noqa: E402
import strategy_backtest as sb  # noqa: E402
import us_market as um  # noqa: E402
from tdx_reader import read_day  # noqa: E402

ETFS = {"纳指ETF": ("sh513100", "纳指"), "标普ETF": ("sh513500", "标普500")}


def etf_price(code: str) -> pd.Series:
    c = read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day").close
    r = c.pct_change()
    r[r.abs() > 0.3] = 0.0
    return (1 + r.fillna(0)).cumprod()


def to_a_days(flag: pd.Series, a_days: pd.DatetimeIndex) -> pd.Series:
    """美国日期 t 的信号 → t 之后的第一个 A 股交易日。"""
    out = pd.Series(False, index=a_days)
    for t in flag.index[flag.fillna(False).astype(bool)]:
        k = a_days.searchsorted(t, side="right")
        if k < len(a_days):
            out.iloc[k] = True
    return out


def etf_frame(name: str) -> pd.DataFrame:
    code, us = ETFS[name]
    px = etf_price(code)
    s = um.signals(us)
    a = px.index
    f = pd.DataFrame({"close": px, "b1": to_a_days(s["△B"], a), "b2": to_a_days(s["★B"], a),
                      "s": to_a_days(s["★S"] | s["▽S"], a), "bbi_two": to_a_days(s["破月BBI"], a)})
    f["cold"] = False
    f["etf"] = False
    return f


def dlv_frame() -> pd.DataFrame:
    import os
    os.environ["NO_PROXY"] = "*"
    d = sbt.daily("csi:H30269")
    s = sb.signals_df(d, sb.load_market(), 200.0)
    tr = sbt.total_return("csi:H30269", "H20269", None).reindex(s.index)
    tr = tr * np.exp(-0.005 * np.arange(len(tr)) / 252)       # 近似扣除 ETF 费率
    s["close"] = tr
    return s.dropna(subset=["close"])


def run_core(s: pd.DataFrame, core: float, step: float) -> tuple[pd.Series, pd.Series]:
    """底仓 core，底部信号每轮加 step（最多 100%），顶部信号每轮减 step（最低回到 core）。"""
    p, pos, turn = core, [], []
    bend = send = -1
    for i, r in enumerate(s.to_dict("records")):
        q = p
        if (r["b1"] or r["b2"]) and i > bend:
            bend, q = i + sb.EPISODE, min(1.0, q + step)
        if r["s"] and i > send:
            send, q = i + sb.EPISODE, max(core, q - step)
        turn.append(abs(q - p))
        p = q
        pos.append(p)
    c = s.close.to_numpy()
    ret = np.r_[0, c[1:] / c[:-1] - 1]
    held = np.r_[core, pos[:-1]]
    eq = np.cumprod(1 + held * ret + (1 - held) * sb.CASH_YIELD / 252 - np.array(turn) * sb.COST)
    return pd.Series(eq, index=s.index), pd.Series(pos, index=s.index)


def methods(s: pd.DataFrame) -> dict:
    third = lambda r: 1 / 3  # noqa: E731
    out = {"买入持有": (s.close / s.close.iloc[0], pd.Series(1.0, index=s.index))}
    r = sb.run(s, third, 1 / 3, None)
    out["A 对称（买卖各 1/3）"] = (r["equity"], r["pos"])
    r = sb.run(s, third, 0.0, None)
    out["只在底部信号加仓、不卖"] = (r["equity"], r["pos"])
    out["底仓 60% + 信号加减 20%"] = run_core(s, 0.6, 0.2)
    return out


def stats_table(frames: dict, start: str, end: str = "2099-12-31") -> pd.DataFrame:
    rows = []
    for name, s in frames.items():
        s = s[(s.index >= start) & (s.index <= end)]
        for k, (eq, pos) in methods(s).items():
            rows.append({"品种": name, "方法": k, **sb.stats(eq / eq.iloc[0], pos, 0)})
    return pd.DataFrame(rows)


def combo(frames: dict, w: dict, start: str) -> dict:
    """几只品种买入持有、每年再平衡的组合。"""
    px = pd.DataFrame({k: frames[k].close for k in w}).dropna()
    px = px[px.index >= start]
    r = px.pct_change().fillna(0)
    v = np.array([w[k] for k in px.columns], float)
    eq = [1.0]
    for i in range(1, len(px)):
        v = v * (1 + r.iloc[i].to_numpy())
        if px.index[i].year != px.index[i - 1].year:
            v = v.sum() * np.array([w[k] for k in px.columns])
        eq.append(v.sum())
    return sb.stats(pd.Series(eq, index=px.index), pd.Series(1.0, index=px.index), 0)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    frames = {k: etf_frame(k) for k in ETFS}
    frames["红利低波"] = dlv_frame()
    for start in ("2015-01-05", "2019-01-02"):
        T = stats_table(frames, start)
        print(f"\n===== {start[:4]} 年起（年化 / 最大回撤 / 平均仓位）=====")
        for name in frames:
            print(f"-- {name}")
            for _, r in T[T["品种"] == name].iterrows():
                print(f"   {r['方法']:<20} {r['年化收益']:+.1%} / {r['最大回撤']:.0%} / {r['平均仓位']:.0%}")
        for lab, w in (("红利低波 50% + 纳指ETF 50%", {"红利低波": .5, "纳指ETF": .5}),
                       ("红利低波 50% + 标普ETF 50%", {"红利低波": .5, "标普ETF": .5}),
                       ("红利低波 1/3 + 纳指 1/3 + 标普 1/3", {"红利低波": 1 / 3, "纳指ETF": 1 / 3, "标普ETF": 1 / 3})):
            st = combo(frames, w, start)
            print(f"   组合 {lab:<28} {st['年化收益']:+.1%} / {st['最大回撤']:.0%}")
    r = pd.DataFrame({k: v.close for k, v in frames.items()}).dropna().pct_change().dropna()
    print("\n日收益相关系数（2015 起）：")
    print(r[r.index >= "2015-01-05"].corr().round(2).to_string())
    y = pd.DataFrame({k: v.close for k, v in frames.items()}).dropna()
    y = y[y.index >= "2015-01-01"].resample("YE").last().pct_change().dropna() * 100
    y.index = y.index.year
    print("\n各年涨跌（%）："); print(y.round(0).T.to_string())
