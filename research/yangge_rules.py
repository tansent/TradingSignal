"""检验从公众号"杨哥"文章里提炼的两条大盘规则（2011-08 起，8 个指数）。

1. 年线（250 日均线）：
   a. 底部信号（△B/★B）出现时，指数在年线上方还是下方，之后表现差多少；
      下方再分"刚跌破年线（≤60 个交易日）"和"已在年线下方超过 60 天（他说的'进入熊市几个月'）"。
   b. 当作离场规则：跌破年线就空仓，对比我们的破月BBI离场（都在 strategy_backtest 的"C 分档"框架里，买入都靠底部信号）；
      以及纯趋势持有：年线上方持有、下方空仓，对比月线 BBI 上方持有。
   c. 当作买入过滤：年线下方不接底部信号。
2. 放量确认：底部信号出现后不当天买，等到"上涨且成交额比前一天放大 ≥20%"的那天再买（20 个交易日内没等到就放弃），
   对比信号当天买入。卖出日相同（信号日后第 60 / 120 个交易日），比较哪种买法赚得多、回撤小。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dingdi  # noqa: E402
import strategy_backtest as sb  # noqa: E402
from monthly_bbi import asof_bbi  # noqa: E402
from tdx_reader import read_day  # noqa: E402

IDX = [("上证50", "sh000016"), ("沪深300", "sh000300"), ("中证500", "sh000905"), ("中证1000", "sh000852"),
       ("上证指数", "sh000001"), ("深证成指", "sz399001"), ("创业板指", "sz399006"), ("科创50", "sh000688")]
START = "2011-08-01"
GAP = 20


def load(code):
    return read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day")


def up(x):
    x = pd.Series(x).dropna()
    return (x > 0).mean() if len(x) else np.nan


# ---------- 1a. 底部信号 × 年线 ----------
def bottom_events() -> pd.DataFrame:
    rows = []
    for name, code in IDX:
        d = load(code)
        r = dingdi.compute(d)
        c, lo = d.close, d.low
        ma = c.rolling(250).mean()
        below = c < ma
        # 连续在年线下方的天数
        run = below.groupby((~below).cumsum()).cumsum()
        sig = (r["底部一级"] | r["底部二级"]).to_numpy()
        star = r["底部二级"].to_numpy()
        amt = d.amount
        last = -10**9
        for k in np.flatnonzero(sig):
            first, last = k - last > GAP, k
            if not first or c.index[k] < pd.Timestamp(START) or np.isnan(ma.iloc[k]) or k + 1 >= len(c):
                continue
            f = lambda n: c.iloc[k + n] / c.iloc[k] - 1 if k + n < len(c) else np.nan  # noqa: E731
            pos = "年线上方" if not below.iloc[k] else ("跌破年线≤60天" if run.iloc[k] <= 60 else "年线下方>60天")
            # 放量确认：k 之后 20 天内第一个"上涨且成交额 ≥ 前一天 ×1.2"的交易日
            conf = None
            for j in range(k + 1, min(k + 21, len(c))):
                if c.iloc[j] > c.iloc[j - 1] and amt.iloc[j] >= amt.iloc[j - 1] * 1.2:
                    conf = j
                    break
            row = {"指数": name, "日期": c.index[k], "★B": bool(star[k:k + GAP].any()), "年线": pos,
                   "离年线": c.iloc[k] / ma.iloc[k] - 1,
                   "60天": f(60), "120天": f(120), "250天": f(250),
                   "60天内再跌": lo.iloc[k + 1:k + 61].min() / c.iloc[k] - 1,
                   "120天后站上年线": float(c.iloc[k + 120] > ma.iloc[k + 120]) if k + 120 < len(c) else np.nan,
                   "确认": conf is not None}
            if conf is not None:
                row["等待天数"] = conf - k
                row["确认价/信号价"] = c.iloc[conf] / c.iloc[k] - 1
                for n in (60, 120):
                    row[f"确认买到{n}天"] = c.iloc[k + n] / c.iloc[conf] - 1 if k + n < len(c) else np.nan
                row["确认后60天内再跌"] = lo.iloc[conf + 1:conf + 61].min() / c.iloc[conf] - 1
            rows.append(row)
    return pd.DataFrame(rows)


def table_ma(E: pd.DataFrame) -> pd.DataFrame:
    g = E.groupby("年线")
    return pd.DataFrame({
        "次数": g.size(), "其中有★B": g["★B"].sum(),
        "60天上涨比例": g["60天"].apply(up), "60天中位": g["60天"].median(),
        "120天上涨比例": g["120天"].apply(up), "120天中位": g["120天"].median(),
        "250天上涨比例": g["250天"].apply(up), "250天中位": g["250天"].median(),
        "60天内再跌": g["60天内再跌"].median(), "120天后站上年线": g["120天后站上年线"].mean(),
    }).reindex(["年线上方", "跌破年线≤60天", "年线下方>60天"])


def table_confirm(E: pd.DataFrame, by=None) -> pd.DataFrame:
    def one(x):
        c = x[x["确认"]]
        return pd.Series({
            "次数": len(x), "20天内等到放量": x["确认"].mean(), "平均等待天数": c["等待天数"].mean(),
            "确认价比信号价": c["确认价/信号价"].median(),
            "信号日买-60天": x["60天"].median(), "确认买-同一卖点60天": c["确认买到60天"].median(),
            "信号日买-120天": x["120天"].median(), "确认买-同一卖点120天": c["确认买到120天"].median(),
            "信号日买-60天内再跌": x["60天内再跌"].median(), "确认买-60天内再跌": c["确认后60天内再跌"].median(),
            "没等到的那些-120天": x.loc[~x["确认"], "120天"].median(),
        })
    if by is None:
        return one(E).to_frame("全部").T
    return E.groupby(by).apply(one)


# ---------- 1b/1c. 策略：年线离场、年线过滤、纯趋势 ----------
def strategy_compare() -> pd.DataFrame:
    mkt = sb.load_market()
    etf_ok = mkt["net_yi"].first_valid_index() is not None
    tier = lambda r: (0.4 if r["b2"] else 0.2) + (0.2 if r["cold"] else 0) + (0.2 if (etf_ok and r["etf"]) else 0)  # noqa: E731
    out = []
    for name, code, start in (("沪深300", "sh000300", "2011-08-01"), ("上证指数", "sh000001", "2011-08-01"),
                              ("创业板指", "sz399006", "2011-08-01"), ("科创50", "sh000688", "2021-01-04")):
        d = load(code)
        s = sb.signals_df(d, mkt, 200.0)
        ma = d.close.rolling(250).mean()
        below = d.close < ma
        s["ma_break"] = below & ~below.shift(1, fill_value=False)                       # 收盘跌破年线当天
        s["ma_two"] = below & below.shift(1, fill_value=False) & ~below.shift(2, fill_value=False)  # 连续 2 天在年线下方
        s["below_ma"] = below
        s = s[s.index >= start]
        bh = s.close / s.close.iloc[0]
        out.append({"指数": name, "方法": "买入持有", **sb.stats(bh, pd.Series(1.0, index=s.index), 1)})
        for k, bs, ex in (("C 分档 + 破月BBI离场（现行）", tier, "bbi_two"),
                          ("C 分档 + 跌破年线离场", tier, "ma_break"),
                          ("C 分档 + 连续2天跌破年线离场", tier, "ma_two"),
                          ("C 分档 + 破月BBI离场 + 年线下方不接B", lambda r: 0.0 if r["below_ma"] else tier(r), "bbi_two"),
                          ("C 分档 + 破月BBI离场 + 年线下方只接★★B级",
                           lambda r: (tier(r) if (r["cold"] or (etf_ok and r["etf"])) else 0.0) if r["below_ma"] else tier(r), "bbi_two")):
            res = sb.run(s, bs, 0.2, ex)
            out.append({"指数": name, "方法": k, **sb.stats(res["equity"], res["pos"], int(res.trades.iloc[0]))})
        # 纯趋势：上方持有、下方空仓（次日生效）
        for lab, line in (("纯趋势：年线上方持有", ma.reindex(s.index)), ("纯趋势：月线BBI上方持有", asof_bbi(d.close).reindex(s.index))):
            hold = (s.close > line).astype(float).fillna(0)
            ret = s.close.pct_change().fillna(0)
            held = hold.shift(1).fillna(0)
            turn = hold.diff().abs().fillna(0)
            eq = (1 + held * ret + (1 - held) * sb.CASH_YIELD / 252 - turn * sb.COST).cumprod()
            out.append({"指数": name, "方法": lab, **sb.stats(eq, hold, int(turn.sum()))})
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    E = bottom_events()
    print("== 1a. 底部信号（△B/★B，同一指数 20 天内算一次）按出现时与年线的关系 ==")
    print(table_ma(E).round(3).to_string())
    print("\n== 2. 放量确认再买 vs 信号当天买（卖出日相同）==")
    print(table_confirm(E).round(3).to_string())
    print(table_confirm(E, "年线").round(3).to_string())
    print("\n== 1b/1c. 策略对比 ==")
    df = strategy_compare()
    for k in ("年化收益", "最大回撤", "平均仓位"):
        df[k] = df[k].map(lambda v: f"{v:+.1%}" if k != "平均仓位" else f"{v:.0%}")
    df["期末倍数"] = df["期末倍数"].round(2)
    print(df.to_string(index=False))
