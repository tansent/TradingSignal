"""回测三种用顶底信号做指数的仓位方法：对称买卖、买重卖轻、按信号级别分档。

规则（三种方法共用）：
- 仓位 p 在 0–1 之间，信号当天按收盘价调整（顶部信号收盘才确定，等于尾盘操作），次日起按新仓位计收益。
- 买入信号：△B / ★B；卖出信号：★S / ▽S（各指数用自己的信号，与通达信 DINGDI 相同）。
- 同一轮信号：一次动作之后 5 个交易日内的同向信号算同一轮，不重复加减；
  分档法里同一轮内出现更高级别的信号，补足到更高级别的仓位。
- 空仓资金按年化 2% 计息（货币基金/逆回购），每次买卖按成交金额收 0.05% 成本。
- 起点空仓（买入持有除外）。指数是价格指数，不含分红。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dingdi  # noqa: E402
from monthly_bbi import asof_bbi, month_end_bbi  # noqa: E402
from tdx_reader import read_day  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CASH_YIELD = 0.02
COST = 0.0005
EPISODE = 5


def load_market() -> pd.DataFrame:
    """全市场条件：中线温度、宽基 ETF 净申购（亿元）。"""
    t = pd.read_parquet(ROOT / "data" / "temperature_v1.parquet")
    sys.path.insert(0, str(ROOT / "daily"))
    from update import etf_net_creation
    e = etf_net_creation()
    return pd.DataFrame({"mid": t["中线温度"]}).join(e[["net_yi"]], how="left")


def signals(code: str, mkt: pd.DataFrame, etf_yi: float) -> pd.DataFrame:
    df = read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day")
    r = dingdi.compute(df)
    s = pd.DataFrame({"close": df.close, "b1": r["底部一级"], "b2": r["底部二级"],
                      "s": r["顶部"] | r["顶部一级"]})
    s = s.join(mkt, how="left")
    s["cold"] = s["mid"] <= 10
    # 国家队：近 3 个交易日内有一天宽基 ETF 净申购 ≥ etf_yi 亿元（份额数据次日公布，所以只认已公布的）
    s["etf"] = (s["net_yi"].shift(1) >= etf_yi).rolling(3, min_periods=1).max().astype(bool)
    s["bbi"] = asof_bbi(df.close)
    me = month_end_bbi(df.close)
    brk = (me.close < me.bbi) & (me.close.shift() >= me.bbi.shift())
    s["bbi_month"] = False
    for m in me.index[brk]:
        days = s.index[s.index.to_period("M") == m.to_period("M")]
        if len(days):
            s.loc[days[-1], "bbi_month"] = True           # 月末最后一个交易日收盘确认
    below = s.close < s.bbi
    s["bbi_daily"] = below & ~below.shift(1, fill_value=False)
    s["bbi_two"] = r["破月BBI"]                              # 收盘连续 2 天在月线 BBI 下方（与通达信 DINGDI 一致）
    return s


def run(s: pd.DataFrame, buy_size, sell_size: float, exit_rule: str | None = None) -> pd.DataFrame:
    """buy_size(row) -> 这一天信号对应的本轮目标加仓量；exit_rule: None / 'bbi_month' / 'bbi_daily'。"""
    c = s.close.to_numpy()
    n = len(s)
    p = 0.0
    pos = np.zeros(n)
    trades = 0
    buy_ep_end, buy_ep_added, sell_ep_end = -1, 0.0, -1
    turnover = np.zeros(n)
    rows = s.to_dict("records")
    for i in range(n):
        r = rows[i]
        new_p = p
        want = buy_size(r) if (r["b1"] or r["b2"]) else 0.0
        if want > 0:
            if i > buy_ep_end:                              # 新一轮
                buy_ep_end, buy_ep_added = i + EPISODE, 0.0
            add = max(0.0, want - buy_ep_added)
            if add > 0 and new_p < 1:
                add = min(add, 1 - new_p)
                new_p += add
                buy_ep_added += add
                trades += 1
        if r["s"] and i > sell_ep_end and new_p > 0:
            sell_ep_end = i + EPISODE
            new_p = max(0.0, new_p - sell_size)
            trades += 1
        if exit_rule and r[exit_rule] and new_p > 0:
            new_p = 0.0
            trades += 1
        turnover[i] = abs(new_p - p)
        p = new_p
        pos[i] = p
    ret = np.r_[0, c[1:] / c[:-1] - 1]
    held = np.r_[0, pos[:-1]]
    daily = held * ret + (1 - held) * CASH_YIELD / 252 - turnover * COST
    return pd.DataFrame({"pos": pos, "equity": np.cumprod(1 + daily)}, index=s.index).assign(trades=trades)


def stats(eq: pd.Series, pos: pd.Series, trades: int) -> dict:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return {"年化收益": eq.iloc[-1] ** (1 / yrs) - 1, "最大回撤": (eq / eq.cummax() - 1).min(),
            "平均仓位": pos.mean(), "买卖次数": trades, "期末倍数": eq.iloc[-1]}


def strategies(etf_ok: bool):
    tier = lambda r: (0.4 if r["b2"] else 0.2) + (0.2 if r["cold"] else 0) + (0.2 if (etf_ok and r["etf"]) else 0)  # noqa: E731
    return {
        "A 对称（买1/3、卖1/3）": (lambda r: 1 / 3, 1 / 3, None),
        "B 买重卖轻（买1/2、卖1/5）": (lambda r: 0.5, 0.2, None),
        "C0 分档（不设离场）": (tier, 0.2, None),
        "C 分档 + 破月BBI离场": (tier, 0.2, "bbi_two"),
    }


def main(etf_yi: float = 200.0):
    mkt = load_market()
    etf_ok = mkt["net_yi"].first_valid_index() is not None
    out = []
    for name, code, start in (("沪深300", "sh000300", "2011-08-01"), ("创业板指", "sz399006", "2011-08-01"),
                              ("科创50", "sh000688", "2021-01-04")):
        s = signals(code, mkt, etf_yi)
        s = s[s.index >= start]
        bh = s.close / s.close.iloc[0]
        out.append({"指数": name, "方法": "买入持有", **stats(bh, pd.Series(1.0, index=s.index), 1)})
        for k, (bs, ss, ex) in strategies(etf_ok).items():
            r = run(s, bs, ss, ex)
            out.append({"指数": name, "方法": k, **stats(r["equity"], r["pos"], int(r.trades.iloc[0]))})
    df = pd.DataFrame(out)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    df = main()
    for k in ("年化收益", "最大回撤", "平均仓位"):
        df[k] = df[k].map(lambda v: f"{v:+.1%}" if k != "平均仓位" else f"{v:.0%}")
    df["期末倍数"] = df["期末倍数"].round(2)
    print(df.to_string(index=False))
