"""把信号框架用在红利、银行、煤炭上（日线 C、周线全进全出、周线定区域+日线分批等），和买入持有比较。

信号用通达信里的价格指数计算（上证红利 000015、中证银行 399986、中证煤炭 399998）；
收益用全收益：上证红利用中证官方全收益指数 H00015；银行、煤炭没有公开全收益指数，按价格涨跌加估计股息率
（银行年化 4.5%、煤炭 5%）合成。买入持有拿全部股息，信号方法只在持仓期间拿。
这些行业指数在通达信里没有涨跌家数，所以 ★B / ★★B 不会出现。
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import strategy_backtest as sb  # noqa: E402
import zone_backtest as zb  # noqa: E402
from tdx_reader import read_day  # noqa: E402
from weekly_zone import zones  # noqa: E402

os.environ["NO_PROXY"] = "*"
SECTORS = [("上证红利", "sh000015", "H00015", None), ("中证银行", "sz399986", None, 0.045), ("中证煤炭", "sz399998", None, 0.05),
           ("红利低波", "csi:H30269", "H20269", None)]      # 红利低波不在通达信里，日线取自中证指数官网


def csi_daily(symbol: str) -> pd.DataFrame:
    import akshare as ak
    t = ak.stock_zh_index_hist_csindex(symbol=symbol, start_date="20100101", end_date="20991231")
    t = t.assign(date=pd.to_datetime(t["日期"])).set_index("date").sort_index()
    return pd.DataFrame({"open": t["开盘"], "high": t["最高"], "low": t["最低"], "close": t["收盘"],
                         "volume": t["成交量"], "amount": t["成交金额"]}).astype(float)


def daily(code: str) -> pd.DataFrame:
    return csi_daily(code[4:]) if code.startswith("csi:") else read_day(f"D:/TDXapp/vipdoc/{code[:2]}/lday/{code}.day")


def total_return(code: str, cs_code: str | None, yld: float | None) -> pd.Series:
    px = daily(code).close
    if cs_code:
        import akshare as ak
        t = ak.stock_zh_index_hist_csindex(symbol=cs_code, start_date="20100101", end_date="20991231")
        tr = t.assign(日期=pd.to_datetime(t["日期"])).set_index("日期")["收盘"].astype(float)
        return tr.reindex(px.index).ffill()
    r = px.pct_change().fillna(0) + yld / 252
    return (1 + r).cumprod() * px.iloc[0]


def frame(code: str, cs_code, yld, mkt) -> pd.DataFrame:
    d = daily(code)
    s = sb.signals_df(d, mkt, 200.0).join(zones(d, 26))
    s["px"] = s["close"]
    s["close"] = total_return(code, cs_code, yld).reindex(s.index)    # 收益按全收益算
    return s.dropna(subset=["close"])


def table(start: str, mkt) -> pd.DataFrame:
    out = []
    strat = sb.strategies(etf_ok=True)
    for name, code, cs, yld in SECTORS:
        s = frame(code, cs, yld, mkt)
        s = s[s.index >= start]
        res = {"买入持有": (s.close / s.close.iloc[0], pd.Series(1.0, index=s.index))}
        for k in ("A 对称（买1/3、卖1/3）", "B 买重卖轻（买1/2、卖1/5）", "C 分档 + 破月BBI离场"):
            bs, ss, ex = strat[k]
            r = sb.run(s, bs, ss, ex)
            res[k.split("（")[0].split(" + ")[0] if k.startswith("C") else k[:1] + " " + k[2:k.index("（")]] = (r["equity"], r["pos"])
        res["周线全进全出"] = zb.weekly_all_in_out(s)
        res["周线定区域+日线分批"] = zb.run_zone(s, False, False)
        for k, (eq, pos) in res.items():
            st = sb.stats(eq, pos, 0)
            out.append({"板块": name, "方法": k, **st})
    return pd.DataFrame(out)


def signal_quality(mkt) -> pd.DataFrame:
    """各类日线信号之后 60 天的表现（价格指数，不含股息），与所有交易日比较。"""
    rows = []
    for name, code, cs, yld in SECTORS:
        s = frame(code, cs, yld, mkt)
        s = s[s.index >= "2011-08-01"]
        px = s["px"]
        f60 = px.shift(-60) / px - 1
        for lab, m in (("△B 底部", s.b1), ("★S/▽S 顶部", s.s), ("破月BBI", s.bbi_two), ("所有交易日", pd.Series(True, index=s.index))):
            x = f60[m.astype(bool)].dropna()
            rows.append({"板块": name, "信号": lab, "次数": len(x), "60天后上涨比例": (x > 0).mean(), "60天平均": x.mean()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    mkt = sb.load_market()
    Q = signal_quality(mkt)
    print("== 日线信号之后 60 天（价格，2011-08 起）==")
    print(Q.pivot_table(index="信号", columns="板块", values=["次数", "60天后上涨比例", "60天平均"], sort=False).round(3).to_string())
    for start in ("2011-08-01", "2015-06-01", "2019-01-01"):
        T = table(start, mkt)
        print(f"\n===== 起点 {start}（年化收益 / 最大回撤 / 平均仓位）=====")
        for name in [n for n, *_ in SECTORS]:
            g = T[T["板块"] == name].set_index("方法")
            print(f"-- {name}")
            for k, r in g.iterrows():
                print(f"   {k:<18} {r['年化收益']:+.1%} / {r['最大回撤']:.0%} / {r['平均仓位']:.0%}")
