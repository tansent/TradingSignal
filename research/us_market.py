"""美股顶底信号和情绪温度（对应 A 股的 DINGDI 和中线温度）。

数据：新浪美股指数日线（标普500 .INX、纳斯达克综指 .IXIC，2004 起，开高低收）；
FRED：VIX(VIXCLS)、纳指波动率(VXNCLS)、芝加哥联储金融条件指数(NFCI，周度)、美债 10 年期收益率(DGS10)。
- 情绪、★S/▽S/△B：与 A 股公式完全相同（research/dingdi.py）。
- ★B：A 股用"下跌家数 ≥92%"判断恐慌，美股改用当天对应的波动率指数 ≥30（标普用 VIX，纳指用 VXN）。
- 美股温度：VIX 高低、VIX 相对 50 日均值的飙升程度、离 200 日线距离、60 日涨跌、NFCI、10 年期收益率 60 日变化，
  各自取过去 5 年的分位（恐慌方向记为低分），再平均，0–100。
"""
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dingdi  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
US = ROOT / "data" / "us"
INDEX = {"标普500": (".INX", "VIXCLS"), "纳指": (".IXIC", "VXNCLS")}


def fred(sid: str, refresh: bool = False) -> pd.Series:
    f = US / f"{sid}.csv"
    if refresh or not f.exists():
        import requests
        err = None
        for proxy in (None, "http://127.0.0.1:7897"):        # 先直连，失败再走本机代理
            try:
                r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid}, timeout=60,
                                 proxies={"http": proxy, "https": proxy})
                if not r.text.startswith(("DATE", "observation_date")):
                    raise ValueError(r.text[:60])
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = ["date", sid]
                df[sid] = pd.to_numeric(df[sid], errors="coerce")
                US.mkdir(parents=True, exist_ok=True)
                df.dropna().to_csv(f, index=False)
                err = None
                break
            except Exception as e:  # noqa: BLE001
                err = e
        if err is not None:
            if not f.exists():
                raise err
            print(f"FRED {sid} 下载失败，沿用上次的数据：{repr(err)[:80]}")
    s = pd.read_csv(f, parse_dates=["date"]).set_index("date")[sid]
    return s


def sina_index(sym: str, refresh: bool = False) -> pd.DataFrame:
    f = US / f"{sym.strip('.')}.csv"
    if refresh or not f.exists():
        os.environ["NO_PROXY"] = "*"
        import akshare as ak
        US.mkdir(parents=True, exist_ok=True)
        try:
            ak.index_us_stock_sina(symbol=sym).to_csv(f, index=False)
        except Exception as e:  # noqa: BLE001
            if not f.exists():
                raise
            print(f"新浪 {sym} 下载失败，沿用上次的数据：{repr(e)[:80]}")
    d = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
    return d[["open", "high", "low", "close", "volume"]].astype(float)


def pct_rank(s: pd.Series, win: int = 1260) -> pd.Series:
    """过去 win 个交易日内的分位（0–1），只用过去数据。"""
    return s.rolling(win, min_periods=250).apply(lambda x: (x[:-1] < x[-1]).mean() + 0.5 * (x[:-1] == x[-1]).mean(), raw=True)


def temperature(px: pd.DataFrame, vol: pd.Series) -> pd.DataFrame:
    c = px.close
    vix = vol.reindex(c.index).ffill()
    nfci = fred("NFCI").reindex(c.index, method="ffill")
    y10 = fred("DGS10").reindex(c.index).ffill()
    parts = pd.DataFrame({
        "VIX": 1 - pct_rank(vix),
        "VIX飙升": 1 - pct_rank(vix / vix.rolling(50).mean()),
        "离200日线": pct_rank(c / c.rolling(200).mean() - 1),
        "60日涨跌": pct_rank(c / c.shift(60) - 1),
        "金融条件": 1 - pct_rank(nfci),
        "利率冲击": 1 - pct_rank(y10 - y10.shift(60)),
    })
    return parts.assign(温度=parts.mean(axis=1) * 100)


def signals(name: str) -> pd.DataFrame:
    sym, vsid = INDEX[name]
    px = sina_index(sym)
    vol = fred(vsid).reindex(px.index).ffill()
    r = dingdi.compute(px)                                  # 没有 adv/dec：底部二级恒为 False
    panic = vol >= 30
    out = pd.DataFrame({"close": px.close, "情绪": r["情绪"], "★S": r["顶部"], "▽S": r["顶部一级"],
                        "△B": r["底部一级"] & ~panic, "★B": r["底部一级"] & panic, "破月BBI": r["破月BBI"], "波动率": vol})
    return out.join(temperature(px, vol)[["温度"]])


def forward_table(name: str, start="2005-01-01") -> pd.DataFrame:
    s = signals(name)
    s = s[s.index >= start]
    c = s.close
    rows = []
    groups = {"△B": s["△B"], "★B（△B 且波动率≥30）": s["★B"], "★S/▽S": s["★S"] | s["▽S"], "破月BBI": s["破月BBI"],
              "温度 <10": s["温度"] < 10, "温度 10–30": s["温度"].between(10, 30), "温度 30–70": s["温度"].between(30, 70),
              "温度 70–90": s["温度"].between(70, 90), "温度 >90": s["温度"] > 90, "所有交易日": pd.Series(True, index=s.index)}
    for lab, m in groups.items():
        m = m.fillna(False).astype(bool)
        row = {"指数": name, "条件": lab, "天数": int(m.sum())}
        for n in (20, 60, 120):
            f = (c.shift(-n) / c - 1)[m].dropna()
            row[f"{n}天上涨比例"] = (f > 0).mean() if len(f) else np.nan
            row[f"{n}天平均"] = f.mean() if len(f) else np.nan
        dd = (s.close[::-1].rolling(60, min_periods=1).min()[::-1].shift(-1) / c - 1)[m].dropna()
        row["60天内再跌中位"] = dd.median() if len(dd) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    for name in INDEX:
        T = forward_table(name)
        cols = ["条件", "天数", "20天上涨比例", "60天上涨比例", "60天平均", "120天上涨比例", "120天平均", "60天内再跌中位"]
        print(f"\n===== {name}（2005 年以来）=====")
        print(T[cols].round(3).to_string(index=False))
        s = signals(name)
        print("最新：", s.index[-1].date(), "情绪", round(s["情绪"].iloc[-1], 1), "温度", round(s["温度"].iloc[-1], 1),
              "波动率", s["波动率"].iloc[-1], "最近★B", s.index[s["★B"]].max().date() if s["★B"].any() else None,
              "最近★S/▽S", s.index[s["★S"] | s["▽S"]].max().date())
        evs = s.index[s["★B"] & (s.index >= "2005-01-01")]
        print("★B 日期（按月合并）：", sorted({d.strftime("%Y-%m") for d in evs}))
