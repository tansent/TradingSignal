"""每日更新：从通达信本地数据 + 公开数据重算全部指标，输出网页用的数据文件到 report_data/。

用法：python daily/update.py
数据：通达信本地日线（D:/TDXapp/vipdoc）；通达信缺最近的交易日时由 daily/sina_fill.py 用新浪数据补上。
好几天没运行时，指数、融资、ETF 会自动补齐，个股（全市场宽度、温度）需要通达信「盘后数据下载」补齐。
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["NO_PROXY"] = "*"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "daily"))

import build_breadth  # noqa: E402
import dingdi  # noqa: E402
import etf_shares  # noqa: E402
import margin_turnover  # noqa: E402
import temperature  # noqa: E402
from overheat_eval import heat  # noqa: E402
from level3_buy import level3  # noqa: E402
from tdx_reader import read_day  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "report_data"
V = Path("D:/TDXapp/vipdoc")
INDEXES = [("上证50", "sh000016", "市值"), ("沪深300", "sh000300", "市值"), ("中证500", "sh000905", "市值"),
           ("中证1000", "sh000852", "市值"), ("上证指数", "sh000001", "板块"), ("深证成指", "sz399001", "板块"),
           ("创业板指", "sz399006", "板块"), ("科创50", "sh000688", "板块")]
PRICE_KEYS = {"上证指数": "sh", "沪深300": "hs300", "深证成指": "sz", "创业板指": "cyb", "科创50": "kc50"}
SENTI_KEYS = {"上证50": "s_sz50", "沪深300": "s_hs300", "中证500": "s_zz500", "中证1000": "s_zz1000",
              "上证指数": "s_sh", "深证成指": "s_sz", "创业板指": "s_cyb", "科创50": "s_kc50"}
# 信号：dingdi 列名、简写、全称、网页图上的代码（按显示优先级排列）
SIGNALS = [("底部三级", "★★B", "★★B 三级底部", "B3"), ("顶部", "★S", "★S 二级顶部", "S"), ("顶部一级", "▽S", "▽S 一级顶部", "S1"),
           ("底部二级", "★B", "★B 二级底部", "B2"), ("底部一级", "△B", "△B 一级底部", "B1"), ("破月BBI", "破", "破 跌破月线BBI", "M")]
ETF_SINA = ["sh510300", "sh510310", "sh510330", "sz159919", "sh510050", "sh510500", "sh512100", "sz159915",
            "sh588000", "sh510180"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def update_margin_sse():
    import akshare as ak
    f = DATA / "margin_sse.csv"
    old = pd.read_csv(f, parse_dates=["date"])
    start = (old.date.max() - pd.Timedelta(days=20)).strftime("%Y%m%d")
    new = ak.stock_margin_sse(start_date=start, end_date=pd.Timestamp.today().strftime("%Y%m%d"))
    new["date"] = pd.to_datetime(new["信用交易日期"].astype(str))
    new = new[["date", "融资余额", "融资买入额", "融资融券余额"]]
    pd.concat([old, new]).drop_duplicates("date", keep="last").sort_values("date").to_csv(f, index=False)


def update_etf_daily():
    import akshare as ak
    (DATA / "etf").mkdir(exist_ok=True)
    for code in ETF_SINA:
        try:
            ak.fund_etf_hist_sina(symbol=code).to_csv(DATA / "etf" / f"{code}.csv", index=False)
        except Exception as e:  # noqa: BLE001
            log("ETF 日线失败", code, repr(e)[:80])


def etf_net_creation() -> pd.DataFrame:
    """宽基 ETF 每日净申购（亿元）= 份额变化 × 当日收盘价；份额单日变化超过 30% 视为拆分，记 0。"""
    f = DATA / "etf_shares_sse.csv"
    if not f.exists():
        return pd.DataFrame(columns=["net_yi", "net_300_yi"], dtype=float)
    s = pd.read_csv(f, dtype={"code": str, "date": str})
    s["date"] = pd.to_datetime(s["date"])
    p = s.pivot(index="date", columns="code", values="shares").sort_index()
    px = {}
    for code in p.columns:
        f = DATA / "etf" / f"sh{code}.csv"
        if f.exists():
            px[code] = pd.read_csv(f, parse_dates=["date"]).set_index("date")["close"]
    px = pd.DataFrame(px).reindex(p.index).ffill()
    d = p.diff()
    d = d.where((d.abs() / p.shift()).fillna(0) <= 0.3, 0)
    net = d * px / 1e8
    out = pd.DataFrame({"net_yi": net.sum(axis=1, min_count=1),
                        "net_300_yi": net[[c for c in ("510300", "510310", "510330") if c in net]].sum(axis=1, min_count=1)})
    return out


def day_signals(r: pd.DataFrame, d) -> list[str]:
    """某天出现的信号简写；★★B 当天不再重复列出 ★B。"""
    out = [short for col, short, _, _ in SIGNALS if r.loc[d, col]]
    return [x for x in out if not (x == "★B" and "★★B" in out)]


def iso(d) -> str | None:
    return None if pd.isna(d) else pd.Timestamp(d).strftime("%Y-%m-%d")


def clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def write(name: str, rows: list[dict]):
    OUT.mkdir(exist_ok=True)
    rows = [{k: clean(v) for k, v in r.items()} for r in rows]
    (OUT / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    log(f"写出 {name}.json：{len(rows)} 行")


def main():
    # 通达信没有最近一个交易日的数据时（没开着、没写盘），用新浪收盘快照补上；--wait 参数保留但不再需要等待
    try:
        import sina_fill
        log(sina_fill.fill())
    except Exception as e:  # noqa: BLE001
        log("新浪补充数据失败，按通达信已有数据计算：", repr(e)[:120])
    tdx = read_day(V / "sh" / "lday" / "sh000001.day")
    asof = tdx.index.max()
    log("通达信数据最新日期", asof.date())

    log("1/6 全市场宽度"); build_breadth.main()
    log("2/6 融资数据")
    try:
        update_margin_sse(); margin_turnover.download()
    except Exception as e:  # noqa: BLE001
        log("融资数据更新失败，沿用旧数据", repr(e)[:100])
    log("3/6 情绪温度、过热分数")
    t = temperature.temperature(); t.to_parquet(DATA / "temperature_v1.parquet")
    mt = margin_turnover.indicators(); mt.to_parquet(DATA / "margin_turnover.parquet")
    h = heat()
    log("4/6 ETF")
    update_etf_daily()
    if not os.environ.get("SKIP_ETF_SHARES"):
        etf_start = asof - pd.Timedelta(days=30)
        if (DATA / "etf_shares_sse.csv").exists():  # 好几周没运行时从上次的日期接着补
            etf_start = min(etf_start, pd.to_datetime(pd.read_csv(DATA / "etf_shares_sse.csv", dtype={"date": str})["date"]).max())
        etf_shares.update(start=etf_start.strftime("%Y-%m-%d"))
    from etf_flow import panel
    ep = panel()
    etf = pd.DataFrame({"ratio_med": ep.median(axis=1), "share_2x": (ep >= 2).sum(axis=1) / ep.notna().sum(axis=1) * 100})
    etf = etf.join(etf_net_creation(), how="left")
    log("5/6 指数信号")
    b = pd.read_parquet(DATA / "breadth_daily.parquet")
    a = b[b.board == "all"].set_index("date").sort_index()
    bd = a.index.max()
    gap_days = int((tdx.index > bd).sum())
    if gap_days:
        log(f"警告：全市场个股数据只到 {bd.date()}，比指数少 {gap_days} 个交易日；温度、涨跌家数、涨跌停暂停在那天。"
            "打开通达信 → 盘后数据下载 补齐后重新运行即可")
    sig = {}
    idx_rows = []
    for name, code, group in INDEXES:
        df = read_day(V / code[:2] / "lday" / f"{code}.day")
        r = dingdi.compute(df)
        r["底部三级"] = level3(r["底部二级"], t["中线温度"], etf["net_yi"])
        sig[name] = (df, r)
        last = r.index.max()
        today = "、".join(day_signals(r, last)) or "无"
        lastd = lambda col: iso(r.index[r[col]].max()) if r[col].any() else None  # noqa: E731
        idx_rows.append({
            "name": name, "group": group, "date": iso(last), "close": df.close.iloc[-1], "pct": (df.close.iloc[-1] / df.close.iloc[-2] - 1) * 100,
            "senti": r["情绪"].iloc[-1], "signal": today, "last_top": lastd("顶部"), "last_s1": lastd("顶部一级"), "last_b2": lastd("底部二级"),
            "last_b1": lastd("底部一级"), "last_b3": lastd("底部三级"), "last_m": lastd("破月BBI"), "dist_high": (df.close.iloc[-1] / df.high.rolling(250).max().iloc[-1] - 1) * 100,
            "adv": df["adv"].iloc[-1], "dec": df["dec"].iloc[-1],
        })
    log("6/6 写出数据文件")
    write("indexes", idx_rows)

    m_last = mt["融资余额(亿)"].dropna()
    etf_l = etf.loc[:asof].dropna(subset=["ratio_med"])
    net5 = etf["net_yi"].loc[:asof].tail(5).sum(min_count=1)
    ar = a.reindex([asof]).iloc[0]  # 个股数据缺天时为空值
    write("summary", [{
        "date": iso(asof), "mid_temp": t["中线温度"].get(asof), "short_temp": t["短线温度_分位"].get(asof),
        "heat": h.H.reindex([asof]).iloc[0] * 100, "margin_yi": m_last.iloc[-1], "margin_date": iso(m_last.index[-1]),
        "amount_yi": ar["amount"] / 1e8, "up": ar["up"], "down": ar["down"],
        "zt": ar["zt"], "dt": ar["dt"], "zb": ar["zb"], "max_lb": ar["max_lb"],
        "breadth_date": iso(bd), "gap_days": gap_days,
        "etf_ratio": etf_l["ratio_med"].iloc[-1], "etf_share_2x": etf_l["share_2x"].iloc[-1],
        "etf_net_yi": etf["net_yi"].reindex([asof]).iloc[0], "etf_net5_yi": net5,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
    }])

    days = tdx.index[-250:]
    hist = pd.DataFrame(index=days)
    for name, key in PRICE_KEYS.items():
        hist[key] = sig[name][0].close
    hist["mid_temp"] = t["中线温度"]
    hist["short_temp"] = t["短线温度_分位"]
    hist["heat"] = h.H * 100
    for name, key in SENTI_KEYS.items():
        hist[key] = sig[name][1]["情绪"]
    code = lambda r: np.select([r[c] for c, *_ in SIGNALS], [k for *_, k in SIGNALS], "")  # noqa: E731
    hist["bbi_sh"] = sig["上证指数"][1]["月线BBI"]
    hist["sig_sh"] = pd.Series(code(sig["上证指数"][1]), index=sig["上证指数"][1].index)
    hist["sig_hs300"] = pd.Series(code(sig["沪深300"][1]), index=sig["沪深300"][1].index)
    write("history", [{"date": iso(d), **row} for d, row in hist.iterrows()])

    mg = mt["融资余额(亿)"]
    tab = []
    for d in tdx.index[-20:][::-1]:
        sigs = [f"{n}{x}" for n, (_, r) in sig.items() if d in r.index for x in day_signals(r, d)]
        shc = sig["上证指数"][0].close
        tab.append({"date": iso(d), "sh": shc.get(d), "sh_pct": (shc.get(d) / shc.shift().get(d) - 1) * 100,
                    "mid_temp": t["中线温度"].get(d), "short_temp": t["短线温度_分位"].get(d), "heat": h.H.get(d, np.nan) * 100,
                    "amount_yi": a["amount"].get(d, np.nan) / 1e8, "up": a["up"].get(d), "down": a["down"].get(d),
                    "zt": a["zt"].get(d), "dt": a["dt"].get(d), "margin_yi": mg.get(d), "margin_chg_yi": mg.diff().get(d),
                    "etf_net_yi": etf["net_yi"].get(d), "signals": "、".join(sigs) if sigs else "—"})
    write("daily", tab)

    br = a.loc[a.index[-60:]]
    write("breadth", [{"date": iso(d), "up": r["up"], "down": r["down"], "zt": r["zt"], "dt": r["dt"], "zb": r["zb"],
                       "max_lb": r["max_lb"], "amount_yi": r["amount"] / 1e8, "nh": r["nh60"], "nl": r["nl60"]}
                      for d, r in br.iterrows()])

    e = etf.loc[:asof].tail(120)
    write("etf", [{"date": iso(d), **r} for d, r in e.iterrows()])

    ev = []
    for name, (df, r) in sig.items():
        for col, _, typ, _ in SIGNALS:
            for d in r.index[r[col] & (r.index >= "2024-01-01")]:
                ev.append({"date": iso(d), "name": name, "type": typ})
    ev.sort(key=lambda x: (x["date"], x["name"]), reverse=True)
    write("signals", ev)
    log("完成")


if __name__ == "__main__":
    main()
