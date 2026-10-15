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
import weekly_zone  # noqa: E402
import national_team  # noqa: E402
import crowding  # noqa: E402
import red_top  # noqa: E402
import icepoint  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "report_data"
V = Path("D:/TDXapp/vipdoc")
INDEXES = [("上证50", "sh000016", "市值"), ("沪深300", "sh000300", "市值"), ("中证500", "sh000905", "市值"),
           ("中证1000", "sh000852", "市值"), ("上证指数", "sh000001", "板块"), ("深证成指", "sz399001", "板块"),
           ("创业板指", "sz399006", "板块"), ("科创50", "sh000688", "板块")]
# 8 个指数之外的品种：数据来自新浪 ETF 日线（存 data/etf_extra，不进宽基 ETF 成交放量统计），取不到时用通达信
EXTRA = [("红利低波ETF", "sh512890", "红利")]
PRICE_KEYS = {"上证指数": "sh", "沪深300": "hs300", "深证成指": "sz", "创业板指": "cyb", "科创50": "kc50"}
SENTI_KEYS = {"红利低波ETF": "s_dlv", "上证50": "s_sz50", "沪深300": "s_hs300", "中证500": "s_zz500", "中证1000": "s_zz1000",
              "上证指数": "s_sh", "深证成指": "s_sz", "创业板指": "s_cyb", "科创50": "s_kc50"}
# 信号：dingdi 列名、简写、全称、网页图上的代码（按显示优先级排列）
SIGNALS = [("底部三级", "★★B", "★★B 三级底部", "B3"), ("顶部", "★S", "★S 二级顶部", "S"), ("顶部一级", "▽S", "▽S 一级顶部", "S1"),
           ("乖离顶", "乖S", "乖S 乖离顶（天量收阴）", "SG"),
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


def load_extra(code: str, asof) -> pd.DataFrame:
    f = DATA / "etf_extra" / f"{code}.csv"
    try:
        import akshare as ak
        f.parent.mkdir(exist_ok=True)
        ak.fund_etf_hist_sina(symbol=code).to_csv(f, index=False)
    except Exception as e:  # noqa: BLE001
        log("ETF 日线下载失败，用已有数据或通达信", code, repr(e)[:80])
    if f.exists():
        df = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
        df = df[["open", "high", "low", "close", "volume", "amount"]].astype(float)
    else:
        df = read_day(V / code[:2] / "lday" / f"{code}.day")
    now = pd.Timestamp.now()
    if df.index.max() < asof == now.normalize() and now.hour * 60 + now.minute >= 15 * 60 + 5:
        # 收盘后新浪历史日线可能还没有今天：用 ETF 实时行情里今天的开高低收补上
        try:
            import akshare as ak
            r = ak.fund_etf_category_sina(symbol="ETF基金").set_index("代码").loc[code]
            df.loc[asof] = [float(r["今开"]), float(r["最高"]), float(r["最低"]), float(r["最新价"]), float(r["成交量"]), float(r["成交额"])]
        except Exception as e:  # noqa: BLE001
            log("ETF 当天行情补充失败", code, repr(e)[:80])
    return df[df.index <= asof]      # 盘中运行时新浪可能给出今天未完成的 K 线


def us_section(asof) -> dict:
    """美股：标普500、纳指的情绪、波动率、温度和顶底信号（research/us_market.py），写出 us.json。
    返回给 history 用的序列；任何一步失败都不影响 A 股部分。"""
    import us_market as um
    for sid in ("VIXCLS", "VXNCLS", "NFCI", "DGS10"):
        um.fred(sid, refresh=True)
    for sym in (".INX", ".IXIC"):
        um.sina_index(sym, refresh=True)
    rows, hist = [], {}
    for name, key, vol_name, etf_code, etf_name in (("标普500", "spx", "VIX", "sh513500", "标普500ETF"),
                                                    ("纳指", "ndx", "VXN", "sh513100", "纳指ETF")):
        s = um.signals(name)
        last = s.index[-1]
        labs = [("★B", "★B"), ("△B", "△B"), ("★S", "★S"), ("▽S", "▽S"), ("破月BBI", "破")]
        today = [short for col, short in labs if s.loc[last, col]]
        lastd = lambda cols: iso(s.index[s[cols].any(axis=1)].max()) if s[cols].any(axis=None) else None  # noqa: E731
        etf = load_extra(etf_code, asof)
        rows.append({"name": name, "date": iso(last), "close": s.close.iloc[-1], "pct": (s.close.iloc[-1] / s.close.iloc[-2] - 1) * 100,
                     "senti": s["情绪"].iloc[-1], "vol_name": vol_name, "vol": s["波动率"].iloc[-1], "temp": s["温度"].iloc[-1],
                     "signal": "、".join(today) or "无", "last_b": lastd(["△B", "★B"]), "last_b2": lastd(["★B"]), "last_s": lastd(["★S", "▽S"]),
                     "dist_high": (s.close.iloc[-1] / s.close.rolling(250).max().iloc[-1] - 1) * 100,
                     "etf_name": etf_name, "etf_code": etf_code[2:], "etf_close": etf.close.iloc[-1],
                     "etf_pct": (etf.close.iloc[-1] / etf.close.iloc[-2] - 1) * 100, "etf_date": iso(etf.index[-1])})
        hist[f"s_{key}"] = s["情绪"]
    write("us", rows)
    return hist


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
    try:                       # A股拥挤度：成交额前 5% 个股的成交额占比（research/crowding.py）
        crowd = crowding.build()["crowd"]
    except Exception as e:  # noqa: BLE001
        log("拥挤度计算失败，本次日报不含拥挤度：", repr(e)[:100])
        crowd = pd.Series(dtype=float)
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
    nt_eps = national_team.episodes(etf["net_yi"], read_day(V / "sh" / "lday" / "sh000300.day"))
    nt_ep = nt_eps[-1] if nt_eps else None      # 最近一轮国家队入场
    national_team.export_tdx(etf["net_yi"], read_day(V / "sh" / "lday" / "sh000300.day"), DATA / "tdx_nt.csv")  # 给通达信 DLL
    if len(crowd):
        crowding.export_tdx(crowd, read_day(V / "sh" / "lday" / "sh000300.day"), DATA / "tdx_crowd.csv")  # 给通达信 DLL（◆S、拥挤度副图）
    mgd = mt["融资余额(亿)"].reindex(tdx.index).ffill()
    mg20k = red_top.margin_chg20_known(mgd)          # 融资余额 20 日变化（收盘时已知，即前一天的数）
    crowding.export_margin_tdx(mg20k, mgd, DATA / "tdx_margin.csv")   # 给通达信 DLL（红色 ◆S 的 M 顶条件）
    log("5/6 指数信号")
    b = pd.read_parquet(DATA / "breadth_daily.parquet")
    a = b[b.board == "all"].set_index("date").sort_index()
    ice = icepoint.icepoint(a)       # 短线冰点：跌停 > 涨停 且 跌停 ≥ 20 家
    icepoint.export_extra_tdx(a, t, DATA / "tdx_extra.csv")   # 给通达信 DLL（21~28 号：涨跌停家数、冰点、温度）
    bd = a.index.max()
    gap_days = int((tdx.index > bd).sum())
    if gap_days:
        log(f"警告：全市场个股数据只到 {bd.date()}，比指数少 {gap_days} 个交易日；温度、涨跌家数、涨跌停暂停在那天。"
            "打开通达信 → 盘后数据下载 补齐后重新运行即可")
    sig = {}
    idx_rows = []
    for name, code, group in INDEXES + EXTRA:
        df = load_extra(code, asof) if group == "红利" else read_day(V / code[:2] / "lday" / f"{code}.day")
        r = dingdi.compute(df)
        r["底部三级"] = level3(r["底部二级"], t["中线温度"], etf["net_yi"])
        sig[name] = (df, r)
        wz = weekly_zone.current(df)
        last = r.index.max()
        today = "、".join(day_signals(r, last)) or "无"
        lastd = lambda col: iso(r.index[r[col]].max()) if r[col].any() else None  # noqa: E731
        idx_rows.append({
            "name": name, "group": group, "date": iso(last), "close": df.close.iloc[-1], "pct": (df.close.iloc[-1] / df.close.iloc[-2] - 1) * 100,
            "senti": r["情绪"].iloc[-1], "signal": today, "last_top": lastd("顶部"), "last_s1": lastd("顶部一级"), "last_b2": lastd("底部二级"),
            "last_b1": lastd("底部一级"), "last_b3": lastd("底部三级"), "last_m": lastd("破月BBI"), "dist_high": (df.close.iloc[-1] / df.high.rolling(250).max().iloc[-1] - 1) * 100,
            "adv": df["adv"].iloc[-1] if "adv" in df else None, "dec": df["dec"].iloc[-1] if "dec" in df else None,
            "wzone": wz["wzone"] or None, "wzone_since": iso(wz["wzone_since"]), "wzone_until": iso(wz["wzone_until"]),
            "wsig": wz["wsig"], "wsig_date": iso(wz["wsig_date"]),
            **({k: (iso(v) if k.endswith("date") else v) for k, v in national_team.defense(df, nt_ep).items()} if nt_ep is not None else {}),
        })
    log("6/6 写出数据文件")
    write("indexes", idx_rows)

    m_last = mt["融资余额(亿)"].dropna()
    etf_l = etf.loc[:asof].dropna(subset=["ratio_med"])
    net5 = etf["net_yi"].loc[:asof].tail(5).sum(min_count=1)
    ar = a.reindex([asof]).iloc[0]  # 个股数据缺天时为空值
    # 拥挤度 ≥48% 且沪深300 在一年高点 5% 以内：历史上 4 次里 3 次是阶段顶部（2017-11、2021-01、2026-05；2014-12 例外）
    hs = sig["沪深300"][0]
    hs_dist = hs.close.iloc[-1] / hs.high.rolling(250).max().iloc[-1] - 1
    crowd_now = crowd.get(asof, np.nan)
    tz = crowding.top_zone(crowd, hs) if len(crowd) else pd.DataFrame(columns=["zone", "mark"])
    cz = tz.reindex([asof]).iloc[0] if len(tz) else pd.Series({"zone": 0, "mark": 0})
    red_s, red_ser = [], {}      # 红色 ◆S：二次冲顶（再涨≥5%+拥挤≥50%+情绪≥85+放量）或 M 顶（30 天内回到前高附近、拥挤不降、量和融资不减）
    if len(tz):
        div = {n for n, _, g in EXTRA if g == "红利"}     # ◆S 不适用于红利类：拥挤度高之后红利低波反而偏强
        for name, (df, r) in sig.items():
            if name not in div and "amount" in df and asof in df.index:
                two = red_top.red_signal(df, r, crowd, tz["mark"]).reindex(df.index, fill_value=False)
                mtop = red_top.red_signal_mtop(df, r, crowd, tz["mark"], mg20k).reindex(df.index, fill_value=False)
                red_ser[name] = two | mtop
                kinds = [k for k, s in (("二次冲顶", two), ("M顶", mtop)) if bool(s.get(asof, False))]
                if kinds:
                    red_s.append(f"{name}（{'、'.join(kinds)}）")
    # 浅蓝 B：全市场冰点 + 该指数前 20 天无 S（含红色 ◆S）、前 5 天无 B、离 20 日高点 ≥5%、10 天内第一个（research/icepoint.py）
    ice_chart = {n: icepoint.chart_ice(df, r, ice, red_ser.get(n)) for n, (df, r) in sig.items() if n not in {x for x, *_ in EXTRA}}
    ice_idx = [n for n, s in ice_chart.items() if s.get(asof, False)]
    write("summary", [{
        "date": iso(asof), "mid_temp": t["中线温度"].get(asof), "short_temp": t["短线温度_分位"].get(asof),
        "heat": h.H.reindex([asof]).iloc[0] * 100, "margin_yi": m_last.iloc[-1], "margin_date": iso(m_last.index[-1]),
        "amount_yi": ar["amount"] / 1e8, "up": ar["up"], "down": ar["down"],
        "zt": ar["zt"], "dt": ar["dt"], "zb": ar["zb"], "max_lb": ar["max_lb"],
        "ice": bool(ice.get(asof, False)), "ice_last": iso(ice[ice].index.max()) if ice.any() else None,
        "ice_idx": "、".join(ice_idx) or None,
        "breadth_date": iso(bd), "gap_days": gap_days,
        "crowd": crowd_now, "crowd_max20": crowd.loc[:asof].tail(20).max() if len(crowd) else None,
        "hs300_dist_high": hs_dist * 100, "crowd_warn": bool(cz["zone"] == 2), "crowd_mark": bool(cz["mark"] == 1),
        "red_s": "、".join(red_s) or None,
        "etf_ratio": etf_l["ratio_med"].iloc[-1], "etf_share_2x": etf_l["share_2x"].iloc[-1],
        "etf_net_yi": etf["net_yi"].reindex([asof]).iloc[0], "etf_net5_yi": net5,
        "nt_first": iso(nt_ep.index[0]) if nt_ep is not None else None, "nt_last": iso(nt_ep.index[-1]) if nt_ep is not None else None,
        "nt_days": len(nt_ep) if nt_ep is not None else 0, "nt_total_yi": float(nt_ep.net.sum()) if nt_ep is not None else None,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
    }])
    alerts = [f"{r['name']} {k}" for r in idx_rows if r["group"] != "红利"
              for k in ("★★B", "★B") if k in (r["signal"] or "").split("、")]
    if cz["mark"] == 1:
        alerts.append("全市场 橙色◆S 抱团顶")
    if red_s:
        alerts.append("红色◆S：" + "、".join(red_s))
    if alerts:
        log("★ 重要提醒：" + "；".join(alerts))
    if ice.get(asof, False):
        log(f"全市场跌停 {int(ar['dt'])} 家 > 涨停 {int(ar['zt'])} 家；"
            + (f"浅蓝 B 冰点：{'、'.join(ice_idx)}（只作短线参考）" if ice_idx else "各指数都不符合浅蓝 B 的条件"))

    days = tdx.index[-250:]
    hist = pd.DataFrame(index=days)
    for name, key in PRICE_KEYS.items():
        hist[key] = sig[name][0].close
    hist["mid_temp"] = t["中线温度"]
    hist["short_temp"] = t["短线温度_分位"]
    hist["heat"] = h.H * 100
    hist["crowd"] = crowd
    for name, key in SENTI_KEYS.items():
        hist[key] = sig[name][1]["情绪"]
    code = lambda r: np.select([r[c] for c, *_ in SIGNALS], [k for *_, k in SIGNALS], "")  # noqa: E731
    hist["bbi_sh"] = sig["上证指数"][1]["月线BBI"]
    zsh = weekly_zone.zones(sig["上证指数"][0])
    hist["wz_sh"] = zsh["z_top"].astype(int) - zsh["z_bot"].astype(int)   # 1 周线顶部区域，-1 底部区域，0 中间
    hist["sig_sh"] = pd.Series(code(sig["上证指数"][1]), index=sig["上证指数"][1].index)
    hist["sig_hs300"] = pd.Series(code(sig["沪深300"][1]), index=sig["沪深300"][1].index)
    try:
        for k, v in us_section(asof).items():
            hist[k] = v.reindex(hist.index, method="ffill")      # 美股按日期对齐到 A 股交易日（取当天或之前最近一个美股交易日）
    except Exception as e:  # noqa: BLE001
        log("美股数据更新失败，本次日报不含美股：", repr(e)[:120])
    write("history", [{"date": iso(d), **row} for d, row in hist.iterrows()])

    mg = mt["融资余额(亿)"]
    tab = []
    for d in tdx.index[-20:][::-1]:
        sigs = [f"{n}冰点" for n, s in ice_chart.items() if s.get(d, False)] + [f"{n}{x}" for n, (_, r) in sig.items() if d in r.index for x in day_signals(r, d)]
        shc = sig["上证指数"][0].close
        tab.append({"date": iso(d), "sh": shc.get(d), "sh_pct": (shc.get(d) / shc.shift().get(d) - 1) * 100,
                    "mid_temp": t["中线温度"].get(d), "short_temp": t["短线温度_分位"].get(d), "heat": h.H.get(d, np.nan) * 100,
                    "crowd": crowd.get(d), "amount_yi": a["amount"].get(d, np.nan) / 1e8, "up": a["up"].get(d), "down": a["down"].get(d),
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
    try:
        import events                   # 事件预警：财经日历（判断由定时任务写入）
        events.write(events.build())
    except Exception as e:  # noqa: BLE001
        log("事件日历更新失败，本次日报不更新事件预警：", repr(e)[:120])
    log("完成")


if __name__ == "__main__":
    main()
