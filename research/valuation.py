"""指数估值分位（2026-10-17）：只在极端时有意义（老王："平时没必要看估值，极高或极低时配合其它条件"）。

数据：
  中证指数官网的滚动市盈率（日度，akshare stock_zh_index_hist_csindex）：上证50、沪深300、中证500、中证1000、上证指数、红利低波；
  蛋卷基金的估值历史（周度，2016 年起）：深证成指、创业板指用市盈率，科创50 用市净率（亏损公司多，市盈率失真：
  同一天中证给 63、蛋卷给 111）。都存在 data/valuation/。
分位用"当时可知"的算法：只和当天之前的历史比（至少 3 年），和 App 里用全部历史算的"估值分位"不同（那种有事后偏差）。

research/narrative_tests.py 的检验（5 个中证指数，2011 年起，当时可知的 PE 分位）：
  估值和之后 60 天涨跌几乎无关（秩相关 -0.07），1 年 -0.26、3 年 -0.47；
  最便宜 20%：3 年后上涨 75%、中位 +22%；最贵 20%：1 年后上涨只有 25%，3 年后 33%、中位 -15%。
  B 点（△B/★B）在便宜时（分位 ≤30%）：60 天上涨 49%（和平时一样），3 年上涨 91%、中位 +26%；
  在贵时（≥70%）：1 年上涨 27%，3 年 23%、中位 -21%（多是反弹）。
  S 点（★S/▽S/乖S）在贵时：1 年后下跌 59%、中位 -5.8%；估值中间时 1 年后下跌只有 27%（多是阶段回调）。
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "data" / "valuation"
# 日报里的指数 → 数据来源
CSI = {"上证50": "000016", "沪深300": "000300", "中证500": "000905", "中证1000": "000852",
       "上证指数": "000001", "红利低波ETF": "H30269", "通信设备": "931160"}
DANJUAN = {"深证成指": ("SZ399001", "pe"), "创业板指": ("SZ399006", "pe"), "科创50": ("SH000688", "pb")}
CHEAP, RICH = 0.2, 0.8
MIN_YEARS = 3           # 至少 3 年历史才算分位


def pe_csi(code: str, refresh: bool = False) -> pd.Series:
    f = VAL / f"pe_{code}.csv"
    if refresh or not f.exists():
        os.environ["NO_PROXY"] = "*"
        import akshare as ak
        d = ak.stock_zh_index_hist_csindex(symbol=code, start_date="20000101", end_date=pd.Timestamp.now().strftime("%Y%m%d"))
        d = d[["日期", "滚动市盈率"]].rename(columns={"日期": "date", "滚动市盈率": "pe"})
        VAL.mkdir(parents=True, exist_ok=True)
        d[d.pe > 0].to_csv(f, index=False)
    return pd.read_csv(f, parse_dates=["date"]).set_index("date")["pe"].dropna()


def danjuan(code: str, kind: str, refresh: bool = False) -> pd.Series:
    f = VAL / f"dj_{code}_{kind}.csv"
    if refresh or not f.exists():
        import requests
        h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
        j = requests.get(f"https://danjuanfunds.com/djapi/index_eva/{kind}_history/{code}?day=all", headers=h, timeout=15).json()
        d = pd.DataFrame(j["data"][f"index_eva_{kind}_growths"])
        d["date"] = (pd.to_datetime(d.ts, unit="ms") + pd.Timedelta(hours=8)).dt.normalize()
        VAL.mkdir(parents=True, exist_ok=True)
        d[["date", kind]].rename(columns={kind: "pe"}).to_csv(f, index=False)
    return pd.read_csv(f, parse_dates=["date"]).set_index("date")["pe"].dropna()


def pct_asof(s: pd.Series, min_years: float = MIN_YEARS) -> pd.Series:
    """当时可知的分位：当天的值在它之前所有历史里排第几（0 = 历史最低）；历史不足 min_years 年时为空。"""
    a, idx = s.to_numpy(), s.index
    out = np.full(len(a), np.nan)
    for i in range(1, len(a)):
        if (idx[i] - idx[0]).days >= min_years * 365:
            out[i] = (a[:i] < a[i]).mean()
    return pd.Series(out, index=idx)


def series(name: str, refresh: bool = False) -> tuple[pd.Series, str] | None:
    """(估值序列, 指标名)；取不到返回 None。更新失败时用本地已有的数据。"""
    for r in (refresh, False) if refresh else (False,):
        try:
            if name in CSI:
                return pe_csi(CSI[name], r), "市盈率"
            if name in DANJUAN:
                code, kind = DANJUAN[name]
                return danjuan(code, kind, r), "市盈率" if kind == "pe" else "市净率"
            return None
        except Exception as e:  # noqa: BLE001
            print("估值数据更新失败，用本地数据：", name, repr(e)[:80])
    return None


def current(names, refresh: bool = False) -> dict:
    """每个指数最新的估值、当时可知分位、数据日期、历史年数。历史不足 3 年或取不到时不返回该指数。"""
    out = {}
    for n in names:
        got = series(n, refresh)
        if got is None:
            continue
        s, metric = got
        years = (s.index[-1] - s.index[0]).days / 365.25
        if years < MIN_YEARS:
            continue
        out[n] = {"val_metric": metric, "val": float(s.iloc[-1]), "val_pct": float((s.iloc[:-1] < s.iloc[-1]).mean()),
                  "val_date": s.index[-1].strftime("%Y-%m-%d"), "val_years": round(years, 1)}
    return out


# 给通达信 DLL 的估值通道：每个指数一个文件 data/tdx_val_<编号>.csv，每行 日期,低分位比例,高分位比例,当前分位(0-100)
# 通道价位 = 当天收盘 × 比例 = 每股盈利 × 历史分位市盈率，只随盈利（和分位门槛）变，不随当天涨跌变；
# 用比例而不是点位，跟踪同一指数的 ETF 图上也能直接用。编号要和 DINGDI 里的"估值号"一致。
TDX_ID = {"上证指数": 1, "上证50": 16, "沪深300": 300, "中证500": 905, "中证1000": 852,
          "深证成指": 399001, "创业板指": 399006, "科创50": 688, "红利低波ETF": 30269, "通信设备": 931160}


def band(name: str, close: pd.Series | None = None) -> pd.DataFrame | None:
    """每天的低、高分位估值和当前估值（当时可知的分位门槛：只用之前至少 3 年的历史）。
    蛋卷的周度数据先按指数收盘价换算成日度（同一周内每股盈利不变）。"""
    got = series(name)
    if got is None:
        return None
    s, _ = got
    q = pd.DataFrame({"v": s})
    hist_ok = (q.index - q.index[0]).days >= MIN_YEARS * 365
    q["lo"] = q.v.expanding().quantile(CHEAP).shift(1).where(hist_ok)
    q["hi"] = q.v.expanding().quantile(RICH).shift(1).where(hist_ok)
    q["pct"] = pct_asof(q.v)
    if name in DANJUAN:
        if close is None:
            return None
        c = close.dropna()
        e = (c.reindex(q.index, method="ffill") / q.v).reindex(c.index, method="ffill")      # 每股盈利，按周更新
        d = pd.DataFrame({"v": c / e, "lo": q.lo.reindex(c.index, method="ffill"),
                          "hi": q.hi.reindex(c.index, method="ffill"), "pct": q.pct.reindex(c.index, method="ffill")})
        return d[d.index >= q.index[0]]
    return q


def export_tdx(closes: dict, out_dir) -> int:
    n = 0
    for name, i in TDX_ID.items():
        d = band(name, closes.get(name))
        if d is None:
            continue
        d = d.dropna(subset=["lo", "hi"])
        lines = [f"{t:%Y%m%d},{r.lo / r.v:.5f},{r.hi / r.v:.5f},{r.pct * 100:.1f}" for t, r in d.iterrows()]
        with open(Path(out_dir) / f"tdx_val_{i}.csv", "w", encoding="ascii", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        n += 1
    return n


def cn10y(refresh: bool = False) -> pd.Series:
    """10 年期国债收益率（%），akshare bond_zh_us_rate，存 data/valuation/cn10y.csv。"""
    f = VAL / "cn10y.csv"
    if refresh or not f.exists():
        os.environ["NO_PROXY"] = "*"
        import akshare as ak
        d = ak.bond_zh_us_rate(start_date="20050101")[["日期", "中国国债收益率10年"]].dropna()
        d.columns = ["date", "y10"]
        VAL.mkdir(parents=True, exist_ok=True)
        d.to_csv(f, index=False)
    return pd.read_csv(f, parse_dates=["date"]).set_index("date")["y10"].dropna()


def erp_now(refresh: bool = False) -> dict:
    """沪深300 股权溢价 = 盈利收益率（1/PE）− 10 年国债收益率（百分点），以及当时可知分位（越高越便宜）。
    检验（2011 年起）：股权溢价最高的 20% 时段 3 年后全部上涨、中位 +22.5%；最低 20% 时 1 年后上涨只有 27%。
    比"股债比"（盈利收益率 ÷ 国债利率）稳定：利率很低时比值会被放大，2026-06 大顶时股债比还有 3.86。"""
    try:
        y = cn10y(refresh)
    except Exception as e:  # noqa: BLE001
        print("国债利率更新失败，用本地数据：", repr(e)[:80])
        y = cn10y(False)
    got = series("沪深300")
    if got is None:
        return {}
    pe, _ = got
    d = pd.DataFrame({"ep": 100 / pe, "y": y.reindex(pe.index, method="ffill")}).dropna()
    e = d.ep - d.y
    return {"erp": float(e.iloc[-1]), "erp_pct": float((e.iloc[:-1] < e.iloc[-1]).mean()),
            "erp_ep": float(d.ep.iloc[-1]), "erp_y10": float(d.y.iloc[-1]), "erp_date": e.index[-1].strftime("%Y-%m-%d")}


def tag(p: float | None) -> str | None:
    if p is None:
        return None
    return "极低" if p <= CHEAP else "极高" if p >= RICH else None
