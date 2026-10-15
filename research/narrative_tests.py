"""检验一段关于"钟摆、叙事、便宜"的交易观点里可以量化的三句话。

1. 叙事透支："叙事打满"之后性价比低、消化时间长。
   127 个通达信研究行业二级指数，前 120 天涨 ≥60% 且创一年新高（同一行业一年内只算第一次），看之后 1/2/3 年。
2. 便宜是硬道理："起点决定涨幅"。
   中证官网的滚动市盈率（2011 年起）和 10 年期国债收益率，算市盈率分位和股债利差（1/PE − 国债收益率）分位，
   看之后 60 天、1 年、3 年、5 年的涨跌（价格指数，不含分红）；再看底部信号（△B/★B）在便宜和不便宜时的差别。
   分位有两种：全样本（事后看，和各种 App 里显示的"估值分位"一样）和当时可知（只用当天之前的数据，至少 3 年）。
3. 重新重合："个股（板块）走势与宽基重新重合，超额收益窗口就关闭了"。
   127 个行业对中证800：前 120 天超额 ≥20% 的行业（正在走自己的叙事），按"与宽基的 60 日相关系数比 60 天前上升/下降"
   和"最近 20 天超额是否走平"分组，看之后 60/120 天的超额收益。每个行业每 20 个交易日取一次样本，减少重叠。
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dingdi  # noqa: E402
from box_break import load  # noqa: E402
from point_line_surface import cfg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "data" / "valuation"
PE_IDX = [("上证50", "000016", "sh000016"), ("沪深300", "000300", "sh000300"), ("中证500", "000905", "sh000905"),
          ("中证1000", "000852", "sh000852"), ("上证指数", "000001", "sh000001")]


def industries():
    return [(f[0], "sh" + f[1]) for f in cfg() if len(f[5]) == 5]


# ---------- 1. 叙事透支 ----------
def overshoot():
    rows, base = [], []
    for name, code in industries():
        d = load(code)
        if d is None:
            continue
        c = d.close
        r120 = c / c.shift(120) - 1
        hit = ((r120 >= 0.6) & (c >= c.rolling(250).max())).to_numpy()
        last = -10**9
        for k in np.flatnonzero(hit):
            first, last = k - last > 250, k
            if not first or c.index[k] < pd.Timestamp("2012-01-01"):
                continue
            f = lambda n: c.iloc[k + n] / c.iloc[k] - 1 if k + n < len(c) else np.nan  # noqa: E731
            rows.append({"行业": name, "日期": c.index[k].date(), "前120天涨幅": r120.iloc[k], "1年": f(250), "2年": f(500),
                         "3年": f(750), "1年内最大回撤": d.low.iloc[k + 1:k + 251].min() / c.iloc[k] - 1})
        base.append(pd.DataFrame({"1年": c.shift(-250) / c - 1, "2年": c.shift(-500) / c - 1,
                                  "3年": c.shift(-750) / c - 1})[c.index >= "2012-01-01"])
    E, B = pd.DataFrame(rows), pd.concat(base)
    up = lambda s: (s.dropna() > 0).mean()  # noqa: E731
    print(f"== 1. 叙事透支：前 120 天涨 ≥60% 且创一年新高，{len(E)} 次，{E['行业'].nunique()} 个行业 ==")
    for col in ("1年", "2年", "3年"):
        print(f"  {col}后：上涨比例 {up(E[col]):.0%}（所有交易日 {up(B[col]):.0%}），中位 {E[col].median():+.1%}"
              f"（所有交易日 {B[col].median():+.1%}），样本 {E[col].notna().sum()}")
    print(f"  1 年内最大回撤中位 {E['1年内最大回撤'].median():.1%}")


# ---------- 2. 便宜是硬道理 ----------
def pe_series(code: str, refresh=False) -> pd.Series:
    f = VAL / f"pe_{code}.csv"
    if refresh or not f.exists():
        os.environ["NO_PROXY"] = "*"
        import akshare as ak
        d = ak.stock_zh_index_hist_csindex(symbol=code, start_date="20000101", end_date=pd.Timestamp.now().strftime("%Y%m%d"))
        d = d[["日期", "滚动市盈率"]].rename(columns={"日期": "date", "滚动市盈率": "pe"})
        VAL.mkdir(parents=True, exist_ok=True)
        d[d.pe > 0].to_csv(f, index=False)
    return pd.read_csv(f, parse_dates=["date"]).set_index("date")["pe"]


def cn10y(refresh=False) -> pd.Series:
    f = VAL / "cn10y.csv"
    if refresh or not f.exists():
        os.environ["NO_PROXY"] = "*"
        import akshare as ak
        d = ak.bond_zh_us_rate(start_date="20050101")[["日期", "中国国债收益率10年"]].dropna()
        d.columns = ["date", "y10"]
        VAL.mkdir(parents=True, exist_ok=True)
        d.to_csv(f, index=False)
    return pd.read_csv(f, parse_dates=["date"]).set_index("date")["y10"]


def expanding_pct(s: pd.Series, min_n=750) -> pd.Series:
    a = s.to_numpy()
    out = np.full(len(a), np.nan)
    for i in range(min_n, len(a)):
        h = a[:i]
        out[i] = (h < a[i]).mean()
    return pd.Series(out, index=s.index)


HZ = {"60天": 60, "1年": 250, "3年": 750, "5年": 1250}


def valuation_panel(refresh=False) -> pd.DataFrame:
    y10 = cn10y(refresh)
    parts = []
    for name, csi, tdx in PE_IDX:
        d = load(tdx)
        pe = pe_series(csi, refresh).reindex(d.index).ffill(limit=5)
        c = d.close
        x = pd.DataFrame({"指数": name, "close": c, "pe": pe, "y10": y10.reindex(d.index).ffill(limit=10)}).dropna()
        x["erp"] = 100 / x.pe - x.y10                                  # 股债利差（百分点），越高越便宜
        x["PE分位(事后)"] = x.pe.rank(pct=True)
        x["PE分位(当时)"] = expanding_pct(x.pe)
        x["利差分位(事后)"] = 1 - x.erp.rank(pct=True)                  # 转成"越低越便宜"
        x["利差分位(当时)"] = 1 - expanding_pct(x.erp)
        for k, n in HZ.items():
            x[k] = c.shift(-n).reindex(x.index) / x.close - 1
        r = dingdi.compute(d).reindex(x.index)
        x["底部信号"] = (r["底部一级"] | r["底部二级"]).fillna(False).astype(bool)
        parts.append(x)
    return pd.concat(parts)


def bucket_table(P: pd.DataFrame, col: str) -> pd.DataFrame:
    b = pd.cut(P[col], [0, .2, .4, .6, .8, 1.0001], right=False, labels=["0–20%（最便宜）", "20–40%", "40–60%", "60–80%", "80–100%（最贵）"])
    rows = {}
    for lab, g in P.groupby(b, observed=True):
        row = {}
        for k in HZ:
            v = g[k].dropna()
            row[f"{k}上涨比例"] = (v > 0).mean() if len(v) else np.nan
            row[f"{k}中位"] = v.median() if len(v) else np.nan
        row["天数"] = len(g)
        rows[lab] = row
    return pd.DataFrame(rows).T


def cheap():
    P = valuation_panel()
    print("\n== 2. 便宜是硬道理（上证50、沪深300、中证500、中证1000、上证指数，2011 年起；价格指数不含分红）==")
    print("各持有期收益与估值分位的秩相关（负数 = 越便宜之后涨得越多）：")
    rows = {}
    for col in ("PE分位(事后)", "PE分位(当时)", "利差分位(事后)", "利差分位(当时)"):
        rows[col] = {k: P[[col, k]].dropna().corr(method="spearman").iloc[0, 1] for k in HZ}
    print(pd.DataFrame(rows).T.round(2).to_string())
    for col in ("利差分位(事后)", "PE分位(当时)"):
        print(f"\n按 {col} 分组：")
        print(bucket_table(P, col).round(3).to_string())
    print("\n底部信号（△B/★B，同一指数 20 天内只算第一次）按估值分组：")
    ev = []
    for name, g in P.groupby("指数"):
        last = -10**9
        for k in np.flatnonzero(g["底部信号"].to_numpy()):
            first, last = k - last > 20, k
            if first:
                ev.append(g.iloc[k])
    E = pd.DataFrame(ev)
    for col in ("利差分位(事后)", "PE分位(当时)"):
        lab = np.where(E[col] <= 0.3, "便宜（分位 ≤30%）", np.where(E[col] >= 0.7, "贵（分位 ≥70%）", "中间"))
        t = E.groupby(lab).agg(次数=("指数", "size"),
                              **{f"{k}上涨比例": (k, lambda v: (v.dropna() > 0).mean()) for k in ("60天", "1年", "3年")},
                              **{f"{k}中位": (k, "median") for k in ("60天", "1年", "3年")})
        print(f"  按 {col}：")
        print(t.round(3).to_string())
    now = P.groupby("指数").tail(1)[["指数", "pe", "erp", "PE分位(事后)", "PE分位(当时)", "利差分位(事后)"]]
    print("\n现在：")
    print(now.round(3).to_string(index=False))


# ---------- 3. 重新重合 ----------
def recouple(bench="sh000906"):
    B = load(bench).close
    rb = np.log(B).diff()
    S = []
    for name, code in industries():
        d = load(code)
        if d is None:
            continue
        c = d.close.reindex(B.index)
        r = np.log(c).diff()
        corr = r.rolling(60).corr(rb)
        x = pd.DataFrame({
            "行业": name,
            "ex120": (c / c.shift(120)) - (B / B.shift(120)),
            "ex20": (c / c.shift(20)) - (B / B.shift(20)),
            "corr": corr, "dcorr": corr - corr.shift(60),
            "未来60天超额": (c.shift(-60) / c) - (B.shift(-60) / B),
            "未来120天超额": (c.shift(-120) / c) - (B.shift(-120) / B),
        })
        x = x[(x.index >= "2012-01-01") & c.notna()]
        S.append(x.iloc[::20])
    S = pd.concat(S).dropna(subset=["ex120", "dcorr", "ex20"])

    def tab(groups):
        rows = {}
        for k, g in groups.items():
            row = {"样本": len(g)}
            for col in ("未来60天超额", "未来120天超额"):
                v = g[col].dropna()
                row[f"{col}为正比例"] = (v > 0).mean()
                row[f"{col}中位"] = v.median()
            rows[k] = row
        return pd.DataFrame(rows).T

    strong = S[S.ex120 >= 0.2]
    print(f"\n== 3. 重新重合（127 个行业 vs 中证800，2012 年起，每 20 天取样）==")
    print(tab({"所有样本": S, "前120天超额 ≥20%（走自己的叙事）": strong}).round(3).to_string())
    print("\n前 120 天超额 ≥20% 的行业，按与宽基的相关系数变化（60 日相关系数比 60 天前）：")
    print(tab({"更加脱钩（下降 ≥0.1）": strong[strong.dcorr <= -0.1], "基本不变": strong[strong.dcorr.between(-0.1, 0.1)],
               "重新重合（上升 ≥0.1）": strong[strong.dcorr >= 0.1]}).round(3).to_string())
    print("\n同上，按最近 20 天超额（相对强弱线是否走平）：")
    print(tab({"还在走高（≥+3%）": strong[strong.ex20 >= 0.03], "走平（±3%）": strong[strong.ex20.between(-0.03, 0.03)],
               "回落（≤−3%）": strong[strong.ex20 <= -0.03]}).round(3).to_string())
    both = strong[(strong.dcorr >= 0.1) & (strong.ex20 < 0.03)]
    keep = strong[(strong.dcorr <= 0) & (strong.ex20 >= 0.03)]
    print("\n两个条件合起来：")
    print(tab({"重新重合且超额不再走高": both, "仍然脱钩且超额还在走高": keep}).round(3).to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    overshoot()
    cheap()
    recouple()
