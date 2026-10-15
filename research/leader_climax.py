"""龙头天量：成交额最大的 10 只个股里，有股票"成交额创一年新高、股价在一年高点 10% 以内、当天收跌"（公众号"杨哥"的
"总龙头高位放天量、筹码松动"，如 2026-06-05 中际旭创）。看这种日子是不是集中在阶段顶部，以及和拥挤度一起用有没有帮助。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_breadth import V, board_of  # noqa: E402
from crowding import load as load_crowd  # noqa: E402
from tdx_reader import read_day  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "leader_climax.parquet"
TOPN = 10


def build() -> pd.DataFrame:
    parts = []
    for m in ("sh", "sz", "bj"):
        for f in sorted((V / m / "lday").glob("*.day")):
            if board_of(f.stem) is None:
                continue
            d = read_day(f)
            if len(d) < 260:
                continue
            x = pd.DataFrame({
                "code": f.stem, "amount": d.amount,
                "amt_max": d.amount >= d.amount.rolling(250, min_periods=120).max(),
                "near_high": d.close >= 0.9 * d.high.rolling(250, min_periods=120).max(),
                "down": d.close < d.close.shift(),
            })
            parts.append(x[x.index >= "2010-01-01"])
    a = pd.concat(parts)
    a["climax"] = a.amt_max & a.near_high & a.down
    a = a.reset_index().rename(columns={"index": "date"})
    top = a.sort_values(["date", "amount"], ascending=[True, False]).groupby("date").head(TOPN)
    out = top.groupby("date").agg(n_climax=("climax", "sum"),
                                  codes=("code", lambda s: ",".join(s[top.loc[s.index, "climax"]])))
    out.to_parquet(OUT)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    lc = build() if "--rebuild" in sys.argv or not OUT.exists() else pd.read_parquet(OUT)
    cr = load_crowd()["crowd"]
    hs = read_day(V / "sh" / "lday" / "sh000300.day")
    cy = read_day(V / "sz" / "lday" / "sz399006.day")
    near = hs.close >= 0.95 * hs.high.rolling(250).max()
    df = pd.DataFrame({"n": lc.n_climax, "codes": lc.codes, "crowd": cr, "near": near}).dropna(subset=["n"])
    df = df[df.index >= "2011-01-01"]

    def fwd(s, t, n):
        k = s.index.get_loc(t)
        return s.iloc[k + n] / s.iloc[k] - 1 if k + n < len(s) else np.nan

    def report(mask, label, gap=40):
        ev, last = [], None
        for t in df.index[mask]:
            if last is None or (df.index.get_loc(t) - df.index.get_loc(last)) > gap:
                ev.append(t)
            last = t
        rows = [{"日期": t.date(), "拥挤度": round(df.crowd.get(t, np.nan), 1), "龙头": df.codes[t],
                 "沪深300 60天": fwd(hs.close, t, 60), "创业板 60天": fwd(cy.close, t, 60),
                 "沪深300 60天内最大回撤": hs.low.iloc[hs.index.get_loc(t) + 1:hs.index.get_loc(t) + 61].min() / hs.close[t] - 1}
                for t in ev if t in hs.index]
        R = pd.DataFrame(rows)
        print(f"\n== {label}：{len(R)} 次 ==")
        if len(R):
            print(R.round(3).to_string(index=False))
            print("60 天后沪深300 下跌的比例", round((R["沪深300 60天"].dropna() < 0).mean(), 2),
                  "| 中位", round(R["沪深300 60天"].median(), 3), "| 最大回撤中位", round(R["沪深300 60天内最大回撤"].median(), 3))

    base = (hs.close.shift(-60) / hs.close - 1)[hs.index >= "2011"].dropna()
    print("基准：沪深300 任意一天之后 60 天下跌的比例", round((base < 0).mean(), 2), "中位", round(base.median(), 3))
    report((df.n >= 1) & df.near, "龙头天量收跌 且 沪深300 在一年高点 5% 以内")
    report((df.n >= 1) & df.near & (df.crowd >= 45), "龙头天量收跌 且 沪深300 近高点 且 拥挤度 ≥45%")
    report((df.crowd >= 48) & df.near, "拥挤度 ≥48% 且 沪深300 近高点（对照）")
