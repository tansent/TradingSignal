"""A股拥挤度：每天成交额排名前 5% 的个股，成交额占全部 A 股总成交额的比例（公众号"杨哥"用的顶部指标）。

他的用法（2026-05-27 文章）：>45% 高危过热；>48% 历史少见、基本是阶段顶部区间，之后 1～4 周内常有中级回调；
跌破 42% 可以小幅接回，跌破 35% 才是安全加仓区。他在 2026-05-26（48.55%）、06-03（51.65%）据此提前空仓。
数据：通达信本地个股日线（沪深京 A 股，不含 ETF、指数），结果存 data/crowding.parquet。
检验：拥挤度首次站上 45% / 48% 之后，指数 20/60/120 天的表现；以及之后跌破 42% / 35% 时再买的表现。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_breadth import V, board_of  # noqa: E402
from tdx_reader import read_day  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "crowding.parquet"


def build() -> pd.DataFrame:
    parts = []
    for m in ("sh", "sz", "bj"):
        for f in sorted((V / m / "lday").glob("*.day")):
            if board_of(f.stem) is None:
                continue
            d = read_day(f)
            a = d["amount"]
            a = a[a > 0]
            if len(a):
                parts.append(a)
    s = pd.concat(parts)
    s = s[s.index >= "2005-01-01"]

    def top5(x: pd.Series) -> float:
        v = np.sort(x.to_numpy())[::-1]
        k = max(1, int(np.ceil(len(v) * 0.05)))
        return v[:k].sum() / v.sum()

    g = s.groupby(level=0)
    df = pd.DataFrame({"crowd": g.apply(top5) * 100, "n": g.size()})
    df.to_parquet(OUT)
    return df


def load() -> pd.DataFrame:
    return pd.read_parquet(OUT) if OUT.exists() else build()


def top_zone(crowd: pd.Series, hs: pd.DataFrame, gap: int = 60) -> pd.DataFrame:
    """抱团顶：拥挤度 ≥48% 且沪深300 在一年高点 5% 以内为 zone=2（高危），≥45% 且近高点为 zone=1（警戒）。
    mark=1：每一轮 zone=2 的第一天，即 ◆S（上一次 zone=2 之后隔了 gap 个交易日以上才算新的一轮）。"""
    near = (hs.close >= 0.95 * hs.high.rolling(250).max()).reindex(crowd.index).fillna(False).astype(bool)
    zone = pd.Series(np.where((crowd >= 48) & near, 2, np.where((crowd >= 45) & near, 1, 0)), index=crowd.index)
    mark = pd.Series(0, index=crowd.index)
    last = -10**9
    for i in np.flatnonzero(zone.to_numpy() == 2):
        if i - last > gap:
            mark.iloc[i] = 1
        last = i
    return pd.DataFrame({"crowd": crowd, "zone": zone, "mark": mark})


def export_tdx(crowd: pd.Series, hs: pd.DataFrame, path) -> int:
    """给通达信 DLL（tdx_plugin/ntdata.dll 的 4/5/6 号函数）读的文件：日期,拥挤度%,区域(2 高危 / 1 警戒 / 0),◆S(1/0)。"""
    t = top_zone(crowd, hs)
    lines = [f"{d:%Y%m%d},{r.crowd:.2f},{int(r.zone)},{int(r.mark)}" for d, r in t.iterrows()]
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def export_margin_tdx(mg20_known: pd.Series, mg: pd.Series, path) -> int:
    """给通达信 DLL（第 7/8 号函数）读的文件：日期,融资余额20日变化%（收盘时已知，即前一天的数）,融资余额亿,0。"""
    df = pd.DataFrame({"chg": mg20_known * 100, "bal": mg.shift(1)}).dropna()
    lines = [f"{d:%Y%m%d},{r.chg:.3f},{r.bal:.1f},0" for d, r in df.iterrows()]
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def episodes(c: pd.Series, level: float, gap: int = 60) -> list[pd.Timestamp]:
    """拥挤度首次站上 level 的日子（上一次站上之后，至少 gap 个交易日低于 level 才算新的一次）。"""
    above = (c >= level).to_numpy()
    out, last = [], -10**9
    for i in np.flatnonzero(above):
        if i - last > gap:
            out.append(c.index[i])
        last = i
    return out


def forward(idx: pd.DataFrame, dates, label: str) -> pd.DataFrame:
    c, lo = idx.close, idx.low
    rows = []
    for t in dates:
        if t not in c.index:
            continue
        k = c.index.get_loc(t)
        f = lambda n: c.iloc[k + n] / c.iloc[k] - 1 if k + n < len(c) else np.nan  # noqa: E731
        rows.append({"条件": label, "日期": t.date(), "20天": f(20), "60天": f(60), "120天": f(120),
                     "60天内最大回撤": lo.iloc[k + 1:k + 61].min() / c.iloc[k] - 1 if k + 1 < len(c) else np.nan})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    rebuild = "--rebuild" in sys.argv
    cr = build() if rebuild or not OUT.exists() else load()
    c = cr["crowd"]
    print("核对他公布的数：2026-05-26", round(c.get(pd.Timestamp("2026-05-26"), np.nan), 2), "（他 48.55）；2026-06-03",
          round(c.get(pd.Timestamp("2026-06-03"), np.nan), 2), "（他 51.65）")
    print("最近 10 天：", c.tail(10).round(1).to_dict())
    print("\n历史分布（2011 起）：", c[c.index >= "2011"].describe(percentiles=[.5, .9, .95, .99]).round(1).to_dict())
    yr = c.groupby(c.index.year).agg(["mean", "max"]).round(1)
    print(yr.T.to_string())
    for lv in (45, 48):
        print(f"\n拥挤度首次站上 {lv}% 的日子：", [d.date().isoformat() for d in episodes(c, lv)])
    for name, code in (("上证指数", "sh000001"), ("沪深300", "sh000300"), ("创业板指", "sz399006")):
        idx = read_day(V / code[:2] / "lday" / f"{code}.day")
        rows = []
        for lv in (45, 48):
            rows.append(forward(idx, episodes(c[c.index >= "2011"], lv), f"首次站上{lv}%"))
        allf = pd.concat(rows)
        print(f"\n== {name} ==")
        print(allf.round(3).to_string(index=False))
