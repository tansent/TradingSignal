"""短线冰点：全市场跌停家数 > 涨停家数，且跌停 ≥ 20 家（一位交易者的说法，2026-10-16 检验）。

2011 年以来（全 A 等权，即全部 A 股平均涨跌）：每年约 15 天；之后 1 天、3 天上涨的比例约 60%，5 天 66%、中位 +1.7%，
20 天 65%、中位 +3.0%（所有交易日分别是 55%、56%、56% / +0.5%、55% / +0.9%）。"跌停 ≥ 涨停 2 倍且 ≥30 家"略强一点
（20 天 67%、中位 +3.8%），已包含在内；"2 倍"不设最低家数时多出来的是涨跌停都很少的平淡日子（每年约 2 天，5 天后上涨只有 40%），不用。
对比"全市场下跌家数 ≥92%"：3 天上涨比例只有 50%（普跌之后常还要再跌一两天）；我们的 ★B 在 5～20 天最强，但一年只有约 3 次。

画在各指数 K 线上的浅蓝 B（chart_ice，8 个宽基，看指数本身之后的涨跌）：全市场冰点，再加四个条件——
  1. 前 20 天没有 ★S/▽S/乖S/红色◆S：顶部之后的冰点没有优势（5 天上涨 50%、20 天 47%；2011-2018、2019-2026 两段都是这样）。
     红色 ◆S 是用户 2026-10-16 提议加的：红色 ◆S 后 20 天内的浅蓝 B 只有 6 次（2026-07-07 四个指数、2021-01-29 科创50 等），
     5 天上涨 33%、20 天 17%、60 天全跌（中位 -12%）；20～40 天的那几次反而不错（2026-07-30、2021-03-09 涨 4～12%），所以窗口用 20 天。
     橙色 ◆S 之后的冰点没有明显变差（20 天上涨 57%），不加；
  2. 前 5 天（含当天）没有 △B/★B：已经有 B，重复提示；
  3. 指数收盘比 20 日最高收盘低 ≥5%：真的跌下来了，不是牛市里的小回调；
  4. 前 10 天没有出现过满足 1-3 的冰点：一段下跌只画第一个。
  每个指数每年约 2.7 次，5 天上涨 64%、20 天 64%、中位 +2.7%（所有交易日 52%、51%）；两段分别是 63%/61% 和 66%/67%。
  只加 1、2 时每年约 8 次（5 天 65%、20 天 58%），图上太密。
  试过但没用的："只在月线 BBI 下方才算"前后两段结论相反；"指数情绪 ≤20"把次数减到 3 次但 5 天上涨比例降到 62%。
  科创50 样本少且结果差，看科创时打折。

也负责写给通达信 DLL 的通用数据文件 data/tdx_extra.csv（第 21～28 号函数按列读取）：
  日期,涨停家数,跌停家数,冰点(1/0),中线温度,短线温度分位,0,0,0
"""
import pandas as pd

ZT_MIN = 20


def icepoint(a: pd.DataFrame) -> pd.Series:
    """a：breadth_daily 里 board=='all' 的全市场宽度（有 zt、dt 两列）。"""
    return (a["dt"] > a["zt"]) & (a["dt"] >= ZT_MIN)


def chart_ice(df: pd.DataFrame, r: pd.DataFrame, ice: pd.Series, red: pd.Series | None = None) -> pd.Series:
    """某个指数 K 线上画浅蓝 B 的日子（和通达信 DINGDI 里的"冰点"一致）。df 指数日线，r = dingdi.compute(df)，ice 全市场冰点，
    red 该指数的红色 ◆S（research/red_top.py，红利类没有）。"""
    top = r["顶部"] | r["顶部一级"] | r["乖离顶"]
    if red is not None:
        top = top | red.reindex(df.index, fill_value=False).astype(bool)
    s = top.rolling(20, min_periods=1).max().astype(bool)
    b = (r["底部一级"] | r["底部二级"]).rolling(5, min_periods=1).max().astype(bool)
    down = df.close <= df.close.rolling(20).max() * 0.95
    cand = ice.reindex(df.index, fill_value=False).astype(bool) & ~s & ~b & down
    return cand & ~cand.shift(1, fill_value=False).rolling(10, min_periods=1).max().astype(bool)


def export_extra_tdx(a: pd.DataFrame, temp: pd.DataFrame, path) -> int:
    df = pd.DataFrame({"zt": a["zt"], "dt": a["dt"], "ice": icepoint(a).astype(int),
                       "mid": temp["中线温度"].reindex(a.index), "short": temp["短线温度_分位"].reindex(a.index)}).fillna(0)
    lines = [f"{r.Index:%Y%m%d},{int(r.zt)},{int(r.dt)},{int(r.ice)},{r.mid:.2f},{r.short:.2f},0,0,0" for r in df.itertuples()]
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)
