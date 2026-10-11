"""用同花顺自带的市场涨跌家数（2014 起）测试宽度条件对底部信号的作用。"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "research")
from lib import load, features, forward_stats
from ths_reader import read
from backtest import first_trigger

H = "D:/tonghua/同花顺/history/sznse/day/"
PAIRS = {"sz399001": ("399001", "深证成指 / 深市宽度"), "sz399006": ("399006", "创业板指 / 创业板宽度")}
START = "2015-01-01"

def breadth(code):
    b = read(H + code + ".day").set_index("date")[["3026", "3027"]].astype(float)
    b.columns = ["up", "dn"]
    b = b[(b.up + b.dn) > 0]
    b["dr"] = b.dn / (b.up + b.dn)               # 下跌家数占比
    b["dr5"] = b.dr.rolling(5).mean()
    b["dr20"] = b.dr.rolling(20).mean()
    return b

def show(name, res, base=None):
    r20, r60 = res.ret20.dropna(), res.ret60.dropna()
    print(f"  {name:<34}{len(res):>4}次  20日均{r20.mean():+6.1%} 涨概率{(r20>0).mean():4.0%}  60日均{r60.mean():+6.1%} 涨概率{(r60>0).mean():4.0%}  60日内最大浮亏中位{res.mae60.median():+6.1%}")

allres = {}
for sym, (tc, title) in PAIRS.items():
    df = load(sym); f = features(df, "close"); b = breadth(tc)
    f = f.join(b, how="left")
    os_price = (f.z5 <= -1.5) & (f.z20 <= -1.3) & (f.zbias60 <= -0.85) & (f.z_dd120 <= -12) & (f.rsi14 <= 30)
    os_loose = (f.z_dd120 <= -8) & (f.rsi14 <= 35)
    rules = {
        "基准(每5天)": None,
        "A 价格超卖(旧规则)": os_price,
        "B1 恐慌日 下跌占比>=90%": f.dr >= 0.90,
        "B2 恐慌日>=90% + 价格偏弱": (f.dr >= 0.90) & os_loose,
        "B3 5日平均下跌占比>=70% + 偏弱": (f.dr5 >= 0.70) & os_loose,
        "B4 恐慌日>=85% 且 20日平均>=60%": (f.dr >= 0.85) & (f.dr20 >= 0.60),
    }
    print(f"\n=== {title}  ({START} 起) ===")
    for name, cond in rules.items():
        if cond is None:
            d = df.index[df.index >= START][::5]
        else:
            s = first_trigger(cond, 20); s = s[s.index >= START]; d = s[s].index
        res = forward_stats(df, d, horizons=(20, 60))
        show(name, res)
        allres.setdefault(name, []).append(res.assign(idx=title[:4]))
    for name in ("B2 恐慌日>=90% + 价格偏弱", "B4 恐慌日>=85% 且 20日平均>=60%"):
        s = first_trigger(rules[name], 20); s = s[s.index >= START]
        r = forward_stats(df, s[s].index, horizons=(20, 60))
        print(f"   {name} 日期: " + ", ".join(f"{d.date()}({x:+.0%})" for d, x in zip(r.date, r.ret60.fillna(np.nan))))
