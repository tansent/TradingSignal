"""事件预警：上一份日报之后已公布的重要数据，以及未来 3 个交易日内可以预见的重要事件（华尔街见闻财经日历）。

Python 负责取日历、筛选重要事件、标出公布值相对预期的高低，并按固定规则（RULES）给常见事件写一句"基础判断"，
写出 report_data/events.json。每日定时任务（Claude）运行时会把判断和总评改写得更完整（judgment_src = "claude"），
见 docs/定时任务说明.md。手动运行（更新情绪日报.bat）时用规则判断，同一事件当天已有的 Claude 判断会保留。

用法：
  python daily/events.py                 重建 report_data/events.json
  python daily/events.py apply 判断.json  把判断合并进 events.json（定时任务用）
判断文件格式：{"overall": "一句话总评", "items": {"<事件 id>": "一句话判断", ...}}
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

os.environ["NO_PROXY"] = "*"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report_data" / "events.json"
AHEAD = 3                      # 未来几个 A 股交易日
MAIN = {"中国", "美国"}
# 重要性 3 的事件只保留中美宏观数据和政策类，其余（新品发布、展会等）要重要性 ≥4 才保留
MACRO = re.compile("美联储|FOMC|议息|利率|央行|LPR|MLF|逆回购|降准|降息|国新办|政治局|国常会|国务院|证监会|金融监管|财政部|发改委|"
                   "关税|贸易|GDP|CPI|PPI|PMI|非农|失业|就业|零售|社融|社会融资|M2|贷款|进口|出口|工业增加值|固定资产|房地产|"
                   "金融街论坛|人大|全会|国债|讲话")
WEEK = "一二三四五六日"


# 基础判断规则：(地区（None 为不限）, 事件关键词正则, 高于时, 低于时, 持平时, 未公布时, 已发生但没有数值时)
# "高于/低于"指公布值相对预期（没有预期时相对前值）。按顺序匹配，先写更具体的规则。
RULES = [
    ("美国", "褐皮书", None, None, None, "美联储对各地经济的定性描述，通常影响很小", "美联储对各地经济的定性描述，通常影响很小"),
    ("美国", "利率决议|联邦基金利率", "利率高于预期（鹰派意外），偏利空", "利率低于预期（降息超预期），偏利多",
     "利率符合预期，关注声明措辞和后续指引", "加息或降息不及预期偏利空，降息超预期偏利多；声明措辞同样重要",
     "决议已公布，具体结果需看新闻"),
    ("美国", "CPI|PPI|PCE|通胀", "通胀偏热，降息预期降温、美债收益率易上行，偏利空美股和成长股；A 股通常一天左右反应完",
     "通胀降温，降息预期升温，偏利多美股和成长股", "影响有限", "高于预期偏利空（降息预期降温），低于预期偏利多", None),
    ("美国", "失业率|初请|申请失业救济", "就业转弱，利率压力减轻，偏利多利率敏感资产，但大幅恶化会引发衰退担忧",
     "就业偏强，降息预期降温，通胀偏高时偏利空", "影响有限", "明显上升（就业转弱）偏利好降息预期，明显下降偏利空；周度数据通常影响不大", None),
    ("美国", "非农|就业人数|ADP", "就业强于预期，降息预期降温，通胀偏高时偏利空", "就业弱于预期，利率压力减轻，但增加经济放缓担忧，偏中性",
     "影响有限", "大幅强于预期偏利空（降息预期降温），明显弱于预期利率压力减轻但引发放缓担忧", None),
    ("美国", "零售销售", "消费强于预期，通胀偏高时加重利率担忧，偏利空", "消费弱于预期，利率压力减轻但反映经济放缓，偏中性",
     "影响有限", "通胀偏高时，明显强于预期偏利空，明显弱于预期利率压力减轻但反映放缓", None),
    ("美国", "GDP|PMI|ISM", "景气好于预期，经济韧性强，但通胀偏高时可能推高利率，中性偏多", "景气弱于预期，放缓担忧上升，偏利空",
     "影响有限", "明显好于预期中性偏多，明显弱于预期偏利空", None),
    ("美国", "美联储|FOMC|讲话|证词", None, None, None, "措辞偏鹰（推迟降息甚至加息）偏利空，偏鸽偏利多",
     "讲话已结束，鹰鸽要看新闻（规则无法判断）"),
    ("中国", "LPR|贷款市场报价利率|MLF|逆回购|存款准备金|降准|降息", "利率上调或高于预期，偏利空", "利率下调，宽松信号，偏利多（地产和权益估值）",
     "维持不变，影响很小", "若下调属宽松信号，偏利多；维持不变是常态，影响很小", "已公布，具体结果需看新闻"),
    ("中国", "CPI|PPI", "物价回升，反映需求改善，偏利多", "物价走低，需求偏弱，偏利空，但增加宽松预期", "影响有限",
     "回升偏利多（需求改善），走低偏利空但增加宽松预期", None),
    ("中国", "房地产|商品房|房价", "地产数据改善，偏利多地产链", "地产继续走弱，偏利空地产链", "影响有限",
     "降幅收窄偏利多地产链，继续走弱偏利空", None),
    ("中国", "GDP|工业增加值|社会消费品|社零|固定资产投资|PMI|出口|进口|社会融资|社融|新增人民币贷款|M2|M1",
     "经济数据改善，偏利多，尤其顺周期板块", "经济数据走弱，偏利空，但会强化稳增长政策预期", "影响有限",
     "好于预期偏利多，弱于预期偏利空但会强化稳增长政策预期", None),
    ("中国", "国新办|政治局|国常会|国务院|证监会|金融监管|金融街论坛|陆家嘴论坛|人大|全会|发布会", None, None, None,
     "政策窗口：出台超预期的稳增长或资本市场政策偏利多，没有新内容则影响有限", "政策事件已举行，具体内容需看新闻（规则无法判断）"),
    (None, "关税|贸易|制裁|港口费|出口管制", None, None, None, "贸易摩擦相关：升级偏利空出口链和风险偏好，缓和偏利多",
     "贸易摩擦相关：升级偏利空出口链和风险偏好，缓和偏利多（具体内容需看新闻）"),
    (None, "英伟达|台积电|半导体|芯片|GTC|SEMICON", None, None, None, "半导体产业事件，结果好偏利多半导体链，对宽基影响小",
     "半导体产业事件，影响半导体和算力板块情绪，对宽基影响小（具体结果需看新闻）"),
    ("美国", "财政部|国债|再融资", None, None, None, "美债发行安排相关，影响美债收益率，通常影响有限", "美债发行安排相关，通常影响有限"),
    (None, "APEC|G20|G7|峰会|财长会|部长会议|论坛", None, None, None, "多边会议或论坛，通常影响很小", "多边会议或论坛，通常影响很小"),
    (None, "发布会|新品|展示|预售|开售|发售|开源|展会|博览会|大会|开幕|IPO|上市|特斯拉|苹果|iPhone|双11",
     None, None, None, "主题性事件，对指数影响很小", "主题性事件，对指数影响很小"),
]
# 基础总评里优先列出的关键事件
TOP = re.compile("CPI|非农|GDP|利率决议|FOMC|LPR|政治局|主席|金融街论坛|降准|降息|国常会")


def rule_judgment(region: str, event: str, kind: str, surprise_text) -> str:
    for reg, pat, hi, lo, flat, upcoming, done in RULES:
        if (reg is None or reg == region) and re.search(pat, event):
            if kind == "即将发生":
                return upcoming or ""
            if surprise_text and hi:
                body = hi if surprise_text.startswith("高于") else lo if surprise_text.startswith("低于") else flat
                return f"{surprise_text}，{body}"
            return done or (f"已公布，数值暂缺；{upcoming}" if upcoming else "")
    return ""


def rule_overall(rows: list) -> str:
    def short(x):
        return x if len(x) <= 18 else x[:17] + "…"
    done = [r for r in rows if r["kind"] == "已公布" and r["surprise"] and "预期" in r["surprise"]]
    nxt = sorted([r for r in rows if r["kind"] == "即将发生"],
                 key=lambda r: (-(r["importance"] + 2 * bool(TOP.search(r["event"]))), r["time"]))[:3]
    parts = []
    if done:
        parts.append("已公布：" + "，".join(f"{short(r['event'])} {r['actual']:g}（预期 {r['forecast']:g}，{r['surprise']}）" for r in done[:3]))
    if nxt:
        parts.append(f"未来 {AHEAD} 个交易日重点：" + "，".join(f"{r['time'][5:10]} {short(r['event'])}" for r in sorted(nxt, key=lambda r: r["time"])))
    return "；".join(parts)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fetch(day: pd.Timestamp) -> pd.DataFrame:
    import akshare as ak
    for i in range(3):
        try:
            return ak.macro_info_ws(date=day.strftime("%Y%m%d"))
        except Exception:  # noqa: BLE001
            if i == 2:
                raise
            time.sleep(2)


def num(v):
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def windows(now: pd.Timestamp):
    """已公布：上一个交易日 15:00 之后到现在；即将发生：现在到第 AHEAD 个后续交易日的 15:00。"""
    sys.path.insert(0, str(ROOT / "daily"))
    from sina_fill import trading_calendar
    cal = trading_calendar()
    today = now.normalize()
    closed = cal[(cal < today) | ((cal == today) & (now.hour * 60 + now.minute >= 15 * 60))]
    last = closed.max()
    prev = cal[cal < last].max()
    nxt = cal[cal > last][:AHEAD]
    if len(nxt) < AHEAD:                           # 新浪交易日历只到年底时，按工作日估
        nxt = pd.bdate_range(last + pd.Timedelta(days=1), periods=AHEAD)
    return prev + pd.Timedelta(hours=15), now, nxt[-1] + pd.Timedelta(hours=15), list(nxt)


def surprise(actual, forecast, previous):
    if actual is None:
        return None
    if forecast is not None:
        return "高于预期" if actual > forecast else "低于预期" if actual < forecast else "符合预期"
    if previous is not None:
        return "高于前值" if actual > previous else "低于前值" if actual < previous else "与前值持平"
    return None


def time_text(t: pd.Timestamp) -> str:
    s = f"{t:%m-%d} 周{WEEK[t.weekday()]}"
    # 日历里 00:00 和 12:02 是"当天、时间不定"的占位
    return s if (t.hour, t.minute) in ((0, 0), (12, 2)) else f"{s} {t:%H:%M}"


def build(now: pd.Timestamp | None = None) -> list[dict]:
    now = now or pd.Timestamp.now()
    start, mid, end, nxt = windows(now)
    days = pd.date_range(start.normalize(), end.normalize())
    cal = pd.concat([fetch(d) for d in days], ignore_index=True)
    cal["t"] = pd.to_datetime(cal["时间"])
    cal = cal[(cal.t > start) & (cal.t <= end)].drop_duplicates(subset=["时间", "事件"])
    imp = cal["imp"] = pd.to_numeric(cal["重要性"], errors="coerce").fillna(0)
    has_num = cal[["今值", "预期", "前值"]].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
    keep = (imp >= 4) | ((imp >= 3) & cal["地区"].isin(MAIN) & (has_num | cal["事件"].str.contains(MACRO)))
    cal = cal[keep].sort_values("t")

    old = {}
    if OUT.exists():
        try:
            old = {r["id"]: r for r in json.loads(OUT.read_text(encoding="utf-8"))}
        except Exception:  # noqa: BLE001
            old = {}
    rows = []
    for _, e in cal.iterrows():
        t = e.t
        a, f, p = num(e["今值"]), num(e["预期"]), num(e["前值"])
        kind = "已公布" if t <= mid else "即将发生"
        eid = hashlib.md5(f"{e['时间']}|{e['地区']}|{e['事件']}".encode()).hexdigest()[:8]
        prev_row = old.get(eid, {})
        sup = surprise(a, f, p) if kind == "已公布" else None
        if prev_row.get("kind") == kind and prev_row.get("judgment") and prev_row.get("judgment_src", "claude") == "claude":
            judg, src = prev_row["judgment"], "claude"
        else:
            judg = rule_judgment(e["地区"], e["事件"], kind, sup)
            src = "rule" if judg else ""
        rows.append({"id": eid, "kind": kind, "time": t.strftime("%Y-%m-%dT%H:%M"), "time_text": time_text(t),
                     "region": e["地区"], "event": e["事件"], "importance": int(e["imp"]), "actual": a, "forecast": f,
                     "previous": p, "surprise": sup, "judgment": judg, "judgment_src": src,
                     "link": e["链接"] if isinstance(e["链接"], str) else ""})
    o = old.get("overall", {})
    same_day = str(o.get("built", ""))[:10] == now.strftime("%Y-%m-%d")
    if same_day and o.get("judgment") and o.get("judgment_src", "claude") == "claude":
        ov, ov_src = o["judgment"], "claude"
    else:
        ov = rule_overall(rows)
        ov_src = "rule" if ov else ""
    rows.append({"id": "overall", "kind": "总评", "built": now.strftime("%Y-%m-%dT%H:%M"),
                 "window": f"{start:%m-%d %H:%M} 至 {end:%m-%d %H:%M}",
                 "next_days": "、".join(f"{d:%m-%d}" for d in nxt), "judgment": ov, "judgment_src": ov_src})
    return rows


def write(rows: list[dict]):
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    n = sum(r["kind"] == "已公布" for r in rows), sum(r["kind"] == "即将发生" for r in rows)
    log(f"写出 events.json：已公布 {n[0]} 条，未来 {AHEAD} 个交易日 {n[1]} 条")


def apply(path: str):
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = json.loads(OUT.read_text(encoding="utf-8"))
    items = j.get("items", {})
    hit = 0
    for r in rows:
        if r["id"] == "overall" and j.get("overall"):
            r["judgment"], r["judgment_src"] = j["overall"].strip(), "claude"
        elif r["id"] in items:
            r["judgment"], r["judgment_src"] = str(items[r["id"]]).strip(), "claude"
            hit += 1
    missing = set(items) - {r["id"] for r in rows}
    OUT.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {hit} 条判断" + ("，总评已写入" if j.get("overall") else "") + (f"；找不到这些 id：{sorted(missing)}" if missing else ""))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "apply":
        apply(sys.argv[2])
    else:
        write(build())
