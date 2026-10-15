"""生成看板数据库要写入的文档（JSON 文件），供 ArtifactData batch 的 file_path 使用。

用法：python daily/make_docs.py assets.json
assets.json：{"summary": {"url": "/_blob/...", "name": "summary.json"}, ...}（upload_asset 的返回）
输出到 report_docs/：dash_meta.json、files_index.json、datasets_<id>.json
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "report_docs"

DATASETS = {
    "summary": ("今日核心指标", "当天的情绪温度、过热分数、全市场涨跌与涨跌停、融资余额和宽基 ETF 净申购，各一个数。由本地 Python 从通达信日线、两融统计、上交所 ETF 份额计算。"),
    "indexes": ("各指数信号", "八个指数（按市值：上证50、沪深300、中证500、中证1000；按板块：上证、深证、创业板、科创50）的收盘、涨跌、复刻情绪、当天的顶底信号、成分股涨跌家数和最近一次各类信号日期。信号逻辑与通达信公式 DINGDI 一致。"),
    "history": ("近一年走势", "最近 250 个交易日的五个指数收盘、中线温度、短线温度（过去两年分位）、过热分数、八个指数的情绪，以及上证和沪深300 的信号标记。"),
    "breadth": ("全市场宽度", "最近 60 个交易日全部 A 股的上涨/下跌家数、涨停/跌停/炸板、最高连板、成交额、60 日新高/新低家数，由通达信个股日线计算。"),
    "etf": ("宽基 ETF 资金", "最近 120 个交易日 10 只宽基 ETF 成交额相对 20 日均值的倍数，以及上交所 8 只宽基 ETF 的净申购金额（份额变化 × 收盘价）。"),
    "signals": ("信号记录", "2024 年以来八个指数出现过的全部 ★S / ★B / △B 信号。"),
    "us": ("美股", "标普500、纳指的复刻情绪、波动率（VIX/VXN）、美股温度、顶底信号，以及对应的 A 股 QDII ETF（513500、513100）价格。数据来自新浪美股指数和 FRED。"),
    "events": ("事件预警", "上一个交易日 15:00 以来已公布的重要经济数据（公布值对比预期），以及未来 3 个交易日内可以预见的重要事件，来自华尔街见闻财经日历；每条的一句话判断和总评由每日定时任务（Claude）生成，仅作提示。"),
    "daily": ("近 20 个交易日", "最近 20 个交易日逐日的上证指数、中线/短线温度、过热分数、成交额、涨跌家数、涨跌停、融资余额及增减、宽基 ETF 净申购和当天出现的信号。"),
}


def main(assets_path: str):
    assets = json.loads(Path(assets_path).read_text(encoding="utf-8"))
    DOCS.mkdir(exist_ok=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%S+08:00")
    (DOCS / "dash_meta.json").write_text(json.dumps({"title": "A股情绪日报"}, ensure_ascii=False), encoding="utf-8")
    html = (ROOT / "report_page" / "index.html").read_text(encoding="utf-8")
    (DOCS / "files_index.json").write_text(json.dumps({"text": html}, ensure_ascii=False), encoding="utf-8")
    for ds, (title, desc) in DATASETS.items():
        a = assets[ds]
        doc = {"title": title, "description": desc,
               "source": {"kind": "file", "url": a["url"], "name": a["name"], "format": "json"},
               "updated": {"at": now, "by": "每日自动更新"},
               "schedule": "每个交易日收盘后（约 16:30）由本机定时任务重算并上传", "staleAfterHours": 96}
        (DOCS / f"datasets_{ds}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print("docs written:", sorted(p.name for p in DOCS.glob("*.json")))


if __name__ == "__main__":
    main(sys.argv[1])
