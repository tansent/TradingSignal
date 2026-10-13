"""生成本地版日报：reports/A股情绪日报.html（双击即可在浏览器打开，不需要 Claude）。

页面代码与 claude.ai 看板相同（report_page/index.html），这里补上它依赖的样式变量和一个
最小的 dash 适配层，并把 report_data/*.json 直接嵌进文件。图表库 d3 从 cdnjs 加载（需要联网）。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "A股情绪日报.html"
DATASETS = ["summary", "indexes", "history", "breadth", "etf", "signals", "daily", "us"]

TOKENS = """
:root {
  color-scheme: light;
  --font-anthropic-sans: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
  --font-anthropic-serif: "Songti SC", "SimSun", Georgia, serif;
  --cds-font-size-caption: 12px; --cds-font-size-body: 14px; --cds-font-size-heading: 16px; --cds-font-size-title: 26px;
  --cds-font-weight-medium: 600;
  --cds-gap-xs: 4px; --cds-gap-sm: 8px; --cds-gap-md: 14px; --cds-gap-lg: 22px; --cds-gap-xl: 32px;
  --cds-pad-xs: 4px; --cds-pad-sm: 8px; --cds-pad-md: 12px; --cds-pad-lg: 16px; --cds-pad-xl: 24px;
  --cds-radius: 8px; --cds-dur-slow: 300ms; --cds-ease-out: cubic-bezier(.2,.8,.2,1);
  --color-fg: #1f1e1b; --color-fg-muted: #6b6a64; --color-bg: #f6f5f1; --color-panel: #ffffff; --color-border-line: #e4e2db;
  --color-ok: #1a7f37; --color-warn: #a86f00; --color-bad: #c9372c;
  --cds-chart-grid: #eceae4; --cds-chart-axis: #c9c7bf; --cds-chart-reference: #9a988f; --cds-chart-muted: #a5a39b;
  --cds-surface-popover: #ffffff; --cds-border: #dedcd4; --cds-text-secondary: #6b6a64;
  --cds-shadow-popover: 0 4px 16px rgba(0,0,0,.12);
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --color-fg: #f2f1ec; --color-fg-muted: #b3b1a8; --color-bg: #1a1a19; --color-panel: #232321; --color-border-line: #3a3936;
    --color-ok: #4cc26e; --color-warn: #e3b341; --color-bad: #f0645a;
    --cds-chart-grid: #2f2e2b; --cds-chart-axis: #55534e; --cds-chart-reference: #8a887f; --cds-chart-muted: #7d7b74;
    --cds-surface-popover: #2b2a28; --cds-border: #45433f; --cds-text-secondary: #b3b1a8; --cds-shadow-popover: 0 4px 16px rgba(0,0,0,.5);
  }
}
body { margin: 0; background: var(--color-bg); color: var(--color-fg); }
#dash-root { padding: 20px max(16px, 3vw) 40px; max-width: 1400px; margin-inline: auto; }
.dash-skeleton:empty { min-height: 1em; }
"""

SHIM = """
window.dash = (() => {
  const DATA = __DATA__;
  const fns = []; let params = {};
  const run = () => fns.forEach((f) => { try { f(); } catch (e) { console.error(e); } });
  let t = null; window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(run, 150); });
  return {
    colors: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
    data: (id) => DATA[id] ? { status: 'ok', data: DATA[id], columns: DATA[id][0] ? Object.keys(DATA[id][0]) : [], meta: [] } : { status: 'missing', data: [], columns: [], meta: [] },
    onData: (fn) => { fns.push(fn); fn(); },
    params: () => params, setParams: (p) => { Object.assign(params, p); run(); }, resetParams: () => { params = {}; run(); },
    setLink: () => {}, refresh: () => {}, calc: () => {}, loader: () => {},
  };
})();
"""


def main():
    data = {}
    for ds in DATASETS:
        f = ROOT / "report_data" / f"{ds}.json"
        if f.exists():
            data[ds] = json.loads(f.read_text(encoding="utf-8"))
    page = (ROOT / "report_page" / "index.html").read_text(encoding="utf-8")
    shim = SHIM.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>A股情绪日报</title><style>{TOKENS}</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>{shim}</script>
</head><body><div id="dash-root">
{page}
</div></body></html>"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print("本地日报已生成：", OUT)
    # 按数据日期存档一份；同一天重复运行只保留最后一次
    date = (data.get("summary") or [{}])[0].get("date")
    if date:
        arc = OUT.parent / "存档" / f"A股情绪日报_{date}.html"
        arc.parent.mkdir(exist_ok=True)
        arc.write_text(html, encoding="utf-8")
        print("已存档：", arc)


if __name__ == "__main__":
    main()
