"""批量把视频转成带时间戳的中文文字稿（本地显卡运行，内容不出本机）。

模型：阿里达摩院 Paraformer（FunASR，中文专用，自带标点和句级时间戳），从魔搭社区下载到 D:/asr/models。
环境：D:/asr/venv（Python 3.12 + PyTorch cu121 + funasr + av），和项目主环境分开。

用法：
  D:/asr/venv/Scripts/python.exe tools/transcribe_videos.py <视频文件夹> [--out D:/TradingSignal/老王/文字稿]
每个视频生成同名的 .srt（字幕格式）和 .txt（每句前面带 [时:分:秒]），写在 --out 文件夹里；已经转过的跳过。
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MODELSCOPE_CACHE", "D:/asr/models")

import av  # noqa: E402
import numpy as np  # noqa: E402

EXTS = {".mp4", ".mov", ".mkv", ".flv", ".avi", ".webm", ".m4a", ".mp3", ".wav", ".aac"}
# 热词：提高金融术语的识别率
HOTWORDS = ("上证指数 深证成指 创业板 科创50 沪深300 中证500 中证1000 上证50 北证 红利低波 宽基 ETF 国家队 汇金 "
            "均线 年线 半年线 五日线 二十日线 六十日线 月线 周线 日线 K线 量能 缩量 放量 换手率 成交额 融资余额 两融 "
            "拥挤度 抱团 主线 题材 板块 龙头 半导体 算力 人工智能 机器人 券商 银行 北向资金 主力资金 MACD KDJ RSI 布林 "
            "背离 金叉 死叉 缺口 支撑 压力 箱体 反抽 回踩 筑底 见顶 牛市 熊市 震荡市 美联储 降息 加息 纳斯达克 标普")


def load_audio(path: Path) -> np.ndarray:
    """用 PyAV 解出 16kHz 单声道 float32 音频。"""
    with av.open(str(path)) as c:
        stream = next(s for s in c.streams if s.type == "audio")
        rs = av.AudioResampler(format="s16", layout="mono", rate=16000)
        chunks = []
        for frame in c.decode(stream):
            for f in rs.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in rs.resample(None):
            chunks.append(f.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def ts(ms: float, srt: bool) -> str:
    s, ms = divmod(int(ms), 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}" if srt else f"{h:02d}:{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="D:/TradingSignal/老王/文字稿")
    ap.add_argument("--limit", type=int, default=0, help="只转前 N 个（试转用）")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS)
    todo = [p for p in files if not (out / (p.stem + ".srt")).exists()]
    if a.limit:
        todo = todo[:a.limit]
    print(f"共 {len(files)} 个音视频，待转 {len(todo)} 个", flush=True)
    if not todo:
        return
    from funasr import AutoModel
    model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc",
                      device="cuda:0", disable_update=True, disable_pbar=True)
    for i, p in enumerate(todo, 1):
        t0 = time.time()
        try:
            wav = load_audio(p)
            res = model.generate(input=wav, batch_size_s=300, hotword=HOTWORDS, sentence_timestamp=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(todo)}] 失败 {p.name}: {e!r}"[:300], flush=True)
            continue
        sents = res[0].get("sentence_info") or [{"text": res[0]["text"], "start": 0, "end": len(wav) / 16}]
        with open(out / (p.stem + ".srt"), "w", encoding="utf-8") as f:
            for k, s in enumerate(sents, 1):
                f.write(f"{k}\n{ts(s['start'], True)} --> {ts(s['end'], True)}\n{s['text']}\n\n")
        with open(out / (p.stem + ".txt"), "w", encoding="utf-8") as f:
            f.write(f"# {p.name}  时长 {ts(len(wav) / 16, False)}\n")
            for s in sents:
                f.write(f"[{ts(s['start'], False)}] {s['text']}\n")
        print(f"[{i}/{len(todo)}] {p.name}：{len(wav) / 16000 / 60:.0f} 分钟音频，{len(sents)} 句，用时 {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    sys.exit(main())
