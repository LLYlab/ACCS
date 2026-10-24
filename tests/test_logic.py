"""ACCS 核心逻辑测试（无需音频硬件/模型下载）。

运行: `.venv\Scripts\python.exe -X utf8 tests\test_logic.py`
纯 assert，不依赖 pytest。
"""
import json
import os
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accs import exporters as X
from accs.config import Config, VadConfig
from accs.output import OutputSink, SegmentRecord
from accs.segmenter import Segmenter


# ============================================================ 配置
def test_config_override():
    cfg = Config.load(overrides={
        "audio.source": "file",
        "audio.file": "x.wav",
        "audio.frame_ms": 256,
        "transcriber.model": "base",
        "transcriber.engine": "sherpa_onnx",
        "vad.overlap_ms": 200,
        "vad.end_ratio": 0.6,
        "output.mode": "all",
    })
    assert cfg.audio.source == "file"
    assert cfg.audio.file == "x.wav"
    assert cfg.audio.frame_size % 512 == 0, "帧长应为 512 的整数倍(Silero VAD 要求)"
    assert cfg.transcriber.model == "base"
    assert cfg.vad.overlap_ms == 200
    assert cfg.vad.end_ratio == 0.6
    assert cfg.output.mode == "all"
    print("[ok] test_config_override")


def test_srt_ts_format():
    assert X.fmt_srt_ts(0) == "00:00:00,000"
    assert X.fmt_srt_ts(1.5) == "00:00:01,500"
    assert X.fmt_srt_ts(3661.23) == "01:01:01,230"
    assert X.fmt_vtt_ts(1.5) == "00:00:01.500"
    print("[ok] test_srt_ts_format")


# ============================================================ 输出汇（CLI）
def test_output_sink():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config.load(overrides={"output.mode": "all", "output.file": os.path.join(tmp, "demo")})
        sink = OutputSink(cfg.output)
        sink.emit(SegmentRecord(0.0, 2.0, "第一句", "zh", index=0))
        sink.emit(SegmentRecord(3.0, 5.0, "第二句", "zh", index=1))
        sink.close()

        txt = open(os.path.join(tmp, "demo.txt"), encoding="utf-8").read()
        srt = open(os.path.join(tmp, "demo.srt"), encoding="utf-8").read()
        jsonl = open(os.path.join(tmp, "demo.jsonl"), encoding="utf-8").read()
        assert "第一句" in txt and "第二句" in txt
        assert "1\n00:00:00,000 --> 00:00:02,000" in srt
        assert "2\n00:00:03,000 --> 00:00:05,000" in srt
        assert '"text": "第一句"' in jsonl
        # all 模式还应产出 md / vtt / json / csv
        for ext in ("md", "vtt", "json", "csv"):
            assert os.path.exists(os.path.join(tmp, f"demo.{ext}")), f"缺少 {ext}"
        print("[ok] test_output_sink")


# ============================================================ 分段器
def _frames(n, size=1600):
    return [np.zeros(size, dtype=np.float32) for _ in range(n)]


def _push(seg, frames, probs):
    out = []
    for f, p in zip(frames, probs):
        out.extend(seg.push(f, p))
    return out


def test_segmenter_silence_split():
    cfg = VadConfig(threshold=0.5, silence_ms=300, min_speech_ms=200,
                    prepend_ms=100, pad_ms=100, max_segment_ms=10000, overlap_ms=0)
    seg = Segmenter(cfg, 16000)
    frames = _frames(10)                 # 每帧 100ms
    probs = [0.9] * 5 + [0.0] * 5         # 500ms 说话 + 500ms 静音
    chunks = _push(seg, frames, probs)
    assert len(chunks) == 1, f"应切出 1 段，实际 {len(chunks)}"
    c = chunks[0]
    assert 0.4 <= (c.end - c.start) <= 0.75, f"时长异常 {c.end - c.start:.2f}s"
    assert abs(c.start - 0.0) < 0.05
    print(f"[ok] test_segmenter_silence_split ({c.end - c.start:.2f}s)")


def test_segmenter_drops_short():
    cfg = VadConfig(threshold=0.5, silence_ms=300, min_speech_ms=400,
                    prepend_ms=100, pad_ms=100, max_segment_ms=10000, overlap_ms=0)
    seg = Segmenter(cfg, 16000)
    chunks = _push(seg, _frames(6), [0.9] + [0.0] * 5)   # 只有 100ms 说话
    assert chunks == [], "过短片段应被丢弃"
    print("[ok] test_segmenter_drops_short")


def test_segmenter_max_split():
    cfg = VadConfig(threshold=0.5, silence_ms=1000, min_speech_ms=200,
                    prepend_ms=50, pad_ms=50, max_segment_ms=2000, overlap_ms=0)
    seg = Segmenter(cfg, 16000)
    chunks = _push(seg, _frames(30), [0.9] * 30)           # 3s 连续说话
    chunks += seg.flush()
    assert len(chunks) >= 2, f"超长应被切开，实际 {len(chunks)}"
    total = sum(c.audio.size for c in chunks)
    assert abs(total - 30 * 1600) <= 1600 * 3, "切分不该丢太多音频"
    for c in chunks:
        assert (c.end - c.start) <= 2.3, f"单段过长 {c.end - c.start:.2f}s"
    print(f"[ok] test_segmenter_max_split ({len(chunks)} 段, {total/16000:.1f}s)")


def test_segmenter_overlap():
    cfg = VadConfig(threshold=0.5, silence_ms=1000, min_speech_ms=200,
                    prepend_ms=50, pad_ms=50, max_segment_ms=2000, overlap_ms=200)
    seg = Segmenter(cfg, 16000)
    chunks = _push(seg, _frames(30), [0.9] * 30)
    chunks += seg.flush()
    assert len(chunks) >= 2
    # 第二段起点应比第一段终点早（重叠）
    assert chunks[1].start < chunks[0].end, "开启 overlap 后相邻片段应重叠"
    print(f"[ok] test_segmenter_overlap (gap={chunks[1].start - chunks[0].end:+.2f}s)")


def test_segmenter_preroll():
    cfg = VadConfig(threshold=0.5, silence_ms=300, min_speech_ms=200,
                    prepend_ms=400, pad_ms=100, max_segment_ms=10000, overlap_ms=0)
    seg = Segmenter(cfg, 16000)
    # 4 帧静音打底，再说话
    chunks = _push(seg, _frames(10), [0.0] * 4 + [0.9] * 3 + [0.0] * 3)
    assert len(chunks) == 1
    # 起点应提前，包含前置静音
    assert chunks[0].start <= 0.4 + 0.05, f"前置补音未生效 start={chunks[0].start}"
    print(f"[ok] test_segmenter_preroll (start={chunks[0].start:.2f}s)")


# ============================================================ 导出
def _rows():
    return [
        X.ExportRow(0.0, 2.0, "第一句", translation="First sentence", language="zh", index=1),
        X.ExportRow(3.0, 5.0, "第二句", translation="Second sentence", language="zh", index=2),
        X.ExportRow(30.0, 33.0, "新章节开始", language="zh", index=3),
    ]


def test_export_formats():
    rows = _rows()
    meta = X.ExportMeta(title="测试课程", source="file", engine="sherpa_onnx", model="sensevoice")
    srt = X.to_srt(rows, bilingual=True)
    assert "First sentence" in srt and "第二句" in srt
    vtt = X.to_vtt(rows, bilingual=True)
    assert vtt.startswith("WEBVTT")
    csv_text = X.to_csv(rows)
    assert csv_text.splitlines()[0].startswith("index,start,end")
    data = json.loads(X.to_json(rows, meta))
    assert data["meta"]["segments"] == 3
    assert len(data["chapters"]) >= 2, "30s 间隔应分出第二章"
    md = X.to_markdown(rows, meta)
    assert "## " in md and "第一句" in md
    zip_bytes = X.to_zip(rows, meta, include_docx=False)
    with zipfile.ZipFile(__import__("io").BytesIO(zip_bytes)) as z:
        names = set(z.namelist())
    for name in ("transcript.txt", "subtitles.srt", "subtitles_bilingual.srt",
                 "transcript.md", "transcript.json", "transcript.csv", "meta.json"):
        assert name in names, f"zip 缺少 {name}"
    print("[ok] test_export_formats")


def test_auto_chapters():
    rows = _rows()
    ch = X.auto_chapters(rows, gap_seconds=2.5)
    assert len(ch) == 2, f"应分出 2 章，实际 {len(ch)}"
    assert ch[1]["start"] == 30.0
    print("[ok] test_auto_chapters")


if __name__ == "__main__":
    test_config_override()
    test_srt_ts_format()
    test_output_sink()
    test_segmenter_silence_split()
    test_segmenter_drops_short()
    test_segmenter_max_split()
    test_segmenter_overlap()
    test_segmenter_preroll()
    test_export_formats()
    test_auto_chapters()
    print("\n全部测试通过 ✔")
