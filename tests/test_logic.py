"""ACCS 核心逻辑的轻量测试(无需音频硬件/模型下载)。

运行: `.venv\Scripts\python.exe -X utf8 tests\test_logic.py`
本脚本不使用 pytest,纯 assert,便于直接跑。
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accs.config import Config
from accs.output import OutputSink, SegmentRecord, _fmt_srt_ts
from accs.pipeline import Pipeline


class FakeVAD:
    """按预设序列返回说话概率,用于测试端点切分逻辑。"""

    def __init__(self, probs):
        self.probs = list(probs)
        self.i = 0

    def probability(self, frame):
        p = self.probs[min(self.i, len(self.probs) - 1)]
        self.i += 1
        return p


def test_config_override():
    cfg = Config.load(overrides={
        "audio.source": "file",
        "audio.file": "x.wav",
        "audio.frame_ms": 256,
        "transcriber.model": "base",
        "output.mode": "all",
    })
    assert cfg.audio.source == "file"
    assert cfg.audio.file == "x.wav"
    assert cfg.audio.frame_size % 512 == 0, "帧长应为 512 的整数倍(Silero VAD 要求)"
    assert cfg.transcriber.model == "base"
    assert cfg.output.mode == "all"
    print("[ok] test_config_override")


def test_srt_ts_format():
    assert _fmt_srt_ts(0) == "00:00:00,000"
    assert _fmt_srt_ts(1.5) == "00:00:01,500"
    assert _fmt_srt_ts(3661.23) == "01:01:01,230"
    print("[ok] test_srt_ts_format")


def test_output_sink():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config.load(overrides={
            "output.mode": "all", "output.file": os.path.join(tmp, "demo"),
        })
        sink = OutputSink(cfg.output)
        sink.emit(SegmentRecord(0.0, 2.0, "第一句", "zh"))
        sink.emit(SegmentRecord(3.0, 5.0, "第二句", "zh"))
        sink.close()

        txt = open(os.path.join(tmp, "demo.txt"), encoding="utf-8").read()
        srt = open(os.path.join(tmp, "demo.srt"), encoding="utf-8").read()
        jsonl = open(os.path.join(tmp, "demo.jsonl"), encoding="utf-8").read()
        assert "第一句" in txt and "第二句" in txt
        # SRT 编号应从 1 顺序递增,时间格式正确
        assert "1\n00:00:00,000 --> 00:00:02,000" in srt
        assert "2\n00:00:03,000 --> 00:00:05,000" in srt
        assert '"text": "第一句"' in jsonl
        print("[ok] test_output_sink")


def test_endpointing():
    cfg = Config.load(overrides={"audio.source": "file", "audio.file": "dummy.wav"})
    p = Pipeline(cfg)
    # 用可控概率替换 VAD: 2帧说话 -> 5帧静音 -> 2帧说话 -> 5帧静音
    probs = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    p.vad = FakeVAD(probs)
    frame = np.zeros(p._frame_size, dtype=np.float32)

    for _ in range(len(probs)):
        p._process_frame(frame)
    p._finalize(force=True)

    # 收集所有入队的片段
    utterances = []
    while not p._work.empty():
        utterances.append(p._work.get_nowait())

    assert len(utterances) >= 2, f"应切出>=2段,实际 {len(utterances)}"
    # 每段都应有足够的音频长度(>= min_speech_ms)
    min_samples = int(16000 * cfg.vad.min_speech_ms / 1000)
    for u in utterances:
        assert u.audio.size >= min_samples, "片段过短,不应被切出"
    # 时间轴单调
    for a, b in zip(utterances, utterances[1:]):
        assert b.start >= a.end, "片段时间不应重叠/回退"
    print(f"[ok] test_endpointing ({len(utterances)} 段)")


if __name__ == "__main__":
    test_config_override()
    test_srt_ts_format()
    test_output_sink()
    test_endpointing()
    print("\n全部测试通过 ✔")
