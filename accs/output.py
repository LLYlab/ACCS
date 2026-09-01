from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional

from .config import OutputConfig


@dataclass
class SegmentRecord:
    """一条已定稿的转写结果,时间轴为相对采集开始的秒数。"""
    start: float
    end: float
    text: str
    language: str = ""


def _fmt_srt_ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class OutputSink:
    """结果输出: 实时终端显示 + 可选 txt / srt / jsonl 落盘。"""

    def __init__(self, cfg: OutputConfig):
        self.cfg = cfg
        self.segments: List[SegmentRecord] = []
        self._files = {}
        self._init_files()

    def _default_path(self, ext: str) -> str:
        if self.cfg.file:
            base, _ = os.path.splitext(self.cfg.file)
        else:
            base = os.path.join(self.cfg.base_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(os.path.dirname(base) or os.getcwd(), exist_ok=True)
        return f"{base}.{ext}"

    def _init_files(self) -> None:
        mode = (self.cfg.mode or "terminal").lower()
        if mode in ("file", "all"):
            path = self._default_path("txt")
            self._files["txt"] = open(path, "w", encoding="utf-8")
            self.txt_path = path
        if mode in ("srt", "all"):
            path = self._default_path("srt")
            self._files["srt"] = open(path, "w", encoding="utf-8")
            self.srt_path = path
        if mode in ("jsonl", "all"):
            path = self._default_path("jsonl")
            self._files["jsonl"] = open(path, "w", encoding="utf-8")
            self.jsonl_path = path

    def emit(self, rec: SegmentRecord) -> None:
        self.segments.append(rec)
        if getattr(self.cfg, "show_transcript_live", True):
            self._print_live(rec)
        self._write(rec)

    def _print_live(self, rec: SegmentRecord) -> None:
        lang = rec.language if (getattr(self.cfg, "show_language", True) and rec.language) else None
        prefix = f"[{lang}] " if lang else ""
        # 用 \r 覆盖当前行,模拟实时打字机效果的分段输出
        print(f"{prefix}{rec.text}", flush=True)

    def _write(self, rec: SegmentRecord) -> None:
        if "txt" in self._files:
            self._files["txt"].write(rec.text + "\n")
            self._files["txt"].flush()
        if "srt" in self._files:
            idx = self.segments.index(rec) + 1  # srt 从 1 开始编号
            block = (
                f"{idx}\n{_fmt_srt_ts(rec.start)} --> {_fmt_srt_ts(rec.end)}\n{rec.text}\n\n"
            )
            self._files["srt"].write(block)
            self._files["srt"].flush()
        if "jsonl" in self._files:
            self._files["jsonl"].write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            self._files["jsonl"].flush()

    @property
    def full_text(self) -> str:
        return "\n".join(s.text for s in self.segments)

    def close(self) -> None:
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass
        self._files.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
