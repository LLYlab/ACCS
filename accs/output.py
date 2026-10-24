from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .config import OutputConfig
from . import exporters as X


@dataclass
class SegmentRecord:
    """一条已定稿的转写结果，时间轴为相对采集开始的秒数。"""
    start: float
    end: float
    text: str
    language: str = ""
    index: int = 0
    reason: str = ""
    words: Optional[List[tuple]] = None   # [(start, end, word), ...]
    engine: str = ""
    confidence: float = 0.0
    translation: str = ""

    def to_row(self) -> X.ExportRow:
        return X.ExportRow(
            start=self.start, end=self.end, text=self.text,
            translation=self.translation, language=self.language,
            index=self.index, confidence=self.confidence,
        )


def _fmt_srt_ts(sec: float) -> str:
    return X.fmt_srt_ts(sec)


class OutputSink:
    """CLI 输出：实时终端 + 落盘。

    mode:
      terminal  只在终端显示
      file      txt
      srt       srt
      jsonl     jsonl
      all       txt + srt + vtt + md + json + csv + jsonl
    """

    def __init__(self, cfg: OutputConfig):
        self.cfg = cfg
        self.segments: List[SegmentRecord] = []
        self._files = {}
        self._init_files()

    def _base(self) -> str:
        if self.cfg.file:
            return os.path.splitext(self.cfg.file)[0]
        base = os.path.join(self.cfg.base_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(os.path.dirname(base) or os.getcwd(), exist_ok=True)
        return base

    def _open(self, ext: str):
        path = f"{self._base()}.{ext}"
        os.makedirs(os.path.dirname(path) or os.getcwd(), exist_ok=True)
        return path, open(path, "w", encoding="utf-8")

    def _init_files(self) -> None:
        mode = (self.cfg.mode or "terminal").lower()
        if mode in ("file", "all"):
            path, f = self._open("txt")
            self._files["txt"] = f
            self.txt_path = path
        if mode in ("srt", "all"):
            path, f = self._open("srt")
            self._files["srt"] = f
            self.srt_path = path
        if mode in ("jsonl", "all"):
            path, f = self._open("jsonl")
            self._files["jsonl"] = f
            self.jsonl_path = path

    # ------------------------------------------------------------------
    def emit(self, rec: SegmentRecord) -> None:
        self.segments.append(rec)
        if getattr(self.cfg, "show_transcript_live", True):
            self._print_live(rec)
        self._write(rec)

    def _print_live(self, rec: SegmentRecord) -> None:
        lang = rec.language if (getattr(self.cfg, "show_language", True) and rec.language) else None
        prefix = f"[{lang}] " if lang else ""
        print(f"{prefix}{rec.text}", flush=True)

    def _write(self, rec: SegmentRecord) -> None:
        if "txt" in self._files:
            self._files["txt"].write(rec.text + "\n")
            self._files["txt"].flush()
        if "srt" in self._files:
            idx = len(self.segments)
            block = f"{idx}\n{X.fmt_srt_ts(rec.start)} --> {X.fmt_srt_ts(rec.end)}\n{rec.text}\n\n"
            self._files["srt"].write(block)
            self._files["srt"].flush()
        if "jsonl" in self._files:
            self._files["jsonl"].write(json.dumps({
                "start": round(rec.start, 3), "end": round(rec.end, 3),
                "text": rec.text, "language": rec.language, "engine": rec.engine,
            }, ensure_ascii=False) + "\n")
            self._files["jsonl"].flush()

    @property
    def full_text(self) -> str:
        return "\n".join(s.text for s in self.segments)

    def rows(self) -> List[X.ExportRow]:
        return [s.to_row() for s in self.segments]

    def meta(self) -> X.ExportMeta:
        return X.ExportMeta(
            title="课程录音", source="", engine="",
            model="", language="",
        )

    def close(self) -> None:
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass
        self._files.clear()
        # "all" 模式再补写别的格式
        if (self.cfg.mode or "").lower() == "all" and self.segments:
            rows = self.rows()
            base = self._base()
            extras = {
                "md": X.to_markdown(rows, self.meta()),
                "vtt": X.to_vtt(rows, bilingual=True),
                "json": X.to_json(rows, self.meta()),
                "csv": "\ufeff" + X.to_csv(rows),
            }
            for ext, content in extras.items():
                try:
                    with open(f"{base}.{ext}", "w", encoding="utf-8") as f:
                        f.write(content)
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
