"""导出层：把转写片段变成可交付的文件。

支持：txt / md / srt（原文或双语）/ vtt / json / jsonl / csv / docx / zip 打包。
所有函数都是纯函数（输入 rows + meta，输出 str 或 bytes），Web 与 CLI 共用。
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

APP_NAME = "ACCS"
APP_VERSION = "2.0.0"


# ============================================================ 数据结构
@dataclass
class ExportRow:
    start: float
    end: float
    text: str
    translation: str = ""
    language: str = ""
    index: int = 0
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ExportMeta:
    title: str = "课程录音"
    source: str = ""            # system | mic | file | browser
    engine: str = ""
    model: str = ""
    language: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    extra: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "app": APP_NAME, "version": APP_VERSION,
            "title": self.title, "source": self.source,
            "engine": self.engine, "model": self.model, "language": self.language,
            "started_at": self.started_at, "segments": None, **self.extra,
        }


# ============================================================ 时间格式
def fmt_srt_ts(sec: float) -> str:
    ms = int(round(max(0.0, sec) * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_vtt_ts(sec: float) -> str:
    return fmt_srt_ts(sec).replace(",", ".")


def fmt_clock(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ============================================================ 自动分章
def auto_chapters(rows: Sequence[ExportRow], gap_seconds: float = 2.5,
                  min_chapter_seconds: float = 10.0) -> List[dict]:
    """按"相邻片段间隔"把课程切成章节。

    间隔 > gap_seconds 视为一次停顿/换话题；章节太短则并入上一章。
    """
    if not rows:
        return []
    chapters: List[dict] = []
    cur = {"start": rows[0].start, "end": rows[0].end, "rows": [rows[0]]}
    for prev, row in zip(rows, rows[1:]):
        gap = row.start - prev.end
        if gap >= gap_seconds:
            if chapters and (cur["end"] - cur["start"]) < min_chapter_seconds:
                # 太短：并到上一章
                chapters[-1]["end"] = cur["end"]
                chapters[-1]["rows"].extend(cur["rows"])
            else:
                chapters.append(cur)
            cur = {"start": row.start, "end": row.end, "rows": [row]}
        else:
            cur["end"] = row.end
            cur["rows"].append(row)
    chapters.append(cur)

    for i, ch in enumerate(chapters, 1):
        text = " ".join(r.text for r in ch["rows"]).strip()
        ch["index"] = i
        ch["title"] = _chapter_title(text, i)
        ch["text"] = text
        ch["duration"] = round(ch["end"] - ch["start"], 2)
    return chapters


def _chapter_title(text: str, index: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return f"第 {index} 节"
    # 取到第一个句号/问号/感叹号或 24 个字
    m = re.search(r"[。！？!?\.]", clean)
    head = clean[: m.start() + 1] if m and m.start() < 30 else clean[:24]
    return head + ("…" if len(clean) > len(head) else "")


# ============================================================ 各格式
def to_txt(rows: Sequence[ExportRow], meta: Optional[ExportMeta] = None,
           with_time: bool = True, with_translation: bool = True) -> str:
    out: List[str] = []
    if meta:
        out.append(f"{meta.title}")
        out.append(f"来源: {meta.source}  引擎: {meta.engine}  模型: {meta.model}  时间: {meta.started_at}")
        out.append("")
    for r in rows:
        prefix = f"[{fmt_clock(r.start)}] " if with_time else ""
        out.append(prefix + r.text)
        if with_translation and r.translation and r.translation != r.text:
            out.append("    ↳ " + r.translation)
    return "\n".join(out)


def to_markdown(rows: Sequence[ExportRow], meta: Optional[ExportMeta] = None,
                chapters: bool = True, with_time: bool = True) -> str:
    out: List[str] = [f"# {meta.title if meta else '课程转录'}", ""]
    if meta:
        out += [
            f"- **来源**：{meta.source or '—'}",
            f"- **引擎**：{meta.engine or '—'}　**模型**：{meta.model or '—'}",
            f"- **开始时间**：{meta.started_at}",
            f"- **片段数**：{len(rows)}",
            "",
        ]
    groups = auto_chapters(rows) if chapters else [{"title": "", "rows": rows, "start": 0}]
    for ch in groups:
        if chapters:
            out += [f"## {ch['title']}", ""]
            out.append(f"> {fmt_clock(ch['start'])} 开始 · 约 {int(ch['duration'] // 60)} 分 {int(ch['duration'] % 60)} 秒")
            out.append("")
        for r in ch["rows"]:
            prefix = f"**{fmt_clock(r.start)}** " if with_time else ""
            out.append(f"{prefix}{r.text}")
            if r.translation and r.translation != r.text:
                out.append("")
                out.append(f"> {r.translation}")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def to_srt(rows: Sequence[ExportRow], bilingual: bool = False) -> str:
    blocks: List[str] = []
    for i, r in enumerate(rows, 1):
        block = [str(i), f"{fmt_srt_ts(r.start)} --> {fmt_srt_ts(r.end)}", r.text]
        if bilingual and r.translation and r.translation != r.text:
            block.append(r.translation)
        blocks.append("\n".join(block))
    return "\n\n".join(blocks) + "\n"


def to_vtt(rows: Sequence[ExportRow], bilingual: bool = False) -> str:
    blocks: List[str] = ["WEBVTT", ""]
    for i, r in enumerate(rows, 1):
        block = [str(i), f"{fmt_vtt_ts(r.start)} --> {fmt_vtt_ts(r.end)}", r.text]
        if bilingual and r.translation and r.translation != r.text:
            block.append(r.translation)
        blocks.append("\n".join(block))
    return "\n\n".join(blocks) + "\n"


def to_json(rows: Sequence[ExportRow], meta: Optional[ExportMeta] = None,
            include_chapters: bool = True) -> str:
    data = {
        "meta": (meta.as_dict() if meta else None),
        "segments": [
            {
                "index": r.index or i,
                "start": round(r.start, 3),
                "end": round(r.end, 3),
                "duration": round(r.duration, 3),
                "text": r.text,
                "translation": r.translation,
                "language": r.language,
                "confidence": round(r.confidence, 4),
            }
            for i, r in enumerate(rows, 1)
        ],
    }
    if meta:
        data["meta"]["segments"] = len(rows)
        data["meta"]["duration"] = round(sum(r.duration for r in rows), 2)
    if include_chapters:
        data["chapters"] = [
            {"index": c["index"], "title": c["title"], "start": c["start"], "end": c["end"],
             "text": c["text"]}
            for c in auto_chapters(rows)
        ]
    return json.dumps(data, ensure_ascii=False, indent=2)


def to_jsonl(rows: Sequence[ExportRow]) -> str:
    return "\n".join(json.dumps({
        "start": round(r.start, 3), "end": round(r.end, 3),
        "text": r.text, "translation": r.translation, "language": r.language,
    }, ensure_ascii=False) for r in rows) + "\n"


def to_csv(rows: Sequence[ExportRow]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["index", "start", "end", "duration_s", "language", "text", "translation"])
    for i, r in enumerate(rows, 1):
        w.writerow([i, f"{r.start:.3f}", f"{r.end:.3f}", f"{r.duration:.3f}",
                    r.language, r.text, r.translation])
    return buf.getvalue()


def to_docx(rows: Sequence[ExportRow], meta: Optional[ExportMeta] = None) -> bytes:
    """Word 文档（需要 python-docx）。带标题、元信息、章节与时间戳。"""
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError as e:
        raise RuntimeError("导出 DOCX 需要 python-docx：pip install python-docx") from e

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Microsoft YaHei"
    style.font.size = Pt(10.5)

    h = doc.add_heading(meta.title if meta else "课程转录", level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT

    if meta:
        p = doc.add_paragraph()
        run = p.add_run(
            f"来源：{meta.source or '—'}　引擎：{meta.engine or '—'}　模型：{meta.model or '—'}　"
            f"时间：{meta.started_at}　片段：{len(rows)}"
        )
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    for ch in auto_chapters(rows):
        doc.add_heading(ch["title"], level=1)
        doc.add_paragraph(f"{fmt_clock(ch['start'])} 开始 · 约 {int(ch['duration'] // 60)} 分钟")
        for r in ch["rows"]:
            p = doc.add_paragraph()
            t = p.add_run(f"{fmt_clock(r.start)}  ")
            t.font.size = Pt(8)
            t.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
            p.add_run(r.text)
            if r.translation and r.translation != r.text:
                p2 = doc.add_paragraph()
                run = p2.add_run("　" + r.translation)
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(0x44, 0x66, 0x99)
                p2.paragraph_format.left_indent = Pt(12)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def to_zip(rows: Sequence[ExportRow], meta: Optional[ExportMeta] = None,
           include_docx: bool = True) -> bytes:
    """打成一个 zip：文本、字幕、Markdown、JSON、CSV、Word（可选）、元数据。"""
    meta = meta or ExportMeta()
    files: Dict[str, bytes] = {
        "transcript.txt": to_txt(rows, meta).encode("utf-8"),
        "transcript.md": to_markdown(rows, meta).encode("utf-8"),
        "subtitles.srt": to_srt(rows, bilingual=False).encode("utf-8"),
        "subtitles_bilingual.srt": to_srt(rows, bilingual=True).encode("utf-8"),
        "subtitles.vtt": to_vtt(rows, bilingual=True).encode("utf-8"),
        "transcript.json": to_json(rows, meta).encode("utf-8"),
        "transcript.jsonl": to_jsonl(rows).encode("utf-8"),
        "transcript.csv": ("\ufeff" + to_csv(rows)).encode("utf-8"),
        "meta.json": json.dumps({
            **(meta.as_dict() or {}),
            "segments": len(rows),
            "duration": round(sum(r.duration for r in rows), 2),
        }, ensure_ascii=False, indent=2).encode("utf-8"),
    }
    if include_docx:
        try:
            files["transcript.docx"] = to_docx(rows, meta)
        except Exception:
            pass  # 没装 python-docx 就不塞 docx，其余照常
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


EXPORT_FORMATS = {
    "txt": ("text/plain; charset=utf-8", "txt"),
    "md": ("text/markdown; charset=utf-8", "md"),
    "srt": ("text/plain; charset=utf-8", "srt"),
    "srt_bilingual": ("text/plain; charset=utf-8", "zh-en.srt"),
    "vtt": ("text/vtt; charset=utf-8", "vtt"),
    "json": ("application/json; charset=utf-8", "json"),
    "jsonl": ("application/x-ndjson; charset=utf-8", "jsonl"),
    "csv": ("text/csv; charset=utf-8", "csv"),
    "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
    "zip": ("application/zip", "zip"),
}


def build(format: str, rows: Sequence[ExportRow], meta: Optional[ExportMeta] = None):
    """统一入口：返回 (bytes, media_type, filename_suffix)。"""
    f = (format or "txt").lower()
    if f == "zip":
        return to_zip(rows, meta), *EXPORT_FORMATS["zip"]
    if f == "docx":
        return to_docx(rows, meta), *EXPORT_FORMATS["docx"]
    if f == "srt_bilingual":
        return to_srt(rows, True).encode("utf-8"), *EXPORT_FORMATS["srt_bilingual"]
    if f == "srt":
        return to_srt(rows, False).encode("utf-8"), *EXPORT_FORMATS["srt"]
    if f == "vtt":
        return to_vtt(rows, True).encode("utf-8"), *EXPORT_FORMATS["vtt"]
    if f == "json":
        return to_json(rows, meta).encode("utf-8"), *EXPORT_FORMATS["json"]
    if f == "jsonl":
        return to_jsonl(rows).encode("utf-8"), *EXPORT_FORMATS["jsonl"]
    if f == "csv":
        return ("\ufeff" + to_csv(rows)).encode("utf-8"), *EXPORT_FORMATS["csv"]
    if f == "md":
        return to_markdown(rows, meta).encode("utf-8"), *EXPORT_FORMATS["md"]
    return to_txt(rows, meta).encode("utf-8"), *EXPORT_FORMATS["txt"]
