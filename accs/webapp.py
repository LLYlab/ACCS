from __future__ import annotations

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import exporters as X
from .paths import app_root, static_dir
from .config import Config
from .output import SegmentRecord
from .pipeline import Pipeline
from .transcriber import engine_catalog, normalize_engine
from .translator import Translator, text_needs_translation

STATIC_DIR = static_dir()
INDEX_PATH = os.path.join(STATIC_DIR, "index.html")
PROJECT_ROOT = app_root()


# ================================================================ 广播
class Broadcaster:
    """线程安全地把事件推给多个 SSE 客户端，并保留历史用于刷新重放。"""

    def __init__(self, history_cap: int = 1000):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subs: set = set()
        self._lock = threading.Lock()
        self.history: List[dict] = []
        self.cap = history_cap

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self):
        q = asyncio.Queue(maxsize=1000)
        with self._lock:
            self._subs.add(q)
            history = list(self.history)
        return q, history

    def unsubscribe(self, q) -> None:
        with self._lock:
            self._subs.discard(q)

    def publish(self, event: dict) -> None:
        with self._lock:
            self.history.append(event)
            if len(self.history) > self.cap:
                self.history = self.history[-self.cap:]
            subs = list(self._subs)
        if self._loop is None:
            return
        for q in subs:
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                pass

    def clear(self) -> None:
        with self._lock:
            self.history.clear()


# ================================================================ 输出汇
class WebSink:
    """把片段广播到 UI，并在后台调用本地 LLM 翻译。"""

    def __init__(self, broadcaster: Broadcaster, translator: Translator, translate_enabled: bool):
        self.broadcaster = broadcaster
        self.translator = translator
        self.translate_enabled = translate_enabled
        self.segments: List[SegmentRecord] = []
        self._rows: List[dict] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._ex = ThreadPoolExecutor(max_workers=3, thread_name_prefix="accs-translate")

    def _next_id(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def emit(self, rec: SegmentRecord) -> None:
        sid = self._next_id()
        with self._lock:
            self.segments.append(rec)
            self._rows.append({
                "id": sid,
                "start": round(rec.start, 3),
                "end": round(rec.end, 3),
                "text": rec.text,
                "language": rec.language,
                "engine": rec.engine,
                "confidence": round(rec.confidence, 4),
                "translation": "",
            })
        needs = text_needs_translation(rec.text)
        self.broadcaster.publish({
            "type": "segment", "id": sid,
            "start": round(rec.start, 3), "end": round(rec.end, 3),
            "text": rec.text, "language": rec.language,
            "engine": rec.engine, "confidence": round(rec.confidence, 4),
            "needs_translation": needs,
        })
        if self.translate_enabled and needs:
            self._ex.submit(self._translate, sid, rec)

    def emit_text(self, text: str, start: float, end: float,
                  language: str = "", engine: str = "browser") -> None:
        self.emit(SegmentRecord(start=start, end=end, text=text,
                                language=language, engine=engine))

    def _translate(self, sid: int, rec: SegmentRecord) -> None:
        try:
            translation = self.translator.translate(rec.text)
        except Exception as e:  # pragma: no cover
            translation = f"⚠ 翻译失败: {e}"
        with self._lock:
            for row in self._rows:
                if row["id"] == sid:
                    row["translation"] = translation
                    break
        self.broadcaster.publish({
            "type": "translation", "id": sid,
            "translation": translation, "original": rec.text,
        })

    def rows(self) -> List[dict]:
        with self._lock:
            return list(self._rows)

    def to_export_rows(self) -> List[X.ExportRow]:
        return [
            X.ExportRow(
                start=r["start"], end=r["end"], text=r["text"],
                translation=r.get("translation", ""), language=r.get("language", ""),
                index=i, confidence=r.get("confidence", 0.0),
            )
            for i, r in enumerate(self.rows(), 1)
        ]

    def full_text(self) -> str:
        parts = []
        for r in self.rows():
            parts.append(r["text"])
            if r["translation"] and r["translation"] != r["text"]:
                parts.append("　↳ " + r["translation"])
        return "\n".join(parts)

    def shutdown(self) -> None:
        try:
            self._ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


# ================================================================ 状态
class AppState:
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.thread: Optional[threading.Thread] = None
        self.running: bool = False
        self.broadcaster = Broadcaster()
        self.translator: Optional[Translator] = None
        self.sink: Optional[WebSink] = None
        self.phase = "idle"      # idle | loading | running | stopping
        self.mode = "pipeline"   # pipeline | browser
        self.meta: dict = {}
        self.last_analysis: Optional[dict] = None
        self.last_error: str = ""
        self.stats: dict = {}

    def cleanup(self) -> None:
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        if self.sink:
            self.sink.shutdown()


state = AppState()


# ================================================================ 请求模型
class StartRequest(BaseModel):
    source: str = "system"
    file: str = ""
    engine: str = "sherpa_onnx"
    model: str = "small"
    funasr_model: str = "paraformer-zh"
    sherpa_onnx_model: str = ""
    sherpa_onnx_type: str = "auto"
    vosk_model: str = ""
    whisper_cpp_bin: str = ""
    language: str = ""
    device: str = "auto"
    translate: bool = True
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "deepseek-r1:8b"
    silence_ms: int = 800
    threshold: float = 0.5
    overlap_ms: int = 0
    max_segment_ms: int = 30000


class BrowserSegmentRequest(BaseModel):
    text: str
    start: float = 0.0
    end: float = 0.0
    language: str = ""
    final: bool = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.broadcaster.bind_loop(asyncio.get_running_loop())
    yield
    state.cleanup()


app = FastAPI(title="ACCS", version=X.APP_VERSION, lifespan=lifespan)

# 静态资源（style.css / app.js / 图标）
try:
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
except Exception:  # pragma: no cover
    pass


# ================================================================ 路由
@app.get("/", response_class=HTMLResponse)
async def index():
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>index.html 缺失</h1>")


@app.get("/api/health")
async def health():
    return {"ok": True, "version": X.APP_VERSION}


@app.get("/api/engines")
async def engines():
    return {"engines": engine_catalog()}


@app.get("/api/models")
async def models():
    """扫描 models/ 下的本地模型（只列真正含模型文件的目录）。"""
    root = os.path.join(PROJECT_ROOT, "models")

    def dir_size(p: str) -> int:
        total = 0
        for dirpath, _, files in os.walk(p):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    pass
        return total

    def is_model_dir(p: str) -> bool:
        try:
            names = set(os.listdir(p))
        except OSError:
            return False
        if names & {"model.onnx", "model.int8.onnx", "tokens.txt"}:
            return True
        if any(n.startswith("ggml-") and n.endswith(".bin") for n in names):
            return True
        if "am" in names or "conf" in names:  # vosk
            return True
        return False

    def kind_of(name: str) -> str:
        low = name.lower()
        if "sense" in low:
            return "sherpa-sensevoice"
        if "paraformer" in low:
            return "sherpa-paraformer"
        if "zipformer" in low:
            return "sherpa-zipformer"
        if "vosk" in low:
            return "vosk"
        if "ggml" in low or "whisper" in low:
            return "whisper.cpp"
        return "sherpa" if "sherpa" in low else "unknown"

    found = []
    if os.path.isdir(root):
        candidates = []
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if not os.path.isdir(p):
                continue
            candidates.append(p)
            try:
                for sub in sorted(os.listdir(p)):
                    sp = os.path.join(p, sub)
                    if os.path.isdir(sp):
                        candidates.append(sp)
            except OSError:
                pass
        for p in candidates:
            if not is_model_dir(p):
                continue
            size = dir_size(p)
            if size < 5 * 1024 * 1024:
                continue
            name = os.path.basename(p)
            found.append({"name": name, "path": p,
                          "size_mb": round(size / 1024 / 1024, 1),
                          "kind": kind_of(name)})
    found.sort(key=lambda m: m["name"])
    return {"models": found}


@app.get("/api/status")
async def status():
    return {
        "running": state.running,
        "phase": state.phase,
        "mode": state.mode,
        "llm_online": (state.translator.available() if state.translator else False),
        "segments": len(state.sink.rows()) if state.sink else 0,
        "stats": state.stats,
        "last_error": state.last_error,
        "meta": state.meta,
    }


@app.post("/api/start")
async def start(req: StartRequest):
    if state.running:
        raise HTTPException(400, "已经在运行中")

    engine = normalize_engine(req.engine)

    # ---- 浏览器模式：后端不采集，只接收前端识别结果 ----
    if engine == "browser":
        state.translator = Translator(req.llm_base_url, req.llm_model)
        state.sink = WebSink(state.broadcaster, state.translator, req.translate)
        state.mode = "browser"
        state.running = True
        state.phase = "running"
        state.meta = {"source": "browser", "engine": engine, "model": "Web Speech API",
                      "language": req.language, "started_at": X.datetime.now().isoformat(timespec="seconds")}
        state.broadcaster.clear()
        state.broadcaster.publish({"type": "status", "running": True, "phase": "running", "mode": "browser"})
        return {"ok": True, "mode": "browser"}

    overrides = {
        "audio.source": req.source,
        "audio.file": req.file,
        "transcriber.engine": engine,
        "transcriber.model": req.model,
        "transcriber.funasr_model": req.funasr_model,
        "transcriber.sherpa_onnx_model": req.sherpa_onnx_model,
        "transcriber.sherpa_onnx_type": req.sherpa_onnx_type,
        "transcriber.vosk_model": req.vosk_model,
        "transcriber.whisper_cpp_bin": req.whisper_cpp_bin,
        "transcriber.language": req.language,
        "transcriber.device": req.device,
        "vad.silence_ms": req.silence_ms,
        "vad.threshold": req.threshold,
        "vad.overlap_ms": req.overlap_ms,
        "vad.max_segment_ms": req.max_segment_ms,
    }
    try:
        config = Config.load(overrides=overrides)
        if req.source == "file" and not req.file:
            raise HTTPException(400, "使用 file 输入时必须提供音频文件")
        pipeline = Pipeline(config)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"初始化失败: {e}") from e

    state.translator = Translator(req.llm_base_url, req.llm_model)
    state.sink = WebSink(state.broadcaster, state.translator, req.translate)
    pipeline.output = state.sink
    state.pipeline = pipeline
    state.mode = "pipeline"
    state.meta = {"source": req.source, "engine": engine, "model": req.model,
                  "language": req.language, "started_at": X.datetime.now().isoformat(timespec="seconds")}
    state.broadcaster.clear()

    def _on_ready() -> None:
        state.phase = "running"
        state.broadcaster.publish({"type": "status", "running": True, "phase": "running", "mode": "pipeline"})

    def _on_stats(stats: dict) -> None:
        state.stats = stats
        state.broadcaster.publish({"type": "stats", "stats": stats})

    def _on_segment(rec: SegmentRecord) -> None:
        state.broadcaster.publish({"type": "raw_segment", "id": rec.index,
                                   "duration": round(rec.end - rec.start, 2)})

    pipeline.on_ready = _on_ready
    pipeline.on_stats = _on_stats
    pipeline.on_segment = _on_segment

    state.phase = "loading"
    state.running = True
    state.last_error = ""
    state.broadcaster.publish({"type": "status", "running": True, "phase": "loading", "mode": "pipeline"})
    state.broadcaster.publish({"type": "info", "message": f"加载引擎 {engine} …"})

    def _run():
        try:
            pipeline.run()
        except Exception as e:
            import traceback
            traceback.print_exc()
            state.last_error = f"{type(e).__name__}: {e}"
            state.broadcaster.publish({"type": "error", "message": state.last_error})
        finally:
            state.running = False
            state.phase = "idle"
            state.broadcaster.publish({"type": "status", "running": False, "phase": "idle"})

    state.thread = threading.Thread(target=_run, name="accs-pipeline", daemon=True)
    state.thread.start()
    return {"ok": True, "phase": "loading", "mode": "pipeline"}


@app.post("/api/stop")
async def stop():
    if not state.running and not state.pipeline:
        raise HTTPException(400, "当前没有运行中的任务")
    state.phase = "stopping"
    state.broadcaster.publish({"type": "status", "running": True, "phase": "stopping"})
    if state.pipeline:
        try:
            state.pipeline.stop()
        except Exception:
            pass
    if state.thread:
        state.thread.join(timeout=10)
    if state.sink:
        state.sink.shutdown()
    state.running = False
    state.phase = "idle"
    state.pipeline = None
    state.broadcaster.publish({"type": "status", "running": False, "phase": "idle"})
    return {"ok": True}


@app.post("/api/browser/segment")
async def browser_segment(req: BrowserSegmentRequest):
    """浏览器识别模式：前端把最终识别的句子 POST 过来。"""
    if not (state.running and state.mode == "browser" and state.sink):
        raise HTTPException(400, "浏览器识别模式未启动")
    if req.text.strip():
        state.sink.emit_text(req.text.strip(), req.start, req.end, req.language, "browser")
    return {"ok": True}


@app.get("/api/stream")
async def stream():
    q, history = state.broadcaster.subscribe()

    async def gen():
        try:
            if history:
                yield _sse({"type": "reset", "events": [e for e in history if e["type"] in ("segment", "translation")]})
            while True:
                evt = await q.get()
                yield _sse(evt)
        except asyncio.CancelledError:
            pass
        finally:
            state.broadcaster.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ================================================================ 导出
@app.get("/api/export")
async def export(format: str = "txt"):
    rows = state.sink.to_export_rows() if state.sink else []
    if not rows:
        raise HTTPException(400, "还没有可导出的内容")
    meta = X.ExportMeta(
        title=(state.meta.get("title") or "课程录音"),
        source=state.meta.get("source", ""),
        engine=state.meta.get("engine", ""),
        model=state.meta.get("model", ""),
        language=state.meta.get("language", ""),
        started_at=state.meta.get("started_at", ""),
    )
    try:
        content, media, suffix = X.build(format, rows, meta)
    except Exception as e:
        raise HTTPException(400, f"导出失败: {e}") from e
    filename = f"accs_{X.datetime.now().strftime('%Y%m%d_%H%M%S')}.{suffix}"
    return Response(content=content, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/chapters")
async def chapters():
    rows = state.sink.to_export_rows() if state.sink else []
    return {"chapters": X.auto_chapters(rows)}


# ================================================================ 分析
@app.post("/api/analyze")
async def analyze():
    if not (state.sink and state.sink.rows()):
        raise HTTPException(400, "没有可分析的录音内容。请先开始识别并采集语音。")
    if not (state.translator and state.translator.available()):
        raise HTTPException(400, "本地分析服务(Ollama)不可用。")
    transcript = "\n".join(
        (r["text"] + ("\n[" + r["translation"] + "]" if r["translation"] and r["translation"] != r["text"] else ""))
        for r in state.sink.rows()
    )
    state.broadcaster.publish({"type": "analysis_status", "state": "analyzing"})

    def _run() -> None:
        try:
            res = state.translator.analyze_course(transcript)
        except Exception as e:
            res = {"corrected": "", "knowledge_points": [], "error": str(e)}
        state.last_analysis = res
        state.broadcaster.publish({"type": "analysis", "data": res})

    threading.Thread(target=_run, name="accs-analyze", daemon=True).start()
    return {"ok": True, "state": "analyzing"}


@app.get("/api/analysis")
async def analysis():
    return state.last_analysis or {}
