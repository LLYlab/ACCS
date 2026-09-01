from __future__ import annotations

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config import Config
from .output import SegmentRecord
from .pipeline import Pipeline
from .translator import Translator, text_needs_translation

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
INDEX_PATH = os.path.join(STATIC_DIR, "index.html")


# ---------------------------------------------------------------- 广播器
class Broadcaster:
    """线程安全地把事件推送给多个 SSE 客户端,并保留历史用于页面刷新重放。"""

    def __init__(self, history_cap: int = 800):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subs: set = set()
        self._lock = threading.Lock()
        self.history: List[dict] = []
        self.cap = history_cap

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self):
        q = asyncio.Queue(maxsize=500)
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


# ---------------------------------------------------------------- 输出汇(取代 OutputSink)
class WebSink:
    """把转写片段广播到 UI,并在后台调用本地 LLM 翻译。"""

    def __init__(self, broadcaster: Broadcaster, translator: Translator, translate_enabled: bool):
        self.broadcaster = broadcaster
        self.translator = translator
        self.translate_enabled = translate_enabled
        self.segments: List[SegmentRecord] = []
        self._rows: List[dict] = []
        # 2 路翻译 worker: 让多个片段的译文更快、更连续地补上(原文始终先行)
        self._ex = ThreadPoolExecutor(max_workers=2, thread_name_prefix="accs-translate")

    def emit(self, rec: SegmentRecord) -> None:
        self.segments.append(rec)
        # 记录用于导出的富文本行(原文+译文)
        self._rows.append({
            "start": round(rec.start, 3),
            "end": round(rec.end, 3),
            "text": rec.text,
            "language": rec.language,
            "translation": "",
        })
        # 判断是否需要翻译: 英文/中英混合 -> 需要; 纯中文 -> 原文即中文,只显示原文
        needs = text_needs_translation(rec.text)
        self.broadcaster.publish({
            "type": "transcript",
            "start": round(rec.start, 3),
            "end": round(rec.end, 3),
            "text": rec.text,
            "language": rec.language,
            "needs_translation": needs,
        })
        # 只有需要翻译(含英文)且开启翻译时才提交;纯中文不需要译文
        if self.translate_enabled and needs:
            self._ex.submit(self._translate, rec)

    def _translate(self, rec: SegmentRecord) -> None:
        try:
            translation = self.translator.translate(rec.text)
        except Exception as e:  # pragma: no cover
            translation = f"⚠ 翻译失败: {e}"
        # 回填到行
        for row in self._rows:
            if abs(row["start"] - round(rec.start, 3)) < 1e-6:
                row["translation"] = translation
                break
        self.broadcaster.publish({
            "type": "translation",
            "start": round(rec.start, 3),
            "end": round(rec.end, 3),
            "original": rec.text,
            "translation": translation,
            "language": rec.language,
        })

    def to_rows(self) -> List[dict]:
        return list(self._rows)

    def full_text(self) -> str:
        parts = []
        for r in self._rows:
            parts.append(r["text"])
            if r["translation"] and r["translation"] != r["text"]:
                parts.append("　↳ " + r["translation"])
        return "\n".join(parts)

    def shutdown(self) -> None:
        try:
            self._ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


# ---------------------------------------------------------------- 应用状态
class AppState:
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.thread: Optional[threading.Thread] = None
        self.running: bool = False
        self.broadcaster = Broadcaster()
        self.translator: Optional[Translator] = None
        self.sink: Optional[WebSink] = None
        self.phase = "idle"  # idle | loading | running | stopping
        self.last_analysis: Optional[dict] = None

    def cleanup(self) -> None:
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        if self.sink:
            self.sink.shutdown()


state = AppState()


# ---------------------------------------------------------------- 请求/响应模型
class StartRequest(BaseModel):
    source: str = "system"          # system | mic | file
    file: str = ""
    model: str = "small"
    engine: str = "sherpa_onnx"     # sherpa_onnx | faster_whisper | openai_whisper | funasr
    funasr_model: str = "paraformer-zh"
    sherpa_onnx_model: str = ""     # sherpa-onnx 模型目录(空=自动探测)
    language: str = ""              # "" 自动
    device: str = "auto"            # auto | cpu | cuda
    translate: bool = True
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "deepseek-r1:8b"
    silence_ms: int = 800
    mode: str = "terminal"


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.broadcaster.bind_loop(asyncio.get_running_loop())
    yield
    state.cleanup()


app = FastAPI(title="ACCS", lifespan=lifespan)


# ---------------------------------------------------------------- 路由
@app.get("/", response_class=HTMLResponse)
async def index():
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>index.html 缺失</h1>")


@app.get("/api/status")
async def status():
    return {
        "running": state.running,
        "phase": state.phase,
        "llm_online": (state.translator.available() if state.translator else False),
        "segments": len(state.sink.segments) if state.sink else 0,
    }


@app.post("/api/start")
async def start(req: StartRequest):
    if state.running:
        raise HTTPException(400, "已经在运行中")

    overrides = {
        "audio.source": req.source,
        "audio.file": req.file,
        "transcriber.model": req.model,
        "transcriber.engine": req.engine,
        "transcriber.funasr_model": req.funasr_model,
        "transcriber.sherpa_onnx_model": req.sherpa_onnx_model,
        "transcriber.language": req.language,
        "transcriber.device": req.device,
    }
    if req.silence_ms:
        overrides["vad.silence_ms"] = req.silence_ms

    try:
        config = Config.load(overrides=overrides)
        # file 模式必须给文件
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

    def _on_ready() -> None:
        # 模型加载完、开始采集时,UI 从“加载中”切到“识别中”
        state.phase = "running"
        state.broadcaster.publish({"type": "status", "running": True, "phase": "running"})

    pipeline.on_ready = _on_ready

    state.phase = "loading"
    state.running = True
    state.broadcaster.publish({"type": "status", "running": True, "phase": "loading"})
    state.broadcaster.publish({"type": "info", "message": f"加载模型 {req.model} …"})

    def _run():
        try:
            pipeline.run()
        except Exception as e:
            state.broadcaster.publish({"type": "error", "message": str(e)})
        finally:
            state.running = False
            state.phase = "idle"
            state.broadcaster.publish({"type": "status", "running": False, "phase": "idle"})

    state.thread = threading.Thread(target=_run, name="accs-pipeline", daemon=True)
    state.thread.start()
    return {"ok": True, "phase": "loading"}


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
        state.thread.join(timeout=8)
    if state.sink:
        state.sink.shutdown()
    state.running = False
    state.phase = "idle"
    state.broadcaster.publish({"type": "status", "running": False, "phase": "idle"})
    return {"ok": True}


@app.get("/api/stream")
async def stream():
    q, history = state.broadcaster.subscribe()

    async def gen():
        try:
            if history:
                yield _sse({"type": "reset", "segments": [e for e in history if e["type"] in ("transcript", "translation")]})
            while True:
                evt = await q.get()
                yield _sse(evt)
        except asyncio.CancelledError:
            pass
        finally:
            state.broadcaster.unsubscribe(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------- 导出
@app.get("/api/export")
async def export(format: str = "txt"):
    rows = state.sink.to_rows() if state.sink else []
    analysis = state.last_analysis

    def _srt_ts(sec: float) -> str:
        ms = int(round(sec * 1000))
        h, ms = divmod(ms, 3600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    f = (format or "txt").lower()
    stem = "accs_course"
    content = ""
    media = "text/plain; charset=utf-8"
    if f == "jsonl":
        import json as _json
        content = "\n".join(_json.dumps(r, ensure_ascii=False) for r in rows)
        media = "application/x-ndjson; charset=utf-8"
        stem += ".jsonl"
    elif f == "srt":
        lines = []
        for i, r in enumerate(rows, 1):
            lines.append(str(i))
            lines.append(f"{_srt_ts(r['start'])} --> {_srt_ts(r['end'])}")
            lines.append(r["text"])
            if r["translation"] and r["translation"] != r["text"]:
                lines.append(r["translation"])
            lines.append("")
        content = "\n".join(lines)
        media = "text/plain; charset=utf-8"
        stem += ".srt"
    elif f == "md":
        out = ["# ACCS 课程转录（含译文）", ""]
        for r in rows:
            out.append(f"- **{r['text']}**")
            if r["translation"] and r["translation"] != r["text"]:
                out.append(f"  > {r['translation']}")
        if analysis and analysis.get("corrected"):
            out += ["", "## 纠正后的课程内容", "", analysis["corrected"]]
        if analysis and analysis.get("knowledge_points"):
            out += ["", "## 知识要点", ""]
            out += [f"- {k}" for k in analysis["knowledge_points"]]
        content = "\n".join(out)
        media = "text/markdown; charset=utf-8"
        stem += ".md"
    else:  # txt
        out = []
        for r in rows:
            out.append(r["text"])
            if r["translation"] and r["translation"] != r["text"]:
                out.append("　↳ " + r["translation"])
        if analysis and analysis.get("corrected"):
            out += ["", "===== 纠正后的课程内容 =====", "", analysis["corrected"]]
        if analysis and analysis.get("knowledge_points"):
            out += ["", "===== 知识要点 =====", ""]
            out += [f"- {k}" for k in analysis["knowledge_points"]]
        content = "\n".join(out)
        stem += ".txt"

    from fastapi.responses import Response
    return Response(
        content=content,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{stem}"'},
    )


# ---------------------------------------------------------------- 全面分析
@app.post("/api/analyze")
async def analyze():
    if not (state.sink and state.sink.to_rows()):
        raise HTTPException(400, "没有可分析的录音内容。请先开始识别并采集语音。")
    if not (state.translator and state.translator.available()):
        raise HTTPException(400, "离线翻译/分析服务(Ollama)不可用。")
    transcript = "\n".join(
        (r["text"] + ("\n[" + r["translation"] + "]" if r["translation"] and r["translation"] != r["text"] else ""))
        for r in state.sink.to_rows()
    )
    state.broadcaster.publish({"type": "analysis_status", "state": "analyzing"})

    def _work() -> dict:
        try:
            result = state.translator.analyze_course(transcript)
        except Exception as e:
            result = {"corrected": "", "knowledge_points": [], "error": str(e)}
        return result

    def _run() -> None:
        res = _work()
        state.last_analysis = res
        state.broadcaster.publish({"type": "analysis", "data": res})

    threading.Thread(target=_run, name="accs-analyze", daemon=True).start()
    return {"ok": True, "state": "analyzing"}


@app.get("/api/analysis")
async def analysis():
    return state.last_analysis or {}
