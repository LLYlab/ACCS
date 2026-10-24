from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .audio import create_audio_source, AudioSource
from .config import Config
from .output import OutputSink, SegmentRecord
from .segmenter import SegmentChunk, Segmenter
from .transcriber import create_transcriber, Transcriber
from .vad import create_vad, BaseVAD


@dataclass
class Utterance:
    audio: np.ndarray  # float32 16k mono
    start: float       # 相对采集开始的秒数
    end: float
    index: int = 0
    reason: str = "silence"


class Pipeline:
    """实时转写流水线：采集 → VAD → 分段 → 后台转录 → 输出。

    - 采集与分段跑在调用线程（实时、不阻塞）
    - 转录在 worker 线程，模型慢也不影响继续采集
    - 分段由 `segmenter.Segmenter` 负责（双阈值、能量谷切分、重叠）
    """

    def __init__(self, config: Config):
        self.config = config
        self.source: AudioSource = create_audio_source(config.audio)
        self.transcriber: Transcriber = create_transcriber(config.transcriber)
        self.vad: Optional[BaseVAD] = create_vad(config.vad) if config.vad.enabled else None
        self.output: OutputSink = OutputSink(config.output)

        self._work: "queue.Queue[Utterance]" = queue.Queue(maxsize=128)
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

        # 回调（Web 端用来切 UI 状态 / 推送统计）
        self.on_ready: Optional[Callable[[], None]] = None
        self.on_stats: Optional[Callable[[dict], None]] = None
        self.on_segment: Optional[Callable[[SegmentRecord], None]] = None

        self._sample_rate = config.audio.sample_rate
        self._frame_size = config.audio.frame_size
        self._segmenter = Segmenter(config.vad, self._sample_rate)
        self._utts: List[Utterance] = []
        self._t0 = 0.0

    # ---------------------------------------------------------------- 主循环
    def run(self) -> None:
        print("== ACCS 实时转写启动 ==", flush=True)
        print(f"  音频来源: {self.config.audio.source}  采样率: {self._sample_rate}", flush=True)
        print(f"  转录引擎: {self.config.transcriber.engine}", flush=True)
        print("  按 Ctrl+C 停止。", flush=True)

        self.source.open()
        try:
            self.transcriber.load()
            if self.vad is not None:
                self.vad.load()
        except Exception as e:
            print(f"[accs] 初始化失败: {e}", flush=True)
            raise

        self._stop.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="accs-transcriber", daemon=True)
        self._worker.start()
        if self.on_ready:
            try:
                self.on_ready()
            except Exception:
                pass

        self._t0 = time.time()
        try:
            self._consume()
        finally:
            self._stop.set()

        self._work.join()
        self._worker.join(timeout=8)
        self.source.close()
        self.transcriber.close()
        try:
            self.output.close()
        except Exception:
            pass
        print("\n== 完成 ==", flush=True)

    def _consume(self) -> None:
        last_stats = 0.0
        frames = 0
        max_prob = 0.0
        max_level = 0.0
        last_level = 0.0
        try:
            for frame in self.source.stream():
                if self._stop.is_set():
                    break
                frame = np.asarray(frame, dtype=np.float32).reshape(-1)
                if frame.size == 0:
                    continue
                frames += 1
                level = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
                last_level = level
                prob = 1.0 if self.vad is None else float(self.vad.probability(frame))
                max_prob = max(max_prob, prob)
                max_level = max(max_level, level)
                if frames == 1:
                    print(f"[accs] 首帧 {frame.size} 采样, vad={getattr(self.vad, 'name', 'none')}, prob={prob:.3f}",
                          flush=True)
                for chunk in self._segmenter.push(frame, prob):
                    self._enqueue(chunk)

                now = time.time()
                if self.on_stats and now - last_stats > 0.5:
                    last_stats = now
                    try:
                        stats = dict(self._segmenter.stats)
                        stats.update({"frames": frames, "max_prob": round(max_prob, 3),
                                      "level": round(last_level, 4), "max_level": round(max_level, 4),
                                      "vad": getattr(self.vad, "name", "none")})
                        self.on_stats(stats)
                    except Exception:
                        pass
            print(f"[accs] 采集结束：{frames} 帧, max_prob={max_prob:.3f}", flush=True)
        except KeyboardInterrupt:
            self._stop.set()
        finally:
            for chunk in self._segmenter.flush():
                self._enqueue(chunk)
            self.source.close()

    def _enqueue(self, chunk: SegmentChunk) -> None:
        utt = Utterance(audio=chunk.audio, start=chunk.start, end=chunk.end,
                        index=chunk.index, reason=chunk.reason)
        self._utts.append(utt)
        try:
            self._work.put(utt, timeout=2)
        except queue.Full:
            print("[accs] 转录队列已满，丢弃一段（模型太慢？）", flush=True)

    # ---------------------------------------------------------------- worker
    def _worker_loop(self) -> None:
        while True:
            try:
                utt = self._work.get(timeout=0.2)
            except queue.Empty:
                if self._stop.is_set() and self._work.empty():
                    break
                continue
            try:
                result = self.transcriber.transcribe(utt.audio)
                if result.text:
                    rec = SegmentRecord(
                        start=utt.start, end=utt.end, text=result.text,
                        language=result.language, index=utt.index,
                        reason=utt.reason,
                        words=[(w.start, w.end, w.word) for w in (result.words or [])],
                        engine=self.config.transcriber.engine,
                    )
                    self.output.emit(rec)
                    if self.on_segment:
                        try:
                            self.on_segment(rec)
                        except Exception:
                            pass
            except Exception as e:  # pragma: no cover
                print(f"[accs] 转录出错: {e}", flush=True)
            finally:
                self._work.task_done()

    # ---------------------------------------------------------------- 控制
    def stop(self) -> None:
        self._stop.set()
        try:
            self.source.close()
        except Exception:
            pass
