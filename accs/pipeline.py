from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .audio import create_audio_source, AudioSource
from .config import Config
from .output import OutputSink, SegmentRecord
from .transcriber import create_transcriber, Transcriber
from .vad import create_vad, BaseVAD


@dataclass
class Utterance:
    audio: np.ndarray  # float32 16k mono
    start: float       # 相对采集开始的秒数
    end: float


class Pipeline:
    """实时转写流水线: 采集 -> VAD 端点切分 -> Whisper 转录 -> 输出。

    采集 + 端点检测运行在调用线程(不阻塞、实时);转录在后台 worker 线程,
    从而即使模型较慢也不会阻塞后续音频的采集。
    """

    def __init__(self, config: Config):
        self.config = config
        self.source: AudioSource = create_audio_source(config.audio)
        self.transcriber: Transcriber = create_transcriber(config.transcriber)
        self.vad: Optional[BaseVAD] = create_vad(config.vad) if config.vad.enabled else None
        self.output: OutputSink = OutputSink(config.output)

        self._work: "queue.Queue[Utterance]" = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._worker = None
        # 可选: 模型/采集就绪后的回调(Web 端用它对 UI 切换状态)
        self.on_ready = None

        self._total_samples = 0
        self._speech_active = False
        self._buf = np.zeros(0, dtype=np.float32)
        self._prepend = np.zeros(0, dtype=np.float32)
        self._silence_samples = 0
        self._seg_start_abs = 0.0
        self._frame_size = config.audio.frame_size
        self._sample_rate = config.audio.sample_rate

        self._init_vad_params()

    # ---- 端点参数(以采样点为单位) ----
    def _init_vad_params(self) -> None:
        vad = self.config.vad
        self._threshold = vad.threshold
        self._prepend_n = int(self._sample_rate * vad.prepend_ms / 1000)
        self._pad_n = int(self._sample_rate * vad.pad_ms / 1000)
        self._silence_n = int(self._sample_rate * vad.silence_ms / 1000)
        self._min_speech_n = int(self._sample_rate * vad.min_speech_ms / 1000)
        self._max_segm_n = int(self._sample_rate * vad.max_segment_ms / 1000)

    def _frame_prob(self, frame: np.ndarray) -> float:
        if self.vad is None:
            # 无 VAD 时,直接按固定窗口切分(每 max_segment_ms 一段)
            return 1.0
        return self.vad.probability(frame)

    def _consume(self) -> None:
        """读取音频帧并做端点切分。"""
        iterator = self.source.stream()
        try:
            for frame in iterator:
                if self._stop.is_set():
                    break
                frame = np.asarray(frame, dtype=np.float32).reshape(-1)
                if frame.size == 0:
                    continue
                self._process_frame(frame)
        except KeyboardInterrupt:
            self._stop.set()
        finally:
            self.source.close()
        # 结束前,强制收尾当前未定稿的片段
        self._finalize(force=True)

    def _process_frame(self, frame: np.ndarray) -> None:
        self._total_samples += frame.size
        prob = self._frame_prob(frame)

        if not self._speech_active:
            if prob >= self._threshold:
                # 起步: 用最近一段静音作为前置,避免截掉字头
                start_sample = self._total_samples - frame.size - self._prepend.size
                self._buf = np.concatenate([self._prepend, frame])
                self._seg_start_abs = max(0.0, start_sample / self._sample_rate)
                self._speech_active = True
                self._silence_samples = 0
                self._prepend = np.zeros(0, dtype=np.float32)
            else:
                # 维持前置静音滚动缓冲
                self._prepend = np.concatenate([self._prepend, frame])
                if self._prepend.size > self._prepend_n:
                    self._prepend = self._prepend[-self._prepend_n:]
            return

        # 处于讲话中
        self._buf = np.concatenate([self._buf, frame])
        if prob >= self._threshold:
            self._silence_samples = 0
        else:
            self._silence_samples += frame.size

        buf_len = self._buf.size
        if self._silence_samples >= self._silence_n:
            # 结尾静音达到阈值: 去掉超出 pad 的静音后定稿
            keep = buf_len - (self._silence_samples - self._pad_n)
            self._finalize(force=False, keep_samples=max(0, keep))
        elif buf_len >= self._max_segm_n:
            # 超长,强制定稿,不裁尾部
            self._finalize(force=True)

    def _finalize(self, force: bool, keep_samples: Optional[int] = None) -> None:
        if not self._speech_active and self._buf.size == 0:
            return
        audio = self._buf
        if keep_samples is not None:
            audio = audio[:keep_samples] if keep_samples > 0 else np.zeros(0, dtype=np.float32)

        if audio.size >= self._min_speech_n:
            end_abs = (self._total_samples - (self._silence_samples - self._pad_n)) / self._sample_rate
            end_abs = max(end_abs, self._seg_start_abs + audio.size / self._sample_rate)
            self._work.put(Utterance(audio=audio, start=self._seg_start_abs, end=end_abs))

        self._buf = np.zeros(0, dtype=np.float32)
        self._speech_active = False
        self._silence_samples = 0
        self._prepend = np.zeros(0, dtype=np.float32)

    # ---- 转录 worker ----
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
                    self.output.emit(SegmentRecord(
                        start=utt.start, end=utt.end, text=result.text, language=result.language,
                    ))
            except Exception as e:  # pragma: no cover
                print(f"[accs] 转录出错: {e}", flush=True)
            finally:
                # 无论成功与否都要标记完成,否则 run() 里的 join() 会一直阻塞
                self._work.task_done()

    def run(self) -> None:
        """阻塞运行直到停止(Ctrl+C)。"""
        print("== ACCS 实时转写启动 ==", flush=True)
        print(f"  音频来源: {self.config.audio.source}  采样率: {self._sample_rate}", flush=True)
        print(f"  转录引擎: {self.config.transcriber.engine}  模型: {self.config.transcriber.model}", flush=True)
        print("  按 Ctrl+C 停止。", flush=True)

        self.source.open()
        try:
            self.transcriber.load()
            self.vad.load() if self.vad is not None else None
        except Exception as e:
            print(f"[accs] 初始化失败: {e}", flush=True)
            raise

        self._stop.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="accs-transcriber", daemon=True)
        self._worker.start()

        # 通知 Web 端: 模型已就绪,开始采集
        if self.on_ready:
            try:
                self.on_ready()
            except Exception:
                pass

        try:
            self._consume()
        finally:
            self._stop.set()

        # 等待后台转录线程把队列里剩余的片段处理完(Queue.join 无 timeout 参数)
        self._work.join()
        self._worker.join(timeout=5)

        print("\n== 完成 ==", flush=True)
        if self.config.output.mode.lower() != "terminal":
            print("  已写出(共 %d 段):" % len(self.output.segments), flush=True)
            for s in self.output.segments:
                print(f"    \"{s.text}\"", flush=True)
        self.source.close()
        self.transcriber.close()

    def stop(self) -> None:
        self._stop.set()
        try:
            self.source.close()
        except Exception:
            pass
