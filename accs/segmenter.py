"""分段器：把连续音频帧按 VAD 概率切成"一句话"。

设计目标（相对旧 pipeline 的改进）
1. **O(1) 追加**：缓冲用帧列表，定稿时才 concat 一次，不再逐帧 `np.concatenate`（旧版长片段是 O(n²)）。
2. **双阈值 + 滞后**：开始用高阈值、结束用低阈值，避免一句话被 VAD 抖动切成碎片。
3. **三级切分**：静音优先；超长时在"能量谷"切，不再 30s 硬切在词中间。
4. **可配置重叠**：相邻片段可保留一段重叠，防止边界字被截断。
5. **纯逻辑**：不碰音频硬件，`push(frame, prob)` 进、`SegmentChunk` 出，可直接单测。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import VadConfig


@dataclass
class SegmentChunk:
    """一段定稿音频（16k 单声道 float32）。"""
    audio: np.ndarray
    start: float          # 相对采集开始的秒数
    end: float
    index: int = 0
    reason: str = "silence"   # silence | max_length | flush


class Segmenter:
    """流式分段器。

    用法::

        seg = Segmenter(cfg.vad, sample_rate=16000)
        for frame, prob in stream:
            for chunk in seg.push(frame, prob):
                handle(chunk)
        for chunk in seg.flush():
            handle(chunk)
    """

    def __init__(self, cfg: VadConfig, sample_rate: int = 16000):
        self.cfg = cfg
        self.sr = int(sample_rate)

        self.start_thr = float(cfg.threshold)
        end_ratio = float(getattr(cfg, "end_ratio", 0.6) or 0.6)
        self.end_thr = self.start_thr * max(0.1, min(1.0, end_ratio))

        self.prepend_n = int(self.sr * cfg.prepend_ms / 1000)
        self.pad_n = int(self.sr * cfg.pad_ms / 1000)
        self.silence_n = int(self.sr * cfg.silence_ms / 1000)
        self.min_speech_n = int(self.sr * cfg.min_speech_ms / 1000)
        self.max_n = int(self.sr * cfg.max_segment_ms / 1000)
        self.overlap_n = int(self.sr * (getattr(cfg, "overlap_ms", 0) or 0) / 1000)

        # 状态
        self._frames: List[np.ndarray] = []      # 当前片段
        self._energy: List[float] = []            # 与 _frames 对齐的 RMS
        self._preroll: List[np.ndarray] = []      # 静音期滚动缓冲（用作前置补音）
        self._preroll_n = 0
        self._total = 0                           # 已喂入的总样本数
        self._active = False
        self._silence = 0
        self._cur_n = 0
        self._start_sample = 0
        self._counter = 0

    # ------------------------------------------------------------------ 状态
    @property
    def stats(self) -> dict:
        return {
            "active": self._active,
            "buffered_ms": round(self._cur_n / self.sr * 1000),
            "silence_ms": round(self._silence / self.sr * 1000),
            "segments": self._counter,
        }

    # ------------------------------------------------------------------ 入口
    def push(self, frame: np.ndarray, prob: float) -> List[SegmentChunk]:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.size == 0:
            return []
        n = frame.size
        self._total += n
        prob = float(prob)

        out: List[SegmentChunk] = []
        if not self._active:
            if prob >= self.start_thr:
                self._begin(frame)
            else:
                self._push_preroll(frame)
            return out

        # 说话中
        self._append(frame)
        if prob >= self.end_thr:
            self._silence = 0
        else:
            self._silence += n

        if self._silence >= self.silence_n:
            chunk = self._finalize(reason="silence", keep_tail_pad=True)
            if chunk:
                out.append(chunk)
        elif self._cur_n >= self.max_n:
            # 超长：在能量谷切，保留后半段继续
            chunk, rest = self._split_at_valley()
            if chunk:
                out.append(chunk)
            if not rest:
                # 切不出来（极少见）：整段定稿
                chunk2 = self._finalize(reason="max_length", keep_tail_pad=False)
                if chunk2:
                    out.append(chunk2)
        return out

    def flush(self) -> List[SegmentChunk]:
        """结束采集：把缓冲里剩下的一并定稿。"""
        if not self._active and not self._frames:
            return []
        chunk = self._finalize(reason="flush", keep_tail_pad=False)
        return [chunk] if chunk else []

    # ------------------------------------------------------------------ 内部
    def _begin(self, frame: np.ndarray) -> None:
        preroll = self._preroll
        pre_n = sum(f.size for f in preroll)
        self._frames = list(preroll) + [frame]
        self._energy = [self._rms(f) for f in self._frames]
        self._cur_n = pre_n + frame.size
        self._start_sample = self._total - self._cur_n
        self._active = True
        self._silence = 0
        self._preroll = []
        self._preroll_n = 0

    def _append(self, frame: np.ndarray) -> None:
        self._frames.append(frame)
        self._energy.append(self._rms(frame))
        self._cur_n += frame.size

    def _push_preroll(self, frame: np.ndarray) -> None:
        self._preroll.append(frame)
        self._preroll_n += frame.size
        # 只保留 prepend_ms 长度
        while self._preroll and self._preroll_n - self._preroll[0].size >= self.prepend_n:
            self._preroll_n -= self._preroll[0].size
            self._preroll.pop(0)

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        if frame.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))

    def _concat(self, frames: List[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.zeros(0, dtype=np.float32)
        if len(frames) == 1:
            return frames[0].copy()
        return np.concatenate(frames)

    def _finalize(self, reason: str, keep_tail_pad: bool) -> Optional[SegmentChunk]:
        if not self._frames:
            self._reset()
            return None

        frames = self._frames
        if keep_tail_pad:
            # 静音已超过阈值：只保留 pad 长度的尾巴
            keep_n = self._cur_n - max(0, self._silence - self.pad_n)
            frames = self._trim(frames, max(0, keep_n))
        audio = self._concat(frames)
        start_s = self._start_sample / self.sr
        end_s = start_s + audio.size / self.sr

        chunk = None
        if audio.size >= self.min_speech_n:
            chunk = SegmentChunk(audio=audio, start=start_s, end=end_s,
                                 index=self._counter, reason=reason)
            self._counter += 1
        # 保留一点尾巴做下一段的前置，避免切掉下一个字
        tail = self._tail_frames(frames, self.prepend_n)
        self._frames = []
        self._energy = []
        self._cur_n = 0
        self._active = False
        self._silence = 0
        self._preroll = tail
        self._preroll_n = sum(f.size for f in tail)
        return chunk

    def _split_at_valley(self):
        """在缓冲区后 40% 里找能量最低的位置切开，前半定稿、后半继续。"""
        frames, energy = self._frames, self._energy
        if len(frames) < 4:
            return None, False

        # 从尾部往前找：至少有 1s 的余量，且不早于一半
        min_keep = int(self.sr * 0.5)
        acc = 0
        lo = len(frames) - 1
        while lo > 0 and acc < min_keep:
            acc += frames[lo].size
            lo -= 1
        search_lo = max(1, int(len(frames) * 0.5))
        if lo <= search_lo:
            lo = search_lo

        idx = min(range(search_lo, max(search_lo + 1, lo)), key=lambda i: energy[i])
        head = frames[:idx]
        tail = frames[idx:]
        if not head or not tail:
            return None, False

        audio = self._concat(head)
        start_s = self._start_sample / self.sr
        chunk = None
        if audio.size >= self.min_speech_n:
            chunk = SegmentChunk(audio=audio, start=start_s,
                                 end=start_s + audio.size / self.sr,
                                 index=self._counter, reason="max_length")
            self._counter += 1

        # 后半段继续（可带 overlap）
        carry = self._tail_frames(head, self.overlap_n) if self.overlap_n else []
        self._frames = list(carry) + tail
        self._energy = [self._rms(f) for f in self._frames]
        self._cur_n = sum(f.size for f in self._frames)
        self._start_sample = self._total - self._cur_n
        self._silence = 0
        return chunk, True

    def _trim(self, frames: List[np.ndarray], keep_n: int) -> List[np.ndarray]:
        if keep_n <= 0:
            return []
        out: List[np.ndarray] = []
        left = keep_n
        for f in frames:
            if left <= 0:
                break
            if f.size <= left:
                out.append(f)
                left -= f.size
            else:
                out.append(f[:left])
                left = 0
        return out

    def _tail_frames(self, frames: List[np.ndarray], n: int) -> List[np.ndarray]:
        """取最后 n 个样本（按整帧近似）。"""
        if n <= 0:
            return []
        out: List[np.ndarray] = []
        acc = 0
        for f in reversed(frames):
            out.insert(0, f)
            acc += f.size
            if acc >= n:
                break
        return out

    def _reset(self) -> None:
        self._frames = []
        self._energy = []
        self._preroll = []
        self._preroll_n = 0
        self._cur_n = 0
        self._active = False
        self._silence = 0
