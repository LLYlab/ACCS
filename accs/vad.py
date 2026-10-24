from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .config import VadConfig


class VADError(RuntimeError):
    pass


class BaseVAD:
    """VAD 基类。`probability(samples)` 返回 [0,1] 的说话概率。"""

    name = "base"

    def load(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def probability(self, samples: np.ndarray) -> float:
        raise NotImplementedError


class SileroVAD(BaseVAD):
    """Silero VAD（通过 faster-whisper 封装的 ONNX 模型）。"""

    name = "silero"

    def __init__(self, cfg: VadConfig):
        self.cfg = cfg
        self._model = None

    def load(self) -> None:
        try:
            from faster_whisper.vad import get_vad_model
        except Exception as e:
            raise VADError(
                "无法加载 faster-whisper 的 Silero VAD。请确认已安装 faster-whisper"
                f"（它会自动获取 ONNX 模型）。详细原因: {e}"
            ) from e
        self._model = get_vad_model()
        self._self_test()

    def _self_test(self) -> None:
        """自检：合成一段音调，确认 ONNX 推理真的能跑（打包后常见 DLL/资源缺失）。"""
        try:
            t = np.arange(512 * 8, dtype=np.float32) / 16000.0
            tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            probs = np.asarray(
                self._model(tone, num_samples=512, context_size_samples=64),
                dtype=np.float32,
            ).reshape(-1)
            if probs.size == 0:
                raise RuntimeError("模型返回空概率")
        except Exception as e:
            self._model = None
            raise VADError(f"Silero VAD 自检失败: {e}") from e

    def probability(self, samples: np.ndarray) -> float:
        if self._model is None:
            self.load()
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return 0.0
        num_samples = 512
        trim = (arr.size // num_samples) * num_samples
        if trim < num_samples:
            return 0.0
        probs = np.asarray(
            self._model(arr[:trim], num_samples=num_samples, context_size_samples=64),
            dtype=np.float32,
        ).reshape(-1)
        if probs.size == 0:
            return 0.0
        return float(probs.max())


class EnergyVAD(BaseVAD):
    """轻量能量 VAD（Silero 不可用时的兜底）。

    噪声底取最近 10 秒的 20% 分位，再按"当前 RMS 是噪声的多少倍"映射到概率，
    比固定阈值稳得多（固定阈值在麦克风增益不同时基本不可用）。
    """

    name = "energy"

    def __init__(self, cfg: VadConfig):
        self.cfg = cfg
        self._hist: deque = deque(maxlen=100)

    def load(self) -> None:
        pass

    def probability(self, samples: np.ndarray) -> float:
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(arr ** 2)))
        self._hist.append(rms)

        if len(self._hist) >= 8:
            noise = float(np.percentile(np.asarray(self._hist), 20))
        else:
            noise = float(min(self._hist))
        noise = max(noise, 1e-5)

        ratio = rms / noise
        # 1.6 倍以下视为静音，4.6 倍以上视为确定说话
        prob = (ratio - 1.6) / 3.0
        return float(min(1.0, max(0.0, prob)))


def create_vad(cfg: VadConfig) -> BaseVAD:
    """优先 Silero（含自检），失败回退能量 VAD。"""
    try:
        vad = SileroVAD(cfg)
        vad.load()
        return vad
    except Exception as e:
        print(f"[accs] Silero VAD 不可用（{e}），回退能量 VAD", flush=True)
        vad = EnergyVAD(cfg)
        vad.load()
        return vad
