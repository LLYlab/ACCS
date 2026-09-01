from __future__ import annotations

from typing import Optional

import numpy as np

from .config import VadConfig


class VADError(RuntimeError):
    pass


class BaseVAD:
    """VAD 基类。`probability(samples)` 返回 [0,1] 的说话概率。"""

    def load(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def probability(self, samples: np.ndarray) -> float:
        raise NotImplementedError


class SileroVAD(BaseVAD):
    """Silero VAD(通过 faster-whisper 封装的 ONNX 模型)。"""

    def __init__(self, cfg: VadConfig):
        self.cfg = cfg
        self._model = None

    def load(self) -> None:
        try:
            from faster_whisper.vad import get_vad_model
        except Exception as e:
            raise VADError(
                "无法加载 faster-whisper 的 Silero VAD。请确认已安装 faster-whisper"
                f"(它将自动获取 ONNX 模型)。详细原因: {e}"
            ) from e
        self._model = get_vad_model()

    def probability(self, samples: np.ndarray) -> float:
        if self._model is None:
            self.load()
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return 0.0
        # Silero 需要输入长度是 512 的整数倍;不足的尾部忽略
        num_samples = 512
        trim = (arr.size // num_samples) * num_samples
        if trim < num_samples:
            return 0.0
        try:
            probs = np.asarray(
                self._model(arr[:trim], num_samples=num_samples, context_size_samples=64),
                dtype=np.float32,
            ).reshape(-1)
        except Exception:
            return 0.0
        if probs.size == 0:
            return 0.0
        # 取窗口概率的最大值,代表"这一帧里有人说话"
        return float(probs.max())


class EnergyVAD(BaseVAD):
    """轻量能量阈值 VAD(Silero 不可用时的兜底)。"""

    def __init__(self, cfg: VadConfig):
        self.cfg = cfg
        self._noise_floor = 0.02

    def load(self) -> None:
        pass

    def probability(self, samples: np.ndarray) -> float:
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(arr ** 2)))
        # 把 RMS 映射到 [0,1] 的粗略概率
        prob = min(1.0, max(0.0, (rms - self._noise_floor) / (0.2 - self._noise_floor)))
        return prob


def create_vad(cfg: VadConfig) -> BaseVAD:
    """优先 Silero,失败回退能量 VAD。"""
    try:
        vad = SileroVAD(cfg)
        vad.load()
        return vad
    except Exception:
        vad = EnergyVAD(cfg)
        vad.load()
        return vad
