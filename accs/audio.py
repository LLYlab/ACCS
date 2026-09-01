from __future__ import annotations

import threading
from typing import Iterator, Optional

import numpy as np

from .config import AudioConfig


class AudioError(RuntimeError):
    """音频采集相关错误。"""


class AudioSource:
    """音频源基类。

    约定: `stream()` 返回一个迭代器,每次产出 float32 单声道采样点数组,
    长度不固定(近似 frame_ms 对应的采样数),采样率为 cfg.sample_rate(16k)。

    在停止时,迭代器应自然结束;也可通过 `close()` 请求提前停止。
    """

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self._closed = False

    @property
    def sample_rate(self) -> int:
        return self.cfg.sample_rate

    def open(self) -> None:  # pragma: no cover - 具体后端实现
        pass

    def close(self) -> None:
        self._closed = True

    def stream(self) -> Iterator[np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    def __enter__(self) -> "AudioSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _match_by_keyword(devices, keyword: str):
    """按名称/ID 子串匹配设备,大小写不敏感。支持"名称子串"或精确 id。"""
    if not keyword:
        return None
    kw = keyword.lower()
    for d in devices:
        name = (getattr(d, "name", "") or "").lower()
        did = str(getattr(d, "id", "") or "")
        if kw in name or kw == did:
            return d
    return None


class SoundcardSource(AudioSource):
    """基于 soundcard(WASAPI/环回)采集。Windows 上可抓取系统播放声音。"""

    def _load(self):
        try:
            import soundcard as sc
        except ImportError as e:  # pragma: no cover
            raise AudioError(
                "未安装 soundcard。请先 `pip install soundcard`。"
            ) from e
        return sc

    def _select_mic(self, sc):
        mics = sc.all_microphones(include_loopback=True)

        def _is_loopback(m):
            return bool(getattr(m, "isloopback", False)) or "loopback" in (m.name or "").lower()

        if self.cfg.source == "system":
            # 系统环回: 依据 isloopback 属性(名称不一定含 loopback)
            loopbacks = [m for m in mics if _is_loopback(m)]
            if not loopbacks:
                raise AudioError(
                    "未找到系统环回(loopback)设备。请确认系统正在播放音频(如网课),"
                    "并检查 soundcard 是否已识别到输出设备。可在 settings 里开启"
                    "“允许应用独占音频”。可用 `python -m accs --list-devices` 查看设备。"
                )
            kw = self.cfg.system_device or self.cfg.device
            return _match_by_keyword(loopbacks, kw) or loopbacks[0]
        # 麦克风
        normal = [m for m in mics if not _is_loopback(m)]
        kw = self.cfg.mic_device or self.cfg.device
        chosen = _match_by_keyword(normal, kw)
        if chosen is not None:
            return chosen
        try:
            return sc.default_microphone()
        except Exception as e:  # pragma: no cover
            raise AudioError(f"未找到可用麦克风: {e}") from e

    def open(self) -> None:
        self._sc = self._load()
        self._mic = self._select_mic(self._sc)

    def stream(self) -> Iterator[np.ndarray]:
        sr = self.cfg.sample_rate
        block = self.cfg.frame_size
        # 若要环回,通常需单声道;保持 channels=1 到后端,减少混音损耗
        with self._mic.recorder(samplerate=sr, channels=1) as rec:
            while not self._closed:
                try:
                    data = rec.record(numframes=block)
                except Exception as e:  # pragma: no cover
                    if self._closed:
                        break
                    raise AudioError(f"soundcard 录音失败: {e}") from e
                if data is None or len(data) == 0:
                    continue
                arr = np.asarray(data, dtype=np.float32)
                if arr.ndim == 2:
                    mono = arr[:, 0]
                else:
                    mono = arr.reshape(-1)
                yield mono


class SounddeviceSource(AudioSource):
    """基于 sounddevice(PortAudio)采集,主要用于麦克风。"""

    def _load(self):
        try:
            import sounddevice as sd
        except ImportError as e:  # pragma: no cover
            raise AudioError("未安装 sounddevice。请先 `pip install sounddevice`。") from e
        return sd

    def _device_index(self, sd):
        kw = self.cfg.device or self.cfg.mic_device
        try:
            if kw:
                devices = sd.query_devices()
                for i, d in enumerate(devices):
                    if kw.lower() in (str(d["name"]).lower()) or str(i) == kw:
                        return i
            return None  # 使用默认输入设备
        except Exception:  # pragma: no cover
            return None

    def open(self) -> None:
        self._sd = self._load()

    def stream(self) -> Iterator[np.ndarray]:
        import queue
        import sounddevice as sd

        sr = self.cfg.sample_rate
        block = self.cfg.frame_size
        q: "queue.Queue[np.ndarray]" = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                # 若只在静默/过载时出现的状态,这里仅提示,不中断
                if status.input_overflow:
                    pass
            q.put(np.asarray(indata[:, 0], dtype=np.float32).copy())

        stream = sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=block,
            callback=callback,
            device=self._device_index(sd),
        )
        with stream:
            while not self._closed:
                try:
                    block_arr = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                yield block_arr


class FileSource(AudioSource):
    """从本地音频文件读取(借助 faster-whisper 的 ffmpeg 解码),用于离线回放/测试。"""

    def open(self) -> None:
        if not self.cfg.file:
            raise AudioError("source=file 时必须在 audio.file 里指定音频文件路径。")
        from faster_whisper.audio import decode_audio

        self._audio = decode_audio(self.cfg.file)  # float32, 16k, mono
        if self._audio.ndim != 1:
            self._audio = self._audio.mean(axis=1)
        self._audio = np.asarray(self._audio, dtype=np.float32)
        if self._audio.size == 0:
            raise AudioError(f"文件未解码到音频数据: {self.cfg.file}")

    def stream(self) -> Iterator[np.ndarray]:
        block = self.cfg.frame_size
        i = 0
        n = self._audio.size
        while i < n:
            if self._closed:
                break
            yield self._audio[i:i + block]
            i += block


def create_audio_source(cfg: AudioConfig) -> AudioSource:
    """根据配置选择合适的音频源。"""
    backend = cfg.backend
    if cfg.source == "file":
        return FileSource(cfg)

    # 后端优先级: 显式 backend -> auto 探测
    candidates = []
    if backend == "soundcard":
        candidates = [SoundcardSource]
    elif backend == "sounddevice":
        candidates = [SounddeviceSource]
    else:  # auto
        # system(环回) 依赖 soundcard; mic 两者皆可
        if cfg.source == "system":
            candidates = [SoundcardSource]
        else:
            candidates = [SoundcardSource, SounddeviceSource]

    last_err: Optional[Exception] = None
    for cls in candidates:
        try:
            src = cls(cfg)
            src.open()  # 提前探测,失败则回退到下一个
            return src
        except Exception as e:
            last_err = e
            continue

    raise AudioError(
        f"无法创建音频源(source={cfg.source}, backend={cfg.backend})。"
        f"可能缺少音频后端: {last_err}"
    ) from last_err


def list_devices() -> str:
    """返回可用的音频设备清单(诊断用)。"""
    lines = []
    try:
        import soundcard as sc

        lines.append("== soundcard 设备 ==")
        for m in sc.all_microphones(include_loopback=True):
            tag = " (loopback)" if "loopback" in (m.name or "").lower() else ""
            lines.append(f"  - {m.name}{tag} | id={m.id}")
    except Exception as e:  # pragma: no cover
        lines.append(f"  soundcard 不可用: {e}")
    try:
        import sounddevice as sd

        lines.append("== sounddevice 设备 ==")
        for i, d in enumerate(sd.query_devices()):
            lines.append(f"  - [{i}] {d['name']} (in={d['max_input_channels']}, out={d['max_output_channels']})")
    except Exception as e:  # pragma: no cover
        lines.append(f"  sounddevice 不可用: {e}")
    return "\n".join(lines)
