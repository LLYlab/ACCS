from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import yaml

# 项目根目录(= config.yaml 所在目录)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.environ.get(
    "ACCS_CONFIG", os.path.join(PROJECT_ROOT, "config.yaml")
)


@dataclass
class AudioConfig:
    source: str = "system"          # system | mic | file
    backend: str = "auto"           # auto | soundcard | sounddevice
    device: str = ""                # keyword for device selection (both)
    mic_device: str = ""            # keyword for mic
    system_device: str = ""         # keyword for system loopback
    file: str = ""                  # path for source=file
    sample_rate: int = 16000
    frame_ms: int = 256
    channels: int = 1

    @property
    def frame_size(self) -> int:
        """每帧采样点数。"""
        return int(self.sample_rate * self.frame_ms / 1000)


@dataclass
class VadConfig:
    enabled: bool = True
    threshold: float = 0.5
    min_speech_ms: int = 500
    prepend_ms: int = 300
    pad_ms: int = 200
    silence_ms: int = 800
    max_segment_ms: int = 30000


@dataclass
class TranscriberConfig:
    engine: str = "sherpa_onnx"     # sherpa_onnx | faster_whisper | openai_whisper | funasr
    model: str = "small"            # tiny/base/small/medium/large-v3 或本地路径
    funasr_model: str = "paraformer-zh"  # FUNASR 中文模型: paraformer-zh / SenseVoiceSmall ...
    sherpa_onnx_model: str = ""     # sherpa-onnx 模型目录(空=自动探测内置 models/)
    sherpa_onnx_type: str = "auto"  # auto | sense_voice | transducer
    device: str = "auto"            # auto | cpu | cuda
    compute_type: str = "auto"      # auto | int8 | float16 | int8_float16 | float32
    language: str = ""              # "" = auto
    beam_size: int = 5
    word_timestamps: bool = True
    vad_filter: bool = False


@dataclass
class OutputConfig:
    mode: str = "terminal"          # terminal | file | srt | jsonl | all
    base_dir: str = "output"
    file: str = ""
    show_transcript_live: bool = True
    show_language: bool = True


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    transcriber: TranscriberConfig = field(default_factory=TranscriberConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls, path: str = DEFAULT_CONFIG_PATH, overrides: Optional[Dict[str, Any]] = None) -> "Config":
        raw: Dict[str, Any] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        raw = raw or {}

        def _mk(section_key, datacls):
            section = raw.get(section_key, {}) or {}
            return datacls(**{k: v for k, v in section.items() if k in datacls.__dataclass_fields__})

        cfg = cls(
            audio=_mk("audio", AudioConfig),
            vad=_mk("vad", VadConfig),
            transcriber=_mk("transcriber", TranscriberConfig),
            output=_mk("output", OutputConfig),
        )

        # 命令行覆盖: 支持点分键, 如 {"audio.source": "file", "transcriber.model": "base"}
        if overrides:
            for key, value in overrides.items():
                cfg._set(key, value)
        return cfg

    def _set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        obj = self
        for part in parts[:-1]:
            if part in ("audio", "vad", "transcriber", "output"):
                obj = getattr(obj, part)
            else:
                return
        setattr(obj, parts[-1], value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
