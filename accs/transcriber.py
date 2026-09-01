from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import TranscriberConfig

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SENSEVOICE_DIR = os.path.join(
    _PROJECT_ROOT, "models", "csukuangfj", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
)
_TRANSDUCER_DIR = os.path.join(
    _PROJECT_ROOT, "models", "csukuangfj",
    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
)


@dataclass
class Word:
    start: float
    end: float
    word: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: Optional[List[Word]] = None


@dataclass
class TranscriptionResult:
    text: str
    language: str = ""
    language_probability: float = 0.0
    segments: List[Segment] = field(default_factory=list)


class TranscriberError(RuntimeError):
    pass


class Transcriber:
    """转录引擎基类。"""

    def __init__(self, cfg: TranscriberConfig):
        self.cfg = cfg
        self._loaded = False

    def load(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        pass

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class FasterWhisperTranscriber(Transcriber):
    """基于 faster-whisper(CTranslate2)。实时性好、支持 int8/GPU 混合精度。"""

    def _resolve_device(self) -> str:
        device = self.cfg.device
        if device != "auto":
            return device
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _resolve_compute_type(self, device: str) -> str:
        ct = self.cfg.compute_type
        if ct != "auto":
            return ct
        return "float16" if device == "cuda" else "int8"

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise TranscriberError(
                "未安装 faster-whisper。请 `pip install faster-whisper`。"
            ) from e

        device = self._resolve_device()
        compute_type = self._resolve_compute_type(device)
        model_path = self.cfg.model
        attempts = [(device, compute_type)]
        if device == "cuda":
            # GPU 需要 cuDNN;若失败自动回退 CPU
            attempts.append(("cpu", "int8"))

        last_exc = None
        for dev, ct in attempts:
            try:
                self._model = WhisperModel(model_path, device=dev, compute_type=ct)
                self._device, self._compute_type = dev, ct
                self._loaded = True
                return
            except Exception as e:  # pragma: no cover
                last_exc = e
        raise TranscriberError(f"加载 Whisper 模型失败({model_path}): {last_exc}。"
                               "若使用 GPU 请确认已安装 CUDA + cuDNN;否则改用 device=cpu、compute_type=int8。") from last_exc

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self._loaded:
            self.load()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return TranscriptionResult(text="")
        lang = self.cfg.language or None
        segments_iter, info = self._model.transcribe(
            audio,
            language=lang,
            beam_size=self.cfg.beam_size,
            word_timestamps=self.cfg.word_timestamps,
            vad_filter=self.cfg.vad_filter,
            condition_on_previous_text=False,
        )
        text_parts: List[str] = []
        segs: List[Segment] = []
        for s in segments_iter:
            text = (s.text or "").strip()
            if not text:
                continue
            text_parts.append(text)
            words = None
            if getattr(s, "words", None):
                words = [
                    Word(start=w.start, end=w.end, word=w.word)
                    for w in s.words if getattr(w, "word", None)
                ]
            segs.append(Segment(start=s.start, end=s.end, text=text, words=words))
        return TranscriptionResult(
            text=" ".join(text_parts).strip(),
            language=getattr(info, "language", "") or "",
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            segments=segs,
        )


class OpenAIWhisperTranscriber(Transcriber):
    """基于 openai-whisper(PyTorch)。复用已安装的 torch(gpu 加速)。"""

    def _resolve_device(self) -> str:
        device = self.cfg.device
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def load(self) -> None:
        try:
            import whisper
        except ImportError as e:
            raise TranscriberError("未安装 openai-whisper。请 `pip install openai-whisper`。") from e
        device = self._resolve_device()
        self._whisper = whisper.load_model(self.cfg.model, device=device)
        self._device = device
        self._loaded = True

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self._loaded:
            self.load()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return TranscriptionResult(text="")
        use_fp16 = self._device == "cuda"
        result = self._whisper.transcribe(
            audio,
            language=(self.cfg.language or None),
            beam_size=self.cfg.beam_size,
            word_timestamps=self.cfg.word_timestamps,
            fp16=use_fp16,
        )
        segs: List[Segment] = []
        text_parts: List[str] = []
        for s in result.get("segments", []):
            text = (s.get("text") or "").strip()
            if not text:
                continue
            text_parts.append(text)
            words = None
            if self.cfg.word_timestamps and s.get("words"):
                words = [
                    Word(start=w.get("start", 0.0), end=w.get("end", 0.0), word=w.get("word", ""))
                    for w in s["words"] if w.get("word")
                ]
            segs.append(Segment(start=s.get("start", 0.0), end=s.get("end", 0.0), text=text, words=words))
        return TranscriptionResult(
            text=" ".join(text_parts).strip(),
            language=result.get("language", "") or "",
            language_probability=float(result.get("language_probability", 0.0) or 0.0),
            segments=segs,
        )


class FunASRTranscriber(Transcriber):
    """基于阿里 FunASR 的中文/多语识别引擎(paraformer-zh / SenseVoice 等)。

    专门为中文训练,带标点、效果好于同尺寸 Whisper,适合"真正的中文识别"。
    """

    def __init__(self, cfg: TranscriberConfig):
        super().__init__(cfg)
        # 使用配置里的 funasr 模型名(也可由 model 字段指定)
        self._funasr_name = cfg.funasr_model or cfg.model or "paraformer-zh"

    def _resolve_device(self) -> str:
        device = self.cfg.device
        if device != "auto":
            return "cuda:0" if device == "cuda" else device
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def load(self) -> None:
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise TranscriberError(
                "未安装 funasr。请 `pip install funasr`(阿里达摩院中文识别)。"
            ) from e
        device = self._resolve_device()
        try:
            self._model = AutoModel(
                model=self._funasr_name,
                device=device,
                disable_update=True,
                trust_remote_code=True,
            )
        except Exception as e:
            raise TranscriberError(f"加载 FunASR 模型 {self._funasr_name} 失败: {e}") from e
        self._device = device
        self._loaded = True

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self._loaded:
            self.load()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return TranscriptionResult(text="")
        # use_itn=True 启用数字/标点归一化,段落级识别
        try:
            res = self._model.generate(input=audio, batch_size_s=300, use_itn=True)
        except Exception as e:
            raise TranscriberError(f"FunASR 识别失败: {e}") from e
        text = ""
        if res and isinstance(res, list) and res and isinstance(res[0], dict):
            text = (res[0].get("text") or "").strip()
        elif res:
            text = str(res[0]).strip()
        return TranscriptionResult(text=text, language="zh", segments=[])


class SherpaOnnxTranscriber(Transcriber):
    """基于 sherpa-onnx 的本地多语识别(中文最佳)。

    优先使用 SenseVoice(zh/en/ja/ko/yue,带标点,双语混合效果好);
    若目录下是 streaming zipformer(encoder/decoder/joiner),则自动按 transducer 识别。
    """

    def __init__(self, cfg: TranscriberConfig):
        super().__init__(cfg)
        self._dir = cfg.sherpa_onnx_model or ""  # 用户指定目录,空=自动探测
        self._ktype = (cfg.sherpa_onnx_type or "auto").lower()

    def _detect_dir(self) -> str:
        if self._dir and os.path.isdir(self._dir):
            return self._dir
        # 自动探测: 先 SenseVoice,再 streaming 双语
        for d in (_SENSEVOICE_DIR, _TRANSDUCER_DIR):
            if os.path.isdir(d):
                return d
        raise TranscriberError(
            "找不到 sherpa-onnx 模型目录。请先下载模型到 models/ 下,"
            "或在 config 的 transcriber.sherpa_onnx_model 指定目录。"
        )

    def _find(self, d: str, name: str) -> Optional[str]:
        p = os.path.join(d, name)
        return p if os.path.exists(p) else None

    def load(self) -> None:
        try:
            from sherpa_onnx import OfflineRecognizer, OnlineRecognizer
        except ImportError as e:
            raise TranscriberError("未安装 sherpa-onnx。请 `pip install sherpa-onnx`。") from e

        d = self._detect_dir()
        threads = 4
        use_sense = self._ktype in ("auto", "sense_voice")
        # SenseVoice: model.onnx / model.int8.onnx + tokens.txt
        model = None
        if use_sense:
            model = (self._find(d, "model.int8.onnx") or self._find(d, "model.onnx"))
            tokens = self._find(d, "tokens.txt")
            if model and tokens:
                self._rec = OfflineRecognizer.from_sense_voice(
                    tokens=tokens, model=model, num_threads=threads,
                    provider="cpu", language="auto", use_itn=True,
                )
                self._online = False
                self._loaded = True
                self._dir = d
                return

        # streaming zipformer 双语 (encoder/decoder/joiner)
        enc, dec, joi, tokens = (
            self._find(d, "encoder-epoch-99-avg-1.int8.onnx") or self._find(d, "encoder-epoch-99-avg-1.onnx"),
            self._find(d, "decoder-epoch-99-avg-1.int8.onnx") or self._find(d, "decoder-epoch-99-avg-1.onnx"),
            self._find(d, "joiner-epoch-99-avg-1.int8.onnx") or self._find(d, "joiner-epoch-99-avg-1.onnx"),
            self._find(d, "tokens.txt"),
        )
        if not (enc and dec and joi and tokens):
            raise TranscriberError(f"{d} 下没找到 SenseVoice 或 transducer 模型文件。")
        self._rec = OnlineRecognizer.from_transducer(
            tokens=tokens, encoder=enc, decoder=dec, joiner=joi,
            num_threads=threads, decoding_method="modified_beam_search",
            enable_endpoint_detection=True,
        )
        self._online = True
        self._loaded = True
        self._dir = d

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self._loaded:
            self.load()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return TranscriptionResult(text="")
        if self._online:
            stream = self._rec.create_stream()
            chunk = int(0.4 * 16000)
            for i in range(0, audio.size, chunk):
                stream.accept_waveform(16000, audio[i:i + chunk])
                while self._rec.is_ready(stream):
                    self._rec.decode_stream(stream)
            tail = np.zeros(int(0.6 * 16000), dtype=np.float32)
            stream.accept_waveform(16000, tail)
            stream.input_finished()
            while self._rec.is_ready(stream):
                self._rec.decode_stream(stream)
            text = self._rec.get_result_all(stream).text
        else:
            stream = self._rec.create_stream()
            stream.accept_waveform(16000, audio)
            self._rec.decode_stream(stream)
            text = stream.result.text
        return TranscriptionResult(text=(text or "").strip(), language="", segments=[])


def create_transcriber(cfg: TranscriberConfig) -> Transcriber:
    engine = (cfg.engine or "faster_whisper").lower()
    if engine in ("sherpa_onnx", "sherpa-onnx", "sherpa"):
        return SherpaOnnxTranscriber(cfg)
    if engine in ("faster_whisper", "faster-whisper"):
        return FasterWhisperTranscriber(cfg)
    if engine in ("openai_whisper", "openai-whisper", "openai"):
        return OpenAIWhisperTranscriber(cfg)
    if engine in ("funasr", "fun_asr"):
        return FunASRTranscriber(cfg)
    raise TranscriberError(f"未知转录引擎: {cfg.engine}")
