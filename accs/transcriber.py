"""转录引擎层：统一接口 + 能力表 + 本地离线实现。

支持的引擎
  sherpa_onnx     SenseVoice / Paraformer / 流式 Zipformer（中文最佳，带标点）
  faster_whisper  CTranslate2 版 Whisper（实时性好）
  openai_whisper  原版 Whisper（复用已装 PyTorch）
  funasr          阿里 FunASR（paraformer-zh / SenseVoice）
  vosk            Vosk 轻量离线（小模型，CPU 友好）
  whisper_cpp     whisper.cpp（GGML，CPU/低配友好）
  browser         浏览器端识别（Web Speech API，由前端完成，后端只存结果）

所有引擎都返回同一种 `TranscriptionResult`，并声明 capabilities，
Web 端据此决定显示哪些选项、能否显示时间戳。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import TranscriberConfig
from .paths import app_root, models_dir

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_ROOT = models_dir()
_SENSEVOICE_DIR = os.path.join(
    _MODELS_ROOT, "csukuangfj", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
)
_TRANSDUCER_DIR = os.path.join(
    _MODELS_ROOT, "csukuangfj",
    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
)


# ============================================================ 数据结构
@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float = 0.0


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
    words: List[Word] = field(default_factory=list)
    confidence: float = 0.0
    engine: str = ""


class TranscriberError(RuntimeError):
    pass


# ============================================================ 基类
class Transcriber:
    """转录引擎基类。"""

    id = "base"
    label = "base"
    # 能力声明：Web 端据此禁用/显示对应控件
    capabilities: Dict[str, Any] = {
        "streaming": False,      # 能否边录边出
        "timestamps": False,     # 是否给词/段时间戳
        "punctuation": False,    # 输出是否自带标点
        "local": True,           # 是否完全本地
        "languages": [],         # 支持语言；空 = 自动/多语
    }

    def __init__(self, cfg: TranscriberConfig):
        self.cfg = cfg
        self._loaded = False

    def load(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        pass

    # 便于统一处理：把 Segment 里的词摊平成 words
    @staticmethod
    def _flatten(segments: List[Segment]) -> List[Word]:
        out: List[Word] = []
        for s in segments:
            if s.words:
                out.extend(s.words)
        return out


# ============================================================ faster-whisper
class FasterWhisperTranscriber(Transcriber):
    id = "faster_whisper"
    label = "faster-whisper（CTranslate2）"
    capabilities = {"streaming": False, "timestamps": True, "punctuation": True,
                    "local": True, "languages": []}

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
            raise TranscriberError("未安装 faster-whisper。请 `pip install faster-whisper`。") from e

        device = self._resolve_device()
        compute_type = self._resolve_compute_type(device)
        attempts = [(device, compute_type)]
        if device == "cuda":
            attempts.append(("cpu", "int8"))

        last_exc = None
        for dev, ct in attempts:
            try:
                self._model = WhisperModel(self.cfg.model, device=dev, compute_type=ct)
                self._device, self._compute_type = dev, ct
                self._loaded = True
                return
            except Exception as e:  # pragma: no cover
                last_exc = e
        raise TranscriberError(
            f"加载 Whisper 模型失败({self.cfg.model}): {last_exc}。"
            "若使用 GPU 请确认已安装 CUDA + cuDNN；否则改用 device=cpu、compute_type=int8。"
        ) from last_exc

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self._loaded:
            self.load()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return TranscriptionResult(text="", engine=self.id)
        segments_iter, info = self._model.transcribe(
            audio,
            language=(self.cfg.language or None),
            beam_size=self.cfg.beam_size,
            word_timestamps=self.cfg.word_timestamps,
            vad_filter=self.cfg.vad_filter,
            condition_on_previous_text=False,
        )
        segs: List[Segment] = []
        for s in segments_iter:
            text = (s.text or "").strip()
            if not text:
                continue
            words = None
            if getattr(s, "words", None):
                words = [Word(w.start, w.end, w.word, float(getattr(w, "probability", 0) or 0))
                         for w in s.words if getattr(w, "word", None)]
            segs.append(Segment(s.start, s.end, text, words))
        words = self._flatten(segs)
        conf = float(np.mean([w.probability for w in words])) if words else 0.0
        return TranscriptionResult(
            text=" ".join(s.text for s in segs).strip(),
            language=getattr(info, "language", "") or "",
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            segments=segs, words=words, confidence=conf, engine=self.id,
        )


# ============================================================ openai-whisper
class OpenAIWhisperTranscriber(Transcriber):
    id = "openai_whisper"
    label = "openai-whisper（PyTorch）"
    capabilities = {"streaming": False, "timestamps": True, "punctuation": True,
                    "local": True, "languages": []}

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
        self._device = self._resolve_device()
        self._whisper = whisper.load_model(self.cfg.model, device=self._device)
        self._loaded = True

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self._loaded:
            self.load()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return TranscriptionResult(text="", engine=self.id)
        result = self._whisper.transcribe(
            audio,
            language=(self.cfg.language or None),
            beam_size=self.cfg.beam_size,
            word_timestamps=self.cfg.word_timestamps,
            fp16=(self._device == "cuda"),
        )
        segs: List[Segment] = []
        for s in result.get("segments", []):
            text = (s.get("text") or "").strip()
            if not text:
                continue
            words = None
            if self.cfg.word_timestamps and s.get("words"):
                words = [Word(w.get("start", 0.0), w.get("end", 0.0), w.get("word", ""),
                              float(w.get("probability", 0) or 0)) for w in s["words"] if w.get("word")]
            segs.append(Segment(s.get("start", 0.0), s.get("end", 0.0), text, words))
        words = self._flatten(segs)
        conf = float(np.mean([w.probability for w in words])) if words else 0.0
        return TranscriptionResult(
            text=" ".join(s.text for s in segs).strip(),
            language=result.get("language", "") or "",
            segments=segs, words=words, confidence=conf, engine=self.id,
        )


# ============================================================ sherpa-onnx
class SherpaOnnxTranscriber(Transcriber):
    id = "sherpa_onnx"
    label = "sherpa-onnx（SenseVoice / Paraformer）"
    capabilities = {"streaming": False, "timestamps": False, "punctuation": True,
                    "local": True, "languages": ["zh", "en", "ja", "ko", "yue"]}

    def __init__(self, cfg: TranscriberConfig):
        super().__init__(cfg)
        self._dir = cfg.sherpa_onnx_model or ""
        self._ktype = (cfg.sherpa_onnx_type or "auto").lower()
        self._online = False
        self._kind = ""

    def _candidates(self) -> List[str]:
        if self._dir and os.path.isdir(self._dir):
            return [self._dir]
        import glob
        dirs = sorted(glob.glob(os.path.join(_MODELS_ROOT, "**", "sherpa-onnx-*"), recursive=True))
        return [d for d in dirs if os.path.isdir(d)]

    def _find(self, d: str, *names: str) -> Optional[str]:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
        return None

    def _detect(self) -> tuple:
        for d in self._candidates():
            model = self._find(d, "model.int8.onnx", "model.onnx")
            tokens = self._find(d, "tokens.txt")
            if model and tokens:
                # SenseVoice / Paraformer 都是离线模型，靠目录名区分
                low = d.lower()
                if "paraformer" in low:
                    return ("paraformer", d, model, tokens)
                return ("sense_voice", d, model, tokens)
            enc = self._find(d, "encoder-epoch-99-avg-1.int8.onnx", "encoder-epoch-99-avg-1.onnx")
            dec = self._find(d, "decoder-epoch-99-avg-1.int8.onnx", "decoder-epoch-99-avg-1.onnx")
            joi = self._find(d, "joiner-epoch-99-avg-1.int8.onnx", "joiner-epoch-99-avg-1.onnx")
            if enc and dec and joi and tokens:
                return ("transducer", d, enc, dec, joi, tokens)
        return (None,)

    def load(self) -> None:
        try:
            from sherpa_onnx import OfflineRecognizer, OnlineRecognizer
        except ImportError as e:
            raise TranscriberError("未安装 sherpa-onnx。请 `pip install sherpa-onnx`。") from e

        det = self._detect()
        kind = det[0]
        if kind is None:
            raise TranscriberError(
                "找不到 sherpa-onnx 模型。请把 SenseVoice / Paraformer / 流式 Zipformer 放到 models/ 下，"
                "或在配置里指定 transcriber.sherpa_onnx_model。"
            )

        lang = (self.cfg.language or "auto") or "auto"
        if kind in ("sense_voice", "paraformer"):
            _, d, model, tokens = det[:4]
            if kind == "paraformer":
                rec = OfflineRecognizer.from_paraformer(tokens=tokens, model=model, num_threads=4)
            else:
                rec = OfflineRecognizer.from_sense_voice(
                    tokens=tokens, model=model, num_threads=4,
                    provider="cpu", language=lang, use_itn=True,
                )
            self._rec, self._online, self._kind = rec, False, kind
        else:
            _, d, enc, dec, joi, tokens = det
            rec = OnlineRecognizer.from_transducer(
                tokens=tokens, encoder=enc, decoder=dec, joiner=joi,
                num_threads=4, decoding_method="modified_beam_search",
                enable_endpoint_detection=True,
            )
            self._rec, self._online, self._kind = rec, True, kind
        self._dir = d
        self._loaded = True
        self.capabilities = dict(self.capabilities, streaming=self._online)

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self._loaded:
            self.load()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return TranscriptionResult(text="", engine=self.id)
        if self._online:
            stream = self._rec.create_stream()
            chunk = int(0.4 * 16000)
            for i in range(0, audio.size, chunk):
                stream.accept_waveform(16000, audio[i:i + chunk])
                while self._rec.is_ready(stream):
                    self._rec.decode_stream(stream)
            stream.accept_waveform(16000, np.zeros(int(0.6 * 16000), dtype=np.float32))
            stream.input_finished()
            while self._rec.is_ready(stream):
                self._rec.decode_stream(stream)
            text = self._rec.get_result_all(stream).text
        else:
            stream = self._rec.create_stream()
            stream.accept_waveform(16000, audio)
            self._rec.decode_stream(stream)
            text = stream.result.text
        return TranscriptionResult(text=(text or "").strip(), language="", engine=self.id)


# ============================================================ FunASR
class FunASRTranscriber(Transcriber):
    id = "funasr"
    label = "FunASR（阿里达摩院）"
    capabilities = {"streaming": False, "timestamps": True, "punctuation": True,
                    "local": True, "languages": ["zh", "en"]}

    def __init__(self, cfg: TranscriberConfig):
        super().__init__(cfg)
        self._funasr_name = cfg.funasr_model or "paraformer-zh"

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
            raise TranscriberError("未安装 funasr。请 `pip install funasr modelscope`。") from e
        self._device = self._resolve_device()
        try:
            self._model = AutoModel(model=self._funasr_name, device=self._device,
                                    disable_update=True, trust_remote_code=True)
        except Exception as e:
            raise TranscriberError(f"加载 FunASR 模型 {self._funasr_name} 失败: {e}") from e
        self._loaded = True

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self._loaded:
            self.load()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return TranscriptionResult(text="", engine=self.id)
        try:
            res = self._model.generate(input=audio, batch_size_s=300, use_itn=True)
        except Exception as e:
            raise TranscriberError(f"FunASR 识别失败: {e}") from e
        text = ""
        if res and isinstance(res, list) and isinstance(res[0], dict):
            text = (res[0].get("text") or "").strip()
        elif res:
            text = str(res[0]).strip()
        return TranscriptionResult(text=text, language="zh", engine=self.id)


# ============================================================ Vosk
class VoskTranscriber(Transcriber):
    id = "vosk"
    label = "Vosk（轻量离线）"
    capabilities = {"streaming": True, "timestamps": True, "punctuation": False,
                    "local": True, "languages": []}

    def _find_model(self) -> str:
        if self.cfg.vosk_model and os.path.isdir(self.cfg.vosk_model):
            return self.cfg.vosk_model
        import glob
        cands = sorted(glob.glob(os.path.join(_MODELS_ROOT, "vosk-*")))
        cands = [c for c in cands if os.path.isdir(c)]
        if not cands:
            raise TranscriberError("找不到 Vosk 模型。请把 vosk-model-* 目录放到 models/ 下，"
                                   "或设置 transcriber.vosk_model。")
        return cands[0]

    def load(self) -> None:
        try:
            import vosk
        except ImportError as e:
            raise TranscriberError("未安装 vosk。请 `pip install vosk`。") from e
        self._vosk = vosk
        vosk.SetLogLevel(-1)
        self._model = vosk.Model(self._find_model())
        self._rec = vosk.KaldiRecognizer(self._model, 16000)
        self._rec.SetWords(True)
        self._loaded = True

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self._loaded:
            self.load()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return TranscriptionResult(text="", engine=self.id)
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16).tobytes()
        self._rec.AcceptWaveform(pcm)
        res = json.loads(self._rec.FinalResult() or "{}")
        words = [Word(w.get("start", 0), w.get("end", 0), w.get("word", ""),
                      float(w.get("conf", 0) or 0)) for w in res.get("result", [])]
        return TranscriptionResult(text=(res.get("text") or "").strip(),
                                   words=words, engine=self.id,
                                   confidence=float(np.mean([w.probability for w in words])) if words else 0.0)


# ============================================================ whisper.cpp
class WhisperCppTranscriber(Transcriber):
    id = "whisper_cpp"
    label = "whisper.cpp（GGML/CPU）"
    capabilities = {"streaming": False, "timestamps": True, "punctuation": True,
                    "local": True, "languages": []}

    def _find_bin(self) -> str:
        if self.cfg.whisper_cpp_bin and os.path.exists(self.cfg.whisper_cpp_bin):
            return self.cfg.whisper_cpp_bin
        import glob
        names = ["whisper-cli.exe", "main.exe", "whisper.exe"]
        for n in names:
            p = shutil.which(n)
            if p:
                return p
        hits = []
        for n in names:
            hits += glob.glob(os.path.join(app_root(), "**", n), recursive=True)
        if hits:
            return hits[0]
        raise TranscriberError("找不到 whisper.cpp 可执行文件。请设置 transcriber.whisper_cpp_bin。")

    def _find_model(self) -> str:
        if self.cfg.whisper_cpp_model and os.path.exists(self.cfg.whisper_cpp_model):
            return self.cfg.whisper_cpp_model
        import glob
        hits = sorted(glob.glob(os.path.join(_MODELS_ROOT, "**", "ggml-*.bin"), recursive=True))
        if not hits:
            raise TranscriberError("找不到 ggml 模型（models/**/ggml-*.bin）。")
        return hits[0]

    def load(self) -> None:
        self._bin = self._find_bin()
        self._model = self._find_model()
        self._loaded = True

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self._loaded:
            self.load()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return TranscriptionResult(text="", engine=self.id)
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(tmp, "seg.wav")
            pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
            with wave.open(wav_path, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                w.writeframes(pcm.tobytes())
            cmd = [self._bin, "-m", self._model, "-f", wav_path, "-oj", "-nt"]
            if self.cfg.language:
                cmd += ["-l", self.cfg.language]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                raise TranscriberError(f"whisper.cpp 失败: {proc.stderr[:400]}")
            text = (proc.stdout or "").strip()
        return TranscriptionResult(text=text, engine=self.id)


# ============================================================ 浏览器端
class BrowserTranscriber(Transcriber):
    """占位：识别在浏览器里完成，后端只负责接收文本。"""

    id = "browser"
    label = "浏览器识别（Web Speech API）"
    capabilities = {"streaming": True, "timestamps": False, "punctuation": False,
                    "local": False, "languages": []}

    def load(self) -> None:
        self._loaded = True

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        return TranscriptionResult(text="", engine=self.id)


# ============================================================ 注册表
_REGISTRY = {
    "sherpa_onnx": SherpaOnnxTranscriber,
    "faster_whisper": FasterWhisperTranscriber,
    "openai_whisper": OpenAIWhisperTranscriber,
    "funasr": FunASRTranscriber,
    "vosk": VoskTranscriber,
    "whisper_cpp": WhisperCppTranscriber,
    "browser": BrowserTranscriber,
}

_ALIASES = {
    "sherpa": "sherpa_onnx", "sherpa-onnx": "sherpa_onnx",
    "faster-whisper": "faster_whisper",
    "openai": "openai_whisper", "openai-whisper": "openai_whisper",
    "fun_asr": "funasr",
    "whisper.cpp": "whisper_cpp", "whispercpp": "whisper_cpp",
    "web": "browser", "webspeech": "browser",
}


def normalize_engine(name: str) -> str:
    key = (name or "").strip().lower()
    return _ALIASES.get(key, key)


def create_transcriber(cfg: TranscriberConfig) -> Transcriber:
    key = normalize_engine(cfg.engine)
    cls = _REGISTRY.get(key)
    if cls is None:
        raise TranscriberError(f"未知转录引擎: {cfg.engine}")
    return cls(cfg)


def engine_catalog() -> List[dict]:
    """给 Web 端用的引擎清单（含安装状态）。"""
    out = []
    for key, cls in _REGISTRY.items():
        inst = cls.__dict__.get("capabilities", {})
        # 依赖是否装了（只做轻量 import 检查，不加载模型）
        installed, note = _check_installed(key)
        out.append({
            "id": key,
            "label": getattr(cls, "label", key),
            "capabilities": inst,
            "installed": installed,
            "note": note,
        })
    return out


def _check_installed(key: str) -> tuple:
    """只查包是否安装，不 import（import torch/whisper 会拖慢接口 10s+）。"""
    import importlib.util

    modules = {
        "sherpa_onnx": "sherpa_onnx",
        "faster_whisper": "faster_whisper",
        "openai_whisper": "whisper",
        "funasr": "funasr",
        "vosk": "vosk",
    }
    if key in modules:
        spec = importlib.util.find_spec(modules[key])
        if spec is None:
            return False, "未安装（pip install " + modules[key] + "）"
        return True, ""
    if key == "whisper_cpp":
        found = shutil.which("whisper-cli.exe") or shutil.which("whisper-cli")
        return (found is not None), "需要 whisper.cpp 可执行文件（whisper-cli.exe）"
    if key == "browser":
        return True, "由浏览器提供，Chrome/Edge 通常会上传到厂商服务器"
    return False, "未安装"
