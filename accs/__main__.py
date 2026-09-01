from __future__ import annotations

import argparse
import sys

from . import __version__
from .audio import list_devices
from .config import Config, DEFAULT_CONFIG_PATH
from .pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="accs",
        description="ACCS — AI Capture Course Sound: 本机实时语音识别(Whisper),转写讲师讲课音频。",
    )
    p.add_argument("-c", "--config", default=None, help="配置文件路径(默认 config.yaml)")
    p.add_argument("--version", action="version", version=f"accs {__version__}")
    p.add_argument("--list-devices", action="store_true", help="列出可用音频设备后退出")

    grp = p.add_argument_group("音频")
    grp.add_argument("--source", choices=["system", "mic", "file"], help="音频来源")
    grp.add_argument("--file", default=None, help="source=file 时的音频文件路径")
    grp.add_argument("--backend", choices=["auto", "soundcard", "sounddevice"], help="音频后端")
    grp.add_argument("--device", default=None, help="设备关键字(按名称/ID 匹配)")
    grp.add_argument("--mic-device", default=None, help="麦克风设备关键字")
    grp.add_argument("--system-device", default=None, help="系统环回设备关键字")
    grp.add_argument("--sample-rate", type=int, default=None, help="采样率(默认 16000)")
    grp.add_argument("--frame-ms", type=int, default=None, help="采集帧时长(默认 256)")

    grp = p.add_argument_group("转录")
    grp.add_argument("--engine", choices=["faster_whisper", "openai_whisper"], help="转录引擎")
    grp.add_argument("--model", default=None, help="Whisper 模型 (tiny/base/small/medium/large-v3 或本地路径)")
    grp.add_argument("--device-auto", choices=["auto", "cpu", "cuda"], dest="device_auto", default=None, help="计算设备")
    grp.add_argument("--compute-type", default=None, help="int8/float16/int8_float16/float32")
    grp.add_argument("--language", default=None, help="语言 (zh/en/...), 留空自动检测")
    grp.add_argument("--beam-size", type=int, default=None, help="beam size (默认 5)")

    grp = p.add_argument_group("VAD")
    grp.add_argument("--no-vad", action="store_true", help="禁用 VAD,按固定窗口切分")
    grp.add_argument("--silence-ms", type=int, default=None, help="判定句子结束的静音时长")

    grp = p.add_argument_group("输出")
    grp.add_argument("--output", choices=["terminal", "file", "srt", "jsonl", "all"], dest="output_mode", default=None, help="输出方式")
    grp.add_argument("--output-file", default=None, help="输出文件路径")
    grp.add_argument("--quiet", action="store_true", help="不实时打印到终端")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        print(list_devices())
        return 0

    overrides = {}
    if args.source:
        overrides["audio.source"] = args.source
    if args.file:
        overrides["audio.file"] = args.file
    if args.backend:
        overrides["audio.backend"] = args.backend
    if args.device:
        overrides["audio.device"] = args.device
    if args.mic_device:
        overrides["audio.mic_device"] = args.mic_device
    if args.system_device:
        overrides["audio.system_device"] = args.system_device
    if args.sample_rate:
        overrides["audio.sample_rate"] = args.sample_rate
    if args.frame_ms:
        overrides["audio.frame_ms"] = args.frame_ms

    if args.engine:
        overrides["transcriber.engine"] = args.engine
    if args.model:
        overrides["transcriber.model"] = args.model
    if args.device_auto:
        overrides["transcriber.device"] = args.device_auto
    if args.compute_type:
        overrides["transcriber.compute_type"] = args.compute_type
    if args.language:
        overrides["transcriber.language"] = args.language
    if args.beam_size:
        overrides["transcriber.beam_size"] = args.beam_size

    if args.no_vad:
        overrides["vad.enabled"] = False
    if args.silence_ms:
        overrides["vad.silence_ms"] = args.silence_ms

    if args.output_mode:
        overrides["output.mode"] = args.output_mode
    if args.output_file:
        overrides["output.file"] = args.output_file
    if args.quiet:
        overrides["output.show_transcript_live"] = False

    try:
        config = Config.load(args.config or DEFAULT_CONFIG_PATH, overrides=overrides)
    except Exception as e:
        print(f"[accs] 配置加载失败: {e}", file=sys.stderr)
        return 2

    if config.audio.source == "file" and not config.audio.file:
        print("[accs] source=file 需要用 --file 指定音频路径。", file=sys.stderr)
        return 2

    try:
        Pipeline(config).run()
        return 0
    except KeyboardInterrupt:
        print("\n[accs] 已停止。", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"[accs] 运行失败: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
