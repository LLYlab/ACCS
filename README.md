# ACCS — AI Capture Course Sound

用**本机运行的 Whisper**对讲课音频做**实时语音识别**,把讲师讲的话实时转成文字,并可调用**本机 DeepSeek R1** 把英文实时翻成中文。

> 目标场景: 一边播放网课/讲座/录屏(或直接对麦克风说话),一边实时得到中文/英文文字稿 + 中文翻译。

---

## 特性

- **带 Web UI**: 浏览器打开即可选择音频来源、开始/停止、实时看到转写与翻译。
- **实时**采集 + 实时转写: 采集到说话就切分、立刻转写,不阻塞后续音频。
- **多种音频来源**:
  - `system` — 抓取**系统正在播放的声音**(网课/讲座/录屏) —— Windows 上通过 WASAPI 环回。
  - `mic` — 用**麦克风**收音。
  - `file` — 对**本地音频文件**离线转写(用于测试/复盘)。
- **本机运行**: 音频不外传,完全本地推理;翻译也走本机 Ollama。
- **VAD 端点切分**: 用 Silero VAD 自动判断说话起止,把连续语音切成句子,避免把整堂课当一段。
- **可配置引擎**: `faster-whisper`(推荐,实时低延迟)或 `openai-whisper`(复用已装好的 PyTorch)。
- **多种输出**: UI 实时显示 + CLI 落盘 `txt` / `srt` / `jsonl`。

---

## 环境要求

- Python 3.9+(已在 3.9.13 验证)
- 本机安装过 `ffmpeg`(用于音频解码,`faster-whisper` 依赖)。可用 `winget install Gyan.FFmpeg`。
- (可选)GPU: 若要 GPU 加速,见下方「GPU 加速」一节。

## 安装

```powershell
cd ACCS
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 运行

```powershell
# 抓取系统正在播放的声音(网课/讲座) —— 最常用
.\.venv\Scripts\python.exe -m accs --source system

# 用麦克风收音
.\.venv\Scripts\python.exe -m accs --source mic

# 对本地文件离线转写(中文, small 模型)
.\.venv\Scripts\python.exe -m accs --source file --file lecture.m4a --language zh --output all

# 先用仓库附带的测试音频快速试一把(中文, tiny 模型)
.\.venv\Scripts\python.exe -m accs --source file --file speech_zh.wav --language zh --output all

# 查看可用音频设备(确认系统环回/麦克风的名称)
.\.venv\Scripts\python.exe -m accs --list-devices
```

停止: 在终端按 `Ctrl+C`。

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `--source system/mic/file` | 音频来源 |
| `--model tiny/base/small/medium/large-v3` | Whisper 模型,越大越准、越慢 |
| `--language zh/en/...` | 语言;留空自动检测 |
| `--engine faster_whisper/openai_whisper` | 转录引擎 |
| `--device-auto cpu/cuda` | 计算设备 |
| `--output terminal/file/srt/jsonl/all` | 输出方式 |
| `--output-file path` | 输出文件路径 |
| `--no-vad` | 关闭 VAD,按固定窗口切分 |
| `--silence-ms 800` | 判定句子结束的静音时长(毫秒) |

所有配置也可在 `config.yaml` 中修改,命令行参数会覆盖它。

---

## Web 界面(推荐)

带界面的应用: 浏览器里选择音频来源、开始/停止,实时看到转写 + 英文→中文翻译。

```powershell
# 启动 Web 服务
.\.venv\Scripts\python.exe -m accs.web --port 8000
```

然后浏览器打开 **http://127.0.0.1:8000** 。

界面上你能:
- 选音频来源(系统播放声音 / 麦克风 / 音频文件)。
- 选 Whisper 模型、语言、引擎。
- 开/关「英→中翻译」。
- 填 LLM 地址与模型(默认 `http://127.0.0.1:11434` + `deepseek-r1:8b`)。

> 翻译依赖本机 **Ollama** 跑 `deepseek-r1:8b`(你机器上已有,`ollama serve` 已在 11434 端口)。若想换模型或没开 Ollama,在界面里改 LLM 地址/模型,或取消勾选翻译。
> R1 每次翻译约 10~20 秒,是**后台**跑的,不会卡住实时转写;转写会先出来,译文随后补上。

> 文件模式: 在「音频文件」框填项目目录内的相对路径(如 `speech_en.wav`)。


## GPU 加速

- **faster-whisper**: GPU 需要 **CUDA + cuDNN**(CTranslate2)。未配置时会自动回退 CPU `int8`,也能实时用 `tiny/base`,但速度较慢。
- **openai-whisper**: 复用已安装的 PyTorch。若你的 torch 是 CUDA 版(如 `torch 2.8.0+cu128`),直接 `--engine openai_whisper` 即可用 GPU。

> 提示: 默认设备为空时,`--device-auto auto` 会自动探测。若在 `cuda` 上加载失败,程序会回退到 CPU `int8`。

---

## 项目结构

```
ACCS/
├── accs/
│   ├── __main__.py      CLI 入口 (python -m accs)
│   ├── web.py           Web 服务入口 (python -m accs.web)
│   ├── webapp.py        FastAPI + SSE 广播 + 流水线管理
│   ├── translator.py    调用本机 Ollama(DeepSeek R1)做英→中翻译
│   ├── config.py        配置加载(读 config.yaml + 命令行覆盖)
│   ├── audio.py         音频采集: system(环回) / mic / file,多后端回退
│   ├── vad.py           Silero VAD(端点检测),能量 VAD 作为兜底
│   ├── transcriber.py   转录引擎: faster-whisper / openai-whisper
│   ├── pipeline.py      流水线: 采集 → VAD 切分 → 转录 → 输出
│   ├── output.py        输出: 实时终端 + txt/srt/jsonl
│   └── static/index.html  Web 界面(单页,原生 JS)
├── config.yaml          默认配置
├── requirements.txt     依赖
├── speech_zh.wav        中文测试音频
├── speech_en.wav        英文测试音频(用于测试翻译)
└── tests/test_logic.py  核心逻辑自检(无需硬件/模型)
```

## 工作原理

```
[音频源: system/mic/file] --16kHz单声道帧--> [Silero VAD 端点检测]
      |                                             |
      |            检测到说话开始: 累积 + 前置补音        |
      |            检测到静音/超长: 定稿当前片段           |
      v                                             v
[后台转录线程: Whisper 转写该片段]  -->  [输出: 终端打印 + 落盘 txt/srt/jsonl]
```

- 采集与 VAD 在调用线程里实时跑,转录在后台线程;即使模型偏慢,也不影响继续采集。
- 一句话说完(静音超过 `silence_ms`)就送一句去转,所以是一句一句实时出来的。

---

## 常见问题

- **没有 loopback 设备?** 确保系统正在播放声音(音量不为 0),并检查 soundcard;有的系统需在声音设置里允许应用独占音频。
- **中文识别效果差?** 用更大模型(`small`/`medium`),或显式 `--language zh`。
- **想要更快的实时?** 用 `small` + GPU(`float16`),或 CPU `int8` + `tiny`。
- **输出到文件看不见?** 用 `--output all` 或 `--output-file path.txt`。
