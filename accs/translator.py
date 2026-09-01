from __future__ import annotations

import re
from typing import Optional

import httpx


class TranslationError(RuntimeError):
    pass


class Translator:
    """调用本机 Ollama 的 DeepSeek R1 等模型做英->中翻译。

    ── 模型说明 ─────────────────────────────────────────────
    Ollama 的 /api/chat 对 R1 这类推理模型会单独返回 thinking 字段,
    message.content 里是干净的最终答案,因此直接取 content 即可。
    防御性清理会去掉可能泄漏的 <...> 思考块。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "deepseek-r1:8b",
        timeout: float = 120.0,
        source_lang: str = "en",
        target_lang: str = "zh",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.source_lang = source_lang
        self.target_lang = target_lang

    def available(self) -> bool:
        """检测服务是否在线且模型存在。"""
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            if r.status_code != 200:
                return False
            for m in r.json().get("models", []):
                if m.get("name", "").split(":")[0] == self.model.split(":")[0]:
                    return True
            return True  # 服务在线,即使未精确匹配也认为可用
        except Exception:
            return False

    def _clean(self, text: str) -> str:
        """去掉可能泄漏的思考块。"""
        t = text or ""
        # 去掉 ... 或 <thinking> ... </thinking> 块
        t = re.sub(r"<thinking>.*?</thinking>", "", t, flags=re.S)
        t = re.sub(r"<[^>]*>", "", t)
        # 去掉 ```think 包裹段落
        t = re.sub(r"```.*?```", "", t, flags=re.S)
        t = t.strip()
        # 去掉 R1 常见的 ``` 包裹标记
        t = t.strip("`").strip()
        return t

    def translate(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        src = {"en": "English", "zh": "Chinese", "ja": "Japanese"}.get(self.source_lang, self.source_lang)
        dst = {"zh": "Simplified Chinese", "en": "English"}.get(self.target_lang, self.target_lang)
        system = (
            "You are a precise translation engine. "
            f"Translate the following {src} text into {dst}. "
            "Output ONLY the translation text. Do not explain, do not add commentary, "
            "do not show your reasoning process. Preserve meaning and natural phrasing."
        )
        prompt = f"Translate into {dst}:\n\n{text}"
        return self._chat(system, prompt, temperature=0.2, num_predict=512)

    def analyze_course(self, transcript: str) -> dict:
        """全面分析: 纠正转写 + 提炼知识点。返回 {"corrected": str, "knowledge_points": [str]}。"""
        transcript = (transcript or "").strip()
        if not transcript:
            return {"corrected": "", "knowledge_points": []}

        # 第 1 步: 纠正成通顺课程正文(不总结、不评论、不加小标题)
        sys_corr = (
            "你是一名资深中文语文校对。下面是课堂语音识别的原始转写，存在大量识别错误、错别字、"
            "缺标点、口语填充词（啊/哎/嗯/对吧/就是等）、中英夹杂。请把它改写成一篇连贯、通顺、"
            "准确的课程讲授正文，按讲课顺序完整讲出这节课的内容。要求：\n"
            "- 写成段落式的讲稿正文，不要用任何标题、编号、列表、小标题；\n"
            "- 保留原有的讲授顺序与内容，不要增删原意；\n"
            "- 修正错词/识别错误、补全缺漏、加标点、删除无意义的口语填充词；\n"
            "- 不要总结、不要评论、不要解释，只输出正文。"
        )
        corrected = self._chat(sys_corr, f"原始转写：\n{transcript}", temperature=0.2, num_predict=4500)

        # 第 2 步: 提炼知识点(每行一个)
        sys_kp = (
            "你是资深课程教研。请从下面的课程内容中提炼关键知识点，每个知识点一句话，精炼概括，"
            "保持讲授顺序。只输出知识点列表，每行一个，不要任何其它文字。"
        )
        kp_raw = self._chat(sys_kp, f"课程内容：\n{corrected}", temperature=0.2, num_predict=1600)
        kpoints = []
        for line in (kp_raw or "").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(("知识点", "要点", "以下是", "课程", "总结")) and len(line) < 12:
                continue
            line = line.lstrip("-•*　 ").strip()
            if line:
                kpoints.append(line)
        return {"corrected": corrected, "knowledge_points": kpoints}

    def _chat(self, system: str, prompt: str, temperature: float = 0.2, num_predict: int = 512) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
            "keep_alive": "10m",
        }
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=180.0)
        except Exception as e:
            raise TranslationError(f"无法连接 Ollama({self.base_url}): {e}") from e
        if r.status_code != 200:
            raise TranslationError(f"Ollama 返回错误 {r.status_code}: {r.text[:300]}")
        data = r.json()
        return self._clean(data["message"]["content"] if "message" in data else "")

    def _parse_analysis(self, raw: str, fallback: str) -> dict:
        import json as _json, re as _re
        text = raw or ""
        # 去掉可能包裹的代码块 ```json ... ```
        text = _re.sub(r"```(?:json)?", "", text).strip("`").strip()
        m = _re.search(r"\{.*\}", text, flags=_re.S)
        if m:
            try:
                obj = _json.loads(m.group(0))
                corrected = str(obj.get("corrected") or "").strip()
                kps = obj.get("knowledge_points") or []
                if isinstance(kps, list):
                    kps = [str(k).strip() for k in kps if str(k).strip()]
                    return {"corrected": corrected, "knowledge_points": kps}
            except Exception:
                pass
        # 兜底: 整体作为纠正文本
        return {"corrected": text or fallback, "knowledge_points": []}


def text_is_chinese(text: str, threshold: float = 0.3) -> bool:
    """粗略判断文本是否已主要是中文。"""
    text = text or ""
    if not text:
        return False
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return (cjk / len(text)) >= threshold


def text_has_english(text: str) -> bool:
    """文本是否含有英文字母(用于判断是否需要翻译成中文)。

    纯中文 -> False(无需翻译); 纯英文 / 中英混合 -> True(需要翻译)。
    """
    text = text or ""
    return any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in text)


def text_needs_translation(text: str) -> bool:
    """是否需要翻译成中文: 只要含英文(英文或中英混合)就需要翻译。"""
    return text_has_english(text)

