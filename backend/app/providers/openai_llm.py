from __future__ import annotations

import json
import logging

from tenacity import retry

from ..config import get_settings
from ..errors import PipelineError
from ._common import RETRY_KW, make_client
from .base import Glossary, TranslateItem

log = logging.getLogger(__name__)

TARGET_NAME = {
    "en": "natural spoken American English",
    "vi": "natural spoken Vietnamese (standard, with full diacritics)",
    "zh": "Simplified Chinese (Mandarin, mainland China), no pinyin",
}

TRANSLATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "text"],
                "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
            },
        }
    },
}

SHORTEN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text"],
    "properties": {"text": {"type": "string"}},
}

SYSTEM_TRANSLATE = """You are a professional subtitle and voice-over translator.
Translate each item from {source_lang} into {target_name}.
Rules:
1. The text will be read aloud by a voice actor within a fixed time slot. Each item has max_chars:
   your translation MUST NOT exceed max_chars characters. Prefer concise, natural phrasing; drop filler words.
2. Translate meaning using the context of the whole list; items may be fragments of the same sentence.
3. Keep exactly the same ids. Return every id exactly once. Do not merge or split items.
4. Keep person names, brand names and product names unchanged unless the glossary says otherwise.
5. Glossary (MUST use these translations): {glossary}
6. Write numbers as digits. No explanations, no quotes, no notes."""

SYSTEM_SHORTEN = """You rewrite voice-over lines in {target_name} to be shorter.
Rewrite the given text so it has at most {max_chars} characters, keeping the core meaning and natural spoken style.
Glossary (MUST use these translations): {glossary}
Return only the rewritten text."""

BATCH = 60
SINGLE_CALL_MAX = 80
CONTEXT = 5


def _glossary_lines(glossary: Glossary) -> str:
    lines = [f'"{s}" => "{t}"' for s, t in glossary if s and t]
    return "; ".join(lines) if lines else "none"


class OpenAITranslator:
    def __init__(self) -> None:
        self.client = make_client()

    @retry(**RETRY_KW)
    def _chat(self, system: str, user: str, schema_name: str, schema: dict) -> dict:
        resp = self.client.chat.completions.create(
            model=get_settings().OPENAI_LLM_MODEL,
            temperature=0.2,
            response_format={"type": "json_schema",
                             "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)

    def _translate_batch(self, items: list[TranslateItem], source_lang: str, target_lang: str,
                         glossary: Glossary, context: list[tuple[str, str]]) -> dict[int, str]:
        system = SYSTEM_TRANSLATE.format(source_lang=source_lang or "the source language",
                                         target_name=TARGET_NAME[target_lang],
                                         glossary=_glossary_lines(glossary))
        payload: dict = {"items": [{"id": i.id, "text": i.text, "max_chars": i.max_chars} for i in items]}
        if context:
            payload["previous_context"] = [{"source": s, "translation": t} for s, t in context]
        user = json.dumps(payload, ensure_ascii=False)
        want = {i.id for i in items}
        last_err = ""
        for _ in range(3):  # 1 try + 2 retries
            try:
                data = self._chat(system, user, "translations", TRANSLATE_SCHEMA)
            except json.JSONDecodeError as e:
                last_err = f"JSON lỗi: {e}"
                continue
            got = {int(x["id"]): str(x["text"]).strip() for x in data.get("items", [])}
            if set(got) == want and all(got.values()):
                return got
            last_err = f"thiếu/thừa id: {sorted(want ^ set(got))}"
            log.warning("translate validation failed: %s", last_err)
        raise PipelineError(f"Dịch thất bại ({last_err})")

    def translate(self, items: list[TranslateItem], source_lang: str, target_lang: str,
                  glossary: Glossary) -> dict[int, str]:
        if len(items) <= SINGLE_CALL_MAX:
            return self._translate_batch(items, source_lang, target_lang, glossary, [])
        out: dict[int, str] = {}
        for k in range(0, len(items), BATCH):
            batch = items[k : k + BATCH]
            prev = items[max(0, k - CONTEXT) : k]
            ctx = [(p.text, out[p.id]) for p in prev if p.id in out]
            out.update(self._translate_batch(batch, source_lang, target_lang, glossary, ctx))
        return out

    def shorten(self, text: str, target_lang: str, max_chars: int, glossary: Glossary) -> str:
        system = SYSTEM_SHORTEN.format(target_name=TARGET_NAME[target_lang], max_chars=max_chars,
                                       glossary=_glossary_lines(glossary))
        try:
            data = self._chat(system, json.dumps({"text": text, "max_chars": max_chars}, ensure_ascii=False),
                              "shortened", SHORTEN_SCHEMA)
            new = str(data.get("text", "")).strip()
            return new or text
        except Exception as e:  # noqa: BLE001 - shortening is best effort
            log.warning("shorten failed: %r", e)
            return text
