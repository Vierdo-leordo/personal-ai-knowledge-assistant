"""LLM Engine: a thin, dependency-free wrapper over Gemini, OpenAI and Claude (Anthropic) REST APIs.

If no API key is configured the engine runs in *offline* mode: ``is_offline`` is True and the
higher-level components fall back to extractive / heuristic generation (see ``offline.py``), so the
app is fully usable for demos and tests without any key.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

import requests

from .config import Settings


class LLMError(RuntimeError):
    pass


PROVIDERS = {"groq", "gemini", "openai", "anthropic"}
FALLBACK_ORDER = ["groq", "gemini", "openai", "anthropic"]


def retry_delay_seconds(response) -> Optional[float]:
    """Seconds the API asks us to wait (Retry-After header or 'retry in 25.6s' in the body)."""
    if response is None:
        return None
    header = response.headers.get("retry-after") if getattr(response, "headers", None) else None
    try:
        if header:
            return float(header)
    except ValueError:
        pass
    m = re.search(r"retry (?:in|after) ([\d.]+)\s*s", response.text or "", re.I) or \
        re.search(r'"retryDelay":\s*"([\d.]+)s"', response.text or "")
    return float(m.group(1)) if m else None


class LLMEngine:
    def __init__(self, settings: Settings, provider: Optional[str] = None):
        self.settings = settings
        self.provider = provider or settings.resolved_llm_provider()
        if self.provider not in PROVIDERS | {"offline"}:
            raise ValueError(f"Unknown LLM provider '{self.provider}'")

    # ------------------------------------------------------------------ info
    @property
    def is_offline(self) -> bool:
        return self.provider == "offline"

    @property
    def model_name(self) -> str:
        return {
            "gemini": self.settings.gemini_model,
            "openai": self.settings.openai_model,
            "anthropic": self.settings.anthropic_model,
            "groq": self.settings.groq_model,
            "offline": "extractive (no API key)",
        }[self.provider]

    # ------------------------------------------------------------------ core
    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: int = 2048,
        json_mode: bool = False,
        retries: int = 2,
    ) -> str:
        """Call the configured provider; on quota errors wait if short, else fall back to another keyed provider."""
        if self.is_offline:
            raise LLMError("No LLM API key configured (offline mode).")
        temperature = self.settings.temperature if temperature is None else temperature
        order = [self.provider] + [p for p in FALLBACK_ORDER
                                   if p != self.provider and self.settings.provider_has_key(p)]
        errors: List[str] = []
        for provider in order:
            try:
                return self._call_with_retries(provider, prompt, system, temperature, max_tokens, json_mode, retries)
            except LLMError as e:
                errors.append(str(e))
                if not getattr(e, "retryable_elsewhere", False):
                    raise
        raise LLMError(" | ".join(errors))

    def _call_with_retries(self, provider, prompt, system, temperature, max_tokens, json_mode, retries) -> str:
        fn = {"gemini": self._gemini, "openai": self._openai, "anthropic": self._anthropic,
              "groq": self._groq}[provider]
        for attempt in range(retries + 1):
            try:
                return fn(prompt, system, temperature, max_tokens, json_mode).strip()
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                detail = e.response.text[:500] if e.response is not None else str(e)
                if code == 429:
                    wait = retry_delay_seconds(e.response)
                    wait = 5 * (attempt + 1) if wait is None else wait + 1
                    if attempt < retries and wait <= self.settings.max_rate_limit_wait:
                        time.sleep(wait)  # free-tier per-minute limit: wait it out
                        continue
                if code in (500, 502, 503, 504) and attempt < retries:
                    time.sleep(2 ** attempt * 2)
                    continue
                err = LLMError(f"{provider} API error {code}: {detail}")
                err.retryable_elsewhere = code in (401, 403, 404, 429, 500, 502, 503, 504)
                raise err from e
            except requests.RequestException as e:
                if attempt < retries:
                    time.sleep(2)
                    continue
                err = LLMError(f"{provider} request failed: {e}")
                err.retryable_elsewhere = True
                raise err from e
        raise LLMError(f"{provider}: retries exhausted")  # pragma: no cover

    def generate_json(self, prompt: str, system: str = "", **kw) -> Any:
        text = self.generate(prompt, system=system, json_mode=True, **kw)
        return parse_json(text)

    # ------------------------------------------------------------------ providers
    def _gemini(self, prompt, system, temperature, max_tokens, json_mode) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.settings.gemini_model}:generateContent"
        body: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        r = requests.post(url, headers={"x-goog-api-key": self.settings.gemini_api_key}, json=body, timeout=120)
        if r.status_code == 404:
            # Retired model: Google's error names the replacement ("...use models/gemini-x-flash...").
            m = re.search(r"use models/([\w.\-]+)", r.text)
            if m and m.group(1) != self.settings.gemini_model:
                self.settings.gemini_model = m.group(1)
                return self._gemini(prompt, system, temperature, max_tokens, json_mode)
        r.raise_for_status()
        data = r.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError):
            reason = data.get("promptFeedback", {}).get("blockReason") or data.get("candidates", [{}])[0].get(
                "finishReason"
            )
            raise LLMError(f"Gemini returned no text (reason: {reason})")

    def _openai(self, prompt, system, temperature, max_tokens, json_mode) -> str:
        return self._openai_compatible("openai", "https://api.openai.com/v1", self.settings.openai_api_key,
                                       prompt, system, temperature, max_tokens)

    def _groq(self, prompt, system, temperature, max_tokens, json_mode) -> str:
        return self._openai_compatible("groq", "https://api.groq.com/openai/v1", self.settings.groq_api_key,
                                       prompt, system, temperature, max_tokens)

    # preferred chat models, best first; used when the configured model has been retired
    PREFERRED_MODELS = {
        "groq": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3-32b",
                 "llama-3.1-8b-instant"],
        "openai": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
    }
    _NON_CHAT = ("whisper", "tts", "guard", "embed", "moderation", "dall-e", "image", "audio", "realtime",
                 "transcribe", "orpheus", "prompt-guard", "compound")

    def _candidate_models(self, provider: str, base_url: str, api_key: str, current: str) -> List[str]:
        """Replacement chat models, best first: from the provider's live model list, else a known-good list."""
        prefs = [m for m in self.PREFERRED_MODELS.get(provider, []) if m != current]
        try:
            r = requests.get(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
            r.raise_for_status()
            ids = [m["id"] for m in r.json().get("data", []) if m.get("active", True)]
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return prefs
        chat = [i for i in ids if i != current and not any(x in i.lower() for x in self._NON_CHAT)]
        return [p for p in prefs if p in chat] + [c for c in chat if c not in prefs] or prefs

    @staticmethod
    def _model_missing(r) -> bool:
        t = r.text.lower()
        return r.status_code in (400, 404) and "model" in t and (
            "not_found" in t or "does not exist" in t or "decommissioned" in t or "not found" in t)

    def _openai_compatible(self, provider, base_url, api_key, prompt, system, temperature, max_tokens) -> str:
        model = getattr(self.settings, f"{provider}_model")
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        def call(m):
            body = {"model": m, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
            return requests.post(f"{base_url}/chat/completions", headers={"Authorization": f"Bearer {api_key}"},
                                 json=body, timeout=120)

        r = call(model)
        if self._model_missing(r):  # model retired -> switch to one that exists (and remember it)
            for candidate in self._candidate_models(provider, base_url, api_key, model)[:5]:
                r = call(candidate)
                if not self._model_missing(r):
                    setattr(self.settings, f"{provider}_model", candidate)
                    break
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"] or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models may emit hidden thoughts

    def _anthropic(self, prompt, system, temperature, max_tokens, json_mode) -> str:
        body: Dict[str, Any] = {
            "model": self.settings.anthropic_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
            timeout=120,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")


def parse_json(text: str) -> Any:
    """Parse JSON from an LLM response, tolerating ```json fences and leading chatter."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("[", "]"), ("{", "}")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError("Could not parse JSON from model output")
