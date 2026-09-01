"""OpenAI-compatible chat helper for the legal benchmark, with PER-CALL provider selection so the edit-closure
JUDGE and the EDITOR can be different models (removing same-family circularity in Gate B).

- provider="local": Qwen3.8-27B on the proxmox GPU (:31111, thinking disabled).
- provider="kimi":  Moonshot Kimi over the cloud API (bearer auth, temperature=1, retry-on-empty).
"""
from __future__ import annotations

import json
import os
import urllib.request

_LOCAL_ENDPOINT = os.environ.get("CR_MODEL_ENDPOINT", "http://192.168.40.105:31111/v1")
_DEFAULT = "local"


def configure(provider: str = "local") -> None:
    global _DEFAULT
    _DEFAULT = provider


def _settings(provider: str) -> dict:
    if provider == "openai":                    # third independent family (Judge C); temp=0, deterministic-ish
        return {"endpoint": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "key": os.environ.get("OPENAI_API_KEY", ""),
                "model": os.environ.get("CR_OPENAI_MODEL", "gpt-4o-mini"),
                "vllm": False, "temp": 0.0}
    if provider == "kimi":
        return {"endpoint": (os.environ.get("KIMI_AGENT_BASE_URL") or os.environ.get("KIMI_BASE_URL")
                             or "https://api.moonshot.ai/v1"),
                "key": os.environ.get("KIMI_AGENT_API_KEY") or os.environ.get("KIMI_API_KEY") or "",
                "model": os.environ.get("CR_KIMI_MODEL", "kimi-k2.7-code-highspeed"),
                "vllm": False, "temp": 1.0}
    return {"endpoint": _LOCAL_ENDPOINT, "key": "", "model": os.environ.get("CR_MODEL_NAME", ""),
            "vllm": True, "temp": 0.0}


def model_id(provider: str | None = None) -> str:
    s = _settings(provider or _DEFAULT)
    if s["model"]:
        return s["model"]
    with urllib.request.urlopen(f"{s['endpoint']}/models", timeout=15) as r:
        return json.load(r)["data"][0]["id"]


def chat(system: str, user: str, max_tokens: int = 1500, retries: int = 3,
         provider: str | None = None) -> str:
    s = _settings(provider or _DEFAULT)
    model = s["model"] or model_id(provider or _DEFAULT)
    payload = {"model": model,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "temperature": s["temp"], "max_tokens": max_tokens}
    if s["vllm"]:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if s["key"]:
        headers["Authorization"] = f"Bearer {s['key']}"
    req = urllib.request.Request(f"{s['endpoint']}/chat/completions", data=json.dumps(payload).encode(),
                                 headers=headers)
    content = ""
    for _ in range(max(1, retries)):
        with urllib.request.urlopen(req, timeout=240) as r:
            content = json.load(r)["choices"][0]["message"].get("content") or ""
        if content.strip():
            return content
    return content


def extract_json(text: str):
    import re
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    body = m.group(1) if m else text
    # prefer an object when the body clearly starts with one; else the first array
    order = [("{", "}"), ("[", "]")] if body.lstrip()[:1] == "{" else [("[", "]"), ("{", "}")]
    for open_c, close_c in order:
        i, j = body.find(open_c), body.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(body[i:j + 1])
            except Exception:
                continue
    return None
