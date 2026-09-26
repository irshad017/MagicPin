"""Provider-agnostic LLM client. One `complete(system, prompt)` method.

Supports the same providers as judge_simulator.py so a single key works for both.
With a Groq key the composer runs on an open model (e.g. llama-3.3-70b-versatile);
swap LLM_PROVIDER/LLM_MODEL in .env to move to OpenAI/Gemini/OpenRouter/Anthropic-via-OpenRouter.
"""
import json
import requests

import config


class LLMError(Exception):
    pass


# Some provider edges (Cloudflare) block default library User-Agents; send a browser-like one.
_UA = "Mozilla/5.0 (compatible; vera-bot/1.0)"


def _post_with_retry(url: str, body: dict, headers: dict):
    """POST with bounded backoff on 429/5xx, respecting Retry-After. Stays within the tick budget."""
    import time
    last = None
    for attempt in range(config.LLM_MAX_RETRIES + 1):
        last = requests.post(url, data=json.dumps(body), headers=headers, timeout=config.LLM_TIMEOUT_S)
        if last.status_code not in (429, 500, 502, 503, 529):
            return last
        if attempt == config.LLM_MAX_RETRIES:
            return last
        retry_after = last.headers.get("retry-after")
        try:
            wait = float(retry_after) if retry_after else 0.0
        except ValueError:
            wait = 0.0
        wait = min(max(wait, 0.6 * (2 ** attempt)), config.LLM_MAX_BACKOFF_S)
        time.sleep(wait)
    return last


def _openai_style(url: str, key: str, model: str, system: str, prompt: str,
                  extra_headers: dict | None = None, want_json: bool = True) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": model,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
        "seed": config.LLM_SEED,
    }
    if want_json:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": _UA}
    if extra_headers:
        headers.update(extra_headers)
    r = _post_with_retry(url, body, headers)
    if r.status_code >= 400:
        # Retry once without response_format / seed in case the model/endpoint rejects them.
        if want_json or "seed" in body:
            body.pop("response_format", None)
            body.pop("seed", None)
            r = _post_with_retry(url, body, headers)
    if r.status_code >= 400:
        raise LLMError(f"{r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def _gemini(model: str, key: str, system: str, prompt: str) -> str:
    full = f"{system}\n\n{prompt}" if system else prompt
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    body = {
        "contents": [{"parts": [{"text": full}]}],
        "generationConfig": {
            "temperature": config.LLM_TEMPERATURE,
            "maxOutputTokens": config.LLM_MAX_TOKENS,
            "responseMimeType": "application/json",
        },
    }
    r = requests.post(url, data=json.dumps(body),
                      headers={"Content-Type": "application/json", "User-Agent": _UA},
                      timeout=config.LLM_TIMEOUT_S)
    if r.status_code >= 400:
        raise LLMError(f"{r.status_code}: {r.text[:300]}")
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _ollama(model: str, system: str, prompt: str) -> str:
    full = f"{system}\n\n{prompt}" if system else prompt
    r = requests.post(
        f"{config.OLLAMA_URL}/api/generate",
        data=json.dumps({"model": model, "prompt": full, "stream": False,
                         "format": "json", "options": {"temperature": config.LLM_TEMPERATURE}}),
        headers={"Content-Type": "application/json"}, timeout=90)
    if r.status_code >= 400:
        raise LLMError(f"{r.status_code}: {r.text[:300]}")
    return r.json()["response"]


def complete(system: str, prompt: str) -> str:
    """Return raw model text (expected to be a JSON object string)."""
    p, key, model = config.LLM_PROVIDER, config.LLM_API_KEY, config.LLM_MODEL
    if p == "groq":
        return _openai_style("https://api.groq.com/openai/v1/chat/completions", key, model, system, prompt)
    if p == "openai":
        return _openai_style("https://api.openai.com/v1/chat/completions", key, model, system, prompt)
    if p == "deepseek":
        return _openai_style("https://api.deepseek.com/v1/chat/completions", key, model, system, prompt)
    if p == "openrouter":
        return _openai_style("https://openrouter.ai/api/v1/chat/completions", key, model, system, prompt,
                             extra_headers={"HTTP-Referer": "https://magicpin.com"})
    if p == "gemini":
        return _gemini(model or "gemini-1.5-flash", key, system, prompt)
    if p == "ollama":
        return _ollama(model or "llama3", system, prompt)
    raise LLMError(f"Unknown provider: {p}")
