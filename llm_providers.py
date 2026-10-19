"""
llm_providers.py - one adapter per API provider for the LLM Loftus & Palmer data collection.

Every adapter turns the same provider-neutral message (llm_common.build_parts: text parts and
PNG frames, in order) into that provider's request format, sends it, and normalizes the
response (answer text, finish status, model version, token usage). The content of the message
is identical for all providers; only the wire format differs.

Providers (API key read from .env):
    openai     OpenAI Responses API                   OPENAI_API_KEY
    xai        xAI Responses API (OpenAI-compatible)  XAI_API_KEY
    anthropic  Anthropic Messages API                 ANTHROPIC_API_KEY
    google     Gemini API (google-genai SDK)          GEMINI_API_KEY
    deepseek   DeepSeek Chat Completions (OpenAI SDK) DEEPSEEK_API_KEY

OpenAI and xAI receive exactly the request the pilot sent (same SDK, same payload).
"""

import base64
import time

import llm_common as lc

PROVIDERS = {
    "openai": {"key_env": "OPENAI_API_KEY", "base_url": None},
    "xai": {"key_env": "XAI_API_KEY", "base_url": "https://api.x.ai/v1"},
    "anthropic": {"key_env": "ANTHROPIC_API_KEY", "base_url": None},
    "google": {"key_env": "GEMINI_API_KEY", "base_url": None},
    "deepseek": {"key_env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com"},
}
TRANSIENT_STATUS = (408, 409, 429, 500, 502, 503, 504, 529)


class FatalAPIError(Exception):
    """A request the API rejected; retrying would not help (bad request, auth, quota)."""


class TransientAPIError(Exception):
    """Rate limit, server error, timeout or connection problem; worth retrying."""


def _b64(path):
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _status_error(status_code, message, code=None):
    msg = f"HTTP {status_code} {code or ''}: {message}"
    if (status_code in TRANSIENT_STATUS or status_code >= 500) and code != "insufficient_quota":
        return TransientAPIError(msg)
    return FatalAPIError(msg)


def _normalized(text="", refusal=None, status="", model_version="", response_id="",
                tokens_input=0, tokens_input_cached=0, tokens_output=0, tokens_reasoning=None,
                provider_cost_usd=None):
    """Provider-independent response. tokens_output includes reasoning/thinking tokens;
    tokens_reasoning is None when the provider does not report it separately."""
    return locals()


# =============================================================================
# OpenAI Responses API (OpenAI, xAI)
# =============================================================================
class OpenAIResponsesAdapter:
    sdk = "openai"

    def __init__(self, provider, api_key, timeout):
        from openai import OpenAI
        self.provider = provider
        self.client = OpenAI(api_key=api_key, base_url=PROVIDERS[provider]["base_url"],
                             max_retries=0, timeout=timeout)       # retries handled here

    def list_models(self):
        return {m.id for m in self.client.models.list()}

    def image_detail_label(self, image_detail):
        return image_detail

    def reasoning_request(self, effort):
        return {"reasoning": {"effort": effort}}

    def build_request(self, model, parts, *, reasoning_effort, max_output_tokens, temperature,
                      image_detail):
        req = {"model": model,
               "input": [{"role": "user", "content": lc.openai_content(parts, image_detail)}],
               **self.reasoning_request(reasoning_effort),
               "max_output_tokens": max_output_tokens, "store": False}
        if temperature is not None:
            req["temperature"] = temperature
        return req

    def content_signature(self, req):
        sig = []
        for p in req["input"][0]["content"]:
            if p["type"] == "input_text":
                sig.append(("text", p["text"]))
            else:
                sig.append(("image", base64.b64decode(p["image_url"].split(",", 1)[1])))
        return sig

    def send(self, req):
        import openai
        try:
            return self.client.responses.create(**req)
        except openai.APIStatusError as e:
            raise _status_error(e.status_code, e.message, getattr(e, "code", None)) from None
        except openai.APIConnectionError as e:                # includes timeouts
            raise TransientAPIError(f"{type(e).__name__}: {e}") from None

    def normalize(self, resp):
        # Same logic as the pilot's response_to_row
        text, refusal = "", None
        for out in resp.output or []:
            if getattr(out, "type", None) == "message":
                for part in out.content or []:
                    if part.type == "output_text":
                        text += part.text
                    elif part.type == "refusal":
                        refusal = part.refusal
        status = resp.status or ""
        if status == "incomplete" and resp.incomplete_details is not None:
            status = f"incomplete:{resp.incomplete_details.reason}"
        u = resp.usage
        nano = getattr(u, "cost_in_nano_usd", None) if u else None      # xAI only
        return _normalized(
            text=text, refusal=refusal, status=status, model_version=resp.model,
            response_id=resp.id,
            tokens_input=u.input_tokens if u else 0,
            tokens_input_cached=getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0,
            tokens_output=u.output_tokens if u else 0,
            tokens_reasoning=getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0,
            provider_cost_usd=nano / 1e9 if isinstance(nano, (int, float)) else None)


# =============================================================================
# DeepSeek: OpenAI-format Chat Completions
# =============================================================================
class OpenAIChatAdapter:
    sdk = "openai"
    FINISH = {"stop": "completed", "length": "incomplete:max_output_tokens",
              "content_filter": "blocked:content_filter"}

    def __init__(self, provider, api_key, timeout):
        from openai import OpenAI
        self.provider = provider
        self.client = OpenAI(api_key=api_key, base_url=PROVIDERS[provider]["base_url"],
                             max_retries=0, timeout=timeout)

    def list_models(self):
        return {m.id for m in self.client.models.list()}

    def image_detail_label(self, image_detail):
        return image_detail          # DeepSeek: "high" keeps the original image

    def reasoning_request(self, effort):
        return {"reasoning_effort": effort, "extra_body": {"thinking": {"type": "enabled"}}}

    def build_request(self, model, parts, *, reasoning_effort, max_output_tokens, temperature,
                      image_detail):
        content = []
        for p in parts:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            else:
                content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{_b64(p['path'])}", "detail": image_detail}})
        req = {"model": model, "messages": [{"role": "user", "content": content}],
               "max_tokens": max_output_tokens, **self.reasoning_request(reasoning_effort)}
        if temperature is not None:
            req["temperature"] = temperature
        return req

    def content_signature(self, req):
        sig = []
        for p in req["messages"][0]["content"]:
            if p["type"] == "text":
                sig.append(("text", p["text"]))
            else:
                sig.append(("image", base64.b64decode(p["image_url"]["url"].split(",", 1)[1])))
        return sig

    def send(self, req):
        import openai
        try:
            return self.client.chat.completions.create(**req)
        except openai.APIStatusError as e:
            raise _status_error(e.status_code, e.message, getattr(e, "code", None)) from None
        except openai.APIConnectionError as e:
            raise TransientAPIError(f"{type(e).__name__}: {e}") from None

    def normalize(self, resp):
        choice = resp.choices[0] if resp.choices else None
        msg = choice.message if choice else None
        finish = choice.finish_reason if choice else "no_choices"
        u = resp.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None)
        if cached is None:
            cached = getattr(u, "prompt_cache_hit_tokens", 0)
        return _normalized(
            text=(msg.content or "") if msg else "",
            refusal=getattr(msg, "refusal", None) if msg else None,
            status=self.FINISH.get(finish, str(finish)), model_version=resp.model,
            response_id=resp.id,
            tokens_input=u.prompt_tokens if u else 0, tokens_input_cached=cached or 0,
            tokens_output=u.completion_tokens if u else 0,
            tokens_reasoning=getattr(getattr(u, "completion_tokens_details", None),
                                     "reasoning_tokens", None) if u else None)


# =============================================================================
# Anthropic Messages API
# =============================================================================
class AnthropicAdapter:
    sdk = "anthropic"
    STOP = {"end_turn": "completed", "max_tokens": "incomplete:max_output_tokens",
            "refusal": "refused"}

    def __init__(self, provider, api_key, timeout):
        import anthropic
        self.provider = provider
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=timeout)

    def list_models(self):
        return {m.id for m in self.client.models.list()}

    def image_detail_label(self, image_detail):
        return "native"              # no detail setting; images are used at their 640x360 size

    def reasoning_request(self, effort):
        # Adaptive thinking is on by default on current Claude models; effort sets its depth.
        return {"output_config": {"effort": effort}}

    def build_request(self, model, parts, *, reasoning_effort, max_output_tokens, temperature,
                      image_detail):
        content = []
        for p in parts:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            else:
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": _b64(p["path"])}})
        req = {"model": model, "max_tokens": max_output_tokens,
               "messages": [{"role": "user", "content": content}],
               **self.reasoning_request(reasoning_effort)}
        if temperature is not None:
            req["temperature"] = temperature
        return req

    def content_signature(self, req):
        return [("text", p["text"]) if p["type"] == "text"
                else ("image", base64.b64decode(p["source"]["data"]))
                for p in req["messages"][0]["content"]]

    def send(self, req):
        import anthropic
        try:
            return self.client.messages.create(**req)
        except anthropic.APIStatusError as e:
            raise _status_error(e.status_code, e.message) from None
        except anthropic.APIConnectionError as e:             # includes timeouts
            raise TransientAPIError(f"{type(e).__name__}: {e}") from None

    def normalize(self, resp):
        text = "".join(b.text for b in resp.content if b.type == "text")
        refusal = None
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            refusal = (getattr(details, "explanation", None) or "") if details else ""
        u = resp.usage
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        return _normalized(
            text=text, refusal=refusal,
            status=self.STOP.get(resp.stop_reason, str(resp.stop_reason)),
            model_version=resp.model, response_id=resp.id,
            tokens_input=u.input_tokens + cache_read + cache_write,
            tokens_input_cached=cache_read, tokens_output=u.output_tokens,
            tokens_reasoning=None)   # thinking tokens are part of output_tokens, not reported apart


# =============================================================================
# Google Gemini API (google-genai SDK)
# =============================================================================
class GoogleAdapter:
    sdk = "google-genai"
    BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION",
               "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "LANGUAGE"}

    def __init__(self, provider, api_key, timeout):
        from google import genai
        from google.genai import types
        self.provider, self.types = provider, types
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=int(timeout * 1000),                               # milliseconds
            retry_options=types.HttpRetryOptions(attempts=1)))         # retries handled here

    def list_models(self):
        return {m.name.removeprefix("models/") for m in self.client.models.list()}

    def image_detail_label(self, image_detail):
        return image_detail          # media_resolution HIGH (1,120 tokens per image)

    def reasoning_request(self, effort):
        return {"thinking_config": {"thinking_level": effort.upper()}}

    def build_request(self, model, parts, *, reasoning_effort, max_output_tokens, temperature,
                      image_detail):
        t = self.types
        gparts = [t.Part.from_text(text=p["text"]) if p["type"] == "text"
                  else t.Part.from_bytes(data=p["path"].read_bytes(), mime_type="image/png")
                  for p in parts]
        config = t.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            thinking_config=t.ThinkingConfig(thinking_level=reasoning_effort.upper()),
            media_resolution=f"MEDIA_RESOLUTION_{image_detail.upper()}",
            temperature=temperature)
        return {"model": model, "contents": [t.Content(role="user", parts=gparts)],
                "config": config}

    def content_signature(self, req):
        return [("text", p.text) if p.text is not None else ("image", p.inline_data.data)
                for p in req["contents"][0].parts]

    def send(self, req):
        from google.genai import errors
        import httpx
        try:
            return self.client.models.generate_content(**req)
        except errors.APIError as e:
            raise _status_error(e.code, e.message or str(e), e.status) from None
        except (httpx.TransportError, ConnectionError, TimeoutError) as e:
            raise TransientAPIError(f"{type(e).__name__}: {e}") from None

    def normalize(self, resp):
        cand = resp.candidates[0] if resp.candidates else None
        text = ""
        if cand is not None and cand.content is not None:
            text = "".join(p.text for p in (cand.content.parts or [])
                           if p.text and not getattr(p, "thought", False))
        if cand is None:
            fb = getattr(resp, "prompt_feedback", None)
            reason = getattr(getattr(fb, "block_reason", None), "name", None) or "no_candidates"
            status = f"blocked:{reason}"
        else:
            reason = cand.finish_reason.name if cand.finish_reason is not None else "UNSPECIFIED"
            status = ("completed" if reason == "STOP" else
                      "incomplete:max_output_tokens" if reason == "MAX_TOKENS" else
                      f"blocked:{reason}" if reason in self.BLOCKED else reason.lower())
        u = resp.usage_metadata
        thoughts = (u.thoughts_token_count or 0) if u else 0
        return _normalized(
            text=text, refusal=None, status=status, model_version=resp.model_version or "",
            response_id=resp.response_id or "",
            tokens_input=(u.prompt_token_count or 0) if u else 0,
            tokens_input_cached=(u.cached_content_token_count or 0) if u else 0,
            tokens_output=((u.candidates_token_count or 0) + thoughts) if u else 0,
            tokens_reasoning=thoughts)


ADAPTERS = {"openai": OpenAIResponsesAdapter, "xai": OpenAIResponsesAdapter,
            "anthropic": AnthropicAdapter, "google": GoogleAdapter, "deepseek": OpenAIChatAdapter}


def make_adapter(provider, api_key, timeout):
    if provider not in ADAPTERS:
        raise ValueError(f"Unknown provider {provider!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[provider](provider, api_key, timeout)


def sdk_version(adapter):
    if adapter.sdk == "google-genai":
        from google.genai import version
        return version.__version__
    return __import__(adapter.sdk).__version__


def call_with_retries(adapter, req, *, max_retries, retry_waits, error_log):
    """Send one request. Returns (response, latency_sec), or (None, None) after all retries.
    Raises FatalAPIError for requests that retrying cannot fix."""
    msg = ""
    for attempt in range(max_retries + 1):
        try:
            t0 = time.monotonic()
            resp = adapter.send(req)
            return resp, time.monotonic() - t0
        except FatalAPIError as e:
            lc.append_log(error_log, f"FATAL {e}")
            raise
        except TransientAPIError as e:
            msg = str(e)
        lc.append_log(error_log, f"attempt {attempt + 1} failed: {msg}")
        if attempt < max_retries:
            wait = retry_waits[min(attempt, len(retry_waits) - 1)]
            print(f"    attempt {attempt + 1} failed ({msg}); retrying in {wait} s")
            time.sleep(wait)
    print(f"    failed after {max_retries + 1} attempts ({msg})")
    return None, None
