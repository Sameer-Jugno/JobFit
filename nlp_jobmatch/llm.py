"""Optional Gemini overlay for qualitative gaps. Scores stay in the dictionary matcher."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
MAX_INPUT_CHARS = 6_000
TIMEOUT_SECONDS = 25
DEFAULT_MODEL = "gemini-3.6-flash"
_MODEL_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class LlmReview:
    summary: str
    gaps: list[str]
    error: str | None = None


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or (ROOT / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def api_key_configured() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def review_fit(
    job_text: str,
    resume_text: str,
    known_gaps: list[str],
    missing_skills: list[str],
    *,
    post=None,
) -> LlmReview:
    """Ask Gemini for a short summary and qualitative gaps the dictionary missed."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return LlmReview("", [], "Gemini is off: set GEMINI_API_KEY to enable it.")

    prompt = _prompt(job_text, resume_text, known_gaps, missing_skills)
    try:
        raw = (post or _post_gemini)(prompt, api_key)
        parsed = _parse_response(raw)
    except TimeoutError:
        return LlmReview("", [], "Gemini timed out. Dictionary results are unchanged.")
    except RuntimeError as exc:
        return LlmReview("", [], str(exc))
    except Exception:
        return LlmReview("", [], "Gemini could not review this pair. Dictionary results are unchanged.")

    summary = _clean_text(parsed.get("summary"), 400)
    gaps = []
    seen = {_norm(item) for item in known_gaps}
    raw_gaps = parsed.get("gaps")
    if not isinstance(raw_gaps, list):
        raw_gaps = []
    for item in raw_gaps:
        text = _clean_text(item, 240)
        key = _norm(text)
        if not text or key in seen:
            continue
        seen.add(key)
        gaps.append(text)
        if len(gaps) == 5:
            break
    if not summary and not gaps:
        return LlmReview("", [], "Gemini returned nothing useful.")
    return LlmReview(summary, gaps)


def _prompt(job_text: str, resume_text: str, known_gaps: list[str], missing_skills: list[str]) -> str:
    known = "\n".join(f"- {item}" for item in known_gaps) or "- none"
    missing = ", ".join(missing_skills) or "none"
    return (
        "Compare this job description and resume. "
        "The app already scored skills with a dictionary. Do not invent facts.\n"
        "Return JSON only: {\"summary\": string, \"gaps\": string[]}\n"
        "summary: two short sentences about fit, using only the texts.\n"
        "gaps: at most 5 qualitative gaps the dictionary likely missed "
        "(years of experience, education, domain, certifications, publications, "
        "open-source, work authorization). Empty array if none. "
        "Do not repeat known gaps or missing skills.\n\n"
        f"Known gaps:\n{known}\n"
        f"Missing skills already listed: {missing}\n\n"
        f"JOB:\n{_clip(job_text)}\n\n"
        f"RESUME:\n{_clip(resume_text)}\n"
    )


def _post_gemini(prompt: str, api_key: str) -> dict:
    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if not _MODEL_NAME.fullmatch(model):
        model = DEFAULT_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024,
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingLevel": "minimal"},
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise TimeoutError("Gemini timed out") from exc
    except urllib.error.HTTPError as exc:
        exc.read()
        raise RuntimeError(_http_error_message(exc)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not reach Gemini. Dictionary results are unchanged.") from exc


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    if exc.code in {401, 403}:
        return "Gemini rejected the API key. Check GEMINI_API_KEY in .env."
    if exc.code == 404:
        return "Gemini model was not found. Set GEMINI_MODEL in .env (try gemini-3.6-flash)."
    if exc.code == 429:
        return "Gemini rate limit reached. Try again in a minute."
    if exc.code >= 500:
        return "Gemini is temporarily unavailable. Dictionary results are unchanged."
    return "Gemini could not review this pair. Dictionary results are unchanged."


def _parse_response(raw: dict) -> dict:
    text = ""
    for candidate in raw.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            text += part.get("text") or ""
    text = text.strip()
    if not text:
        raise ValueError("Gemini returned an empty response")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Gemini JSON was not an object")
    if "summary" in data and data["summary"] is not None and not isinstance(data["summary"], str):
        data["summary"] = str(data["summary"])
    if "gaps" in data and not isinstance(data["gaps"], list):
        data["gaps"] = []
    return data


def _clip(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_INPUT_CHARS:
        return text
    return text[:MAX_INPUT_CHARS] + "\n[truncated]"


def _clean_text(value: object, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
