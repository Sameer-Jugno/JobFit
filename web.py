"""FastAPI UI and extract/match APIs for JobFit."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from nlp_jobmatch.documents import MAX_BYTES, MAX_CHARS, MIN_CHARS, DocumentError, read_upload
from nlp_jobmatch.llm import api_key_configured, load_dotenv, review_fit
from nlp_jobmatch.matcher import match_texts, requirement_gaps

load_dotenv()

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "static" / "index.html"

app = FastAPI(title="JobFit", version="0.3.0")


class MatchIn(BaseModel):
    job: str = Field(min_length=MIN_CHARS, max_length=MAX_CHARS)
    resume: str = Field(min_length=MIN_CHARS, max_length=MAX_CHARS)
    use_llm: bool = False


def _hits(items) -> list[dict]:
    return [{"term": hit.term, "weight": hit.weight} for hit in items]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX, headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def status() -> dict:
    return {"llm_ready": api_key_configured()}


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)) -> dict:
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="File is too large (max 5 MB).")
    try:
        text = read_upload(file.filename or "", data)
    except DocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "text": text,
        "filename": Path(file.filename or "").name,
        "chars": len(text),
    }


@app.post("/api/match")
def match(payload: MatchIn) -> dict:
    job = payload.job.strip()
    resume = payload.resume.strip()
    if len(job) < MIN_CHARS or len(resume) < MIN_CHARS:
        raise HTTPException(
            status_code=400,
            detail="Add more text in both boxes before analyzing.",
        )
    if job.casefold() == resume.casefold():
        raise HTTPException(
            status_code=400,
            detail="Job description and resume look the same.",
        )

    result = match_texts(job, resume)
    suggestions = [
        f"Add evidence for “{skill}” — it appears in the job but not the resume."
        for skill in result.missing_skills
    ]
    suggestions.extend(requirement_gaps(job, resume))

    llm_used = False
    llm_summary = None
    llm_error = None
    if payload.use_llm:
        review = review_fit(job, resume, suggestions, result.missing_skills)
        llm_error = review.error
        if not review.error:
            llm_used = True
            llm_summary = review.summary or None
            suggestions.extend(review.gaps)

    return {
        "overall_score": result.overall_score,
        "similarity": result.similarity,
        "skill_coverage": round(result.skill_coverage, 4),
        "matched_skills": result.matched_skills,
        "missing_skills": result.missing_skills,
        "extra_skills": result.extra_skills,
        "shared_keywords": _hits(result.shared_keywords),
        "job_keywords": _hits(result.job_keywords),
        "resume_keywords": _hits(result.resume_keywords),
        "suggestions": suggestions,
        "llm_used": llm_used,
        "llm_summary": llm_summary,
        "llm_error": llm_error,
    }
