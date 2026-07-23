"""Compare a job description against a resume with skills + TF-IDF."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp_jobmatch.preprocess import normalize, word_count
from nlp_jobmatch.skills import SKILLS, extract_skills

SKILL_WEIGHT = 0.7
TEXT_WEIGHT = 0.3

# Generic filler only — no names, cities, companies, or job-specific terms.
_NOISE = {
    "based",
    "best",
    "come",
    "doing",
    "era",
    "experience",
    "good",
    "help",
    "join",
    "love",
    "need",
    "new",
    "please",
    "present",
    "real",
    "role",
    "see",
    "strong",
    "team",
    "today",
    "using",
    "ways",
    "work",
    "world",
}

_DATE = re.compile(r"^\d{1,4}$")


@dataclass(frozen=True)
class KeywordHit:
    term: str
    weight: float


@dataclass(frozen=True)
class MatchResult:
    similarity: float
    overall_score: float
    verdict: str
    job_skills: list[str]
    resume_skills: list[str]
    matched_skills: list[str]
    missing_skills: list[str]
    extra_skills: list[str]
    job_words: int
    resume_words: int
    shared_keywords: list[KeywordHit] = field(default_factory=list)
    job_keywords: list[KeywordHit] = field(default_factory=list)
    resume_keywords: list[KeywordHit] = field(default_factory=list)

    @property
    def skill_coverage(self) -> float:
        if not self.job_skills:
            return 0.0
        return len(self.matched_skills) / len(self.job_skills)


def match_texts(job_text: str, resume_text: str) -> MatchResult:
    job_skills = extract_skills(job_text)
    resume_skills = extract_skills(resume_text)
    job_set = set(job_skills)
    resume_set = set(resume_skills)

    matched = [skill for skill in job_skills if skill in resume_set]
    missing = [skill for skill in job_skills if skill not in resume_set]
    extra = [skill for skill in resume_skills if skill not in job_set]
    similarity = _tfidf_similarity(job_text, resume_text)
    coverage = (len(matched) / len(job_skills)) if job_skills else 0.0
    overall = round(SKILL_WEIGHT * coverage + TEXT_WEIGHT * similarity, 4)
    shared, job_only, resume_only = _keyword_breakdown(job_text, resume_text)

    return MatchResult(
        similarity=similarity,
        overall_score=overall,
        verdict=_verdict(overall),
        job_skills=job_skills,
        resume_skills=resume_skills,
        matched_skills=matched,
        missing_skills=missing,
        extra_skills=extra,
        job_words=word_count(job_text),
        resume_words=word_count(resume_text),
        shared_keywords=shared,
        job_keywords=job_only,
        resume_keywords=resume_only,
    )


def requirement_gaps(job_text: str, resume_text: str) -> list[str]:
    """Human-readable gaps beyond the skill dictionary."""
    gaps: list[str] = []
    job_n = f" {normalize(job_text)} "
    resume_n = f" {normalize(resume_text)} "

    wants_phd = bool(re.search(r"\b(phd|ph d)\b", job_n))
    wants_ms = bool(re.search(r"\b(ms or phd|pursuing a ms|masters?|master s)\b", job_n))
    accepts_bachelor = bool(re.search(r"\bbachelor", job_n))
    has_grad = bool(re.search(r"\b(phd|masters?|master s|ms in)\b", resume_n))
    if (wants_phd or wants_ms) and not accepts_bachelor and not has_grad:
        gaps.append("Job asks for an MS or PhD; the resume does not list a graduate degree.")

    wants_oss = bool(re.search(r"\b(open source|open-source|oss contributions?)\b", job_n))
    has_oss = bool(re.search(r"\b(open source|open-source|oss contribution|github com)\b", resume_n))
    if wants_oss and not has_oss:
        gaps.append("Job asks for open-source contributions; none are listed on the resume.")

    wants_pubs = bool(re.search(r"\b(publications?|published|arxiv)\b", job_n))
    has_pubs = bool(re.search(r"\b(publications?|published|paper|arxiv)\b", resume_n))
    if wants_pubs and not has_pubs:
        gaps.append("Job mentions a publication record; the resume does not list papers.")

    if not extract_skills(job_text):
        gaps.append("No catalog skills were found in the job description, so skill coverage may be incomplete.")

    return gaps


def _verdict(overall: float) -> str:
    if overall >= 0.7:
        return "Strong match"
    if overall >= 0.4:
        return "Partial match"
    return "Weak match"


def _tfidf_similarity(job_text: str, resume_text: str) -> float:
    docs = [normalize(job_text), normalize(resume_text)]
    if not docs[0] or not docs[1]:
        return 0.0
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    matrix = vectorizer.fit_transform(docs)
    full = float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])

    skill_docs = [" ".join(extract_skills(job_text)), " ".join(extract_skills(resume_text))]
    if skill_docs[0] and skill_docs[1]:
        skill_matrix = TfidfVectorizer().fit_transform(skill_docs)
        skill = float(cosine_similarity(skill_matrix[0:1], skill_matrix[1:2])[0, 0])
        return round(0.4 * full + 0.6 * skill, 4)
    return round(full, 4)


def _keyword_breakdown(
    job_text: str,
    resume_text: str,
    top_k: int = 8,
) -> tuple[list[KeywordHit], list[KeywordHit], list[KeywordHit]]:
    docs = [normalize(job_text), normalize(resume_text)]
    if not docs[0] or not docs[1]:
        return [], [], []

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    matrix = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()
    job_w = matrix[0].toarray().ravel()
    resume_w = matrix[1].toarray().ravel()

    shared: list[KeywordHit] = []
    job_only: list[KeywordHit] = []
    resume_only: list[KeywordHit] = []

    for term, job_score, resume_score in zip(terms, job_w, resume_w):
        if _is_noise(term):
            continue
        hit = KeywordHit(term, round(float(max(job_score, resume_score)), 4))
        if job_score > 0 and resume_score > 0:
            shared.append(KeywordHit(term, round(float(min(job_score, resume_score)), 4)))
        elif job_score > 0:
            job_only.append(hit)
        else:
            resume_only.append(hit)

    return (
        _trim(shared, top_k),
        _trim(job_only, top_k),
        _trim(resume_only, top_k),
    )


_PHONE = re.compile(r"\d{5,}")


def _is_noise(term: str) -> bool:
    tokens = term.split()
    if any(token in _NOISE or _DATE.match(token) or _PHONE.search(token) for token in tokens):
        return True
    if len(term) <= 2:
        return True
    # Unigrams must be catalog skills; random words like "skills" or "students" drop out.
    if len(tokens) == 1 and term not in SKILLS:
        return True
    return False


def _trim(hits: list[KeywordHit], top_k: int) -> list[KeywordHit]:
    hits.sort(key=lambda hit: (len(hit.term.split()), hit.weight), reverse=True)
    kept: list[KeywordHit] = []
    for hit in hits:
        if any(hit.term != other.term and hit.term in other.term for other in kept):
            continue
        kept.append(hit)
        if len(kept) == top_k:
            break
    return kept
