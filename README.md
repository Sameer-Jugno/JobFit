# JobFit

Compare a job description and a resume. Paste text, or upload a PDF / TXT / MD file.

Matching score, skills, and keywords come from a local skill catalog plus TF-IDF. Gemini assist is optional: it only adds a short fit summary and extra qualitative gaps. It does not change the score.

## Setup

```bash
cd JobFit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m uvicorn web:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

## Tests

```bash
pytest -q
```

## Limits

- Uploads: `.pdf`, `.txt`, `.md`, max 5 MB, 30 PDF pages
- Pasted text: 20–80,000 characters
- Skills come from a generic catalog covering common AI/ML, GenAI, web, DevOps, data, and cloud roles — not a specific resume or job

## Optional Gemini assist

Off by default. When you turn it on, JobFit sends the job description and resume to Google Gemini. The UI shows a warning first. If Gemini fails, dictionary results still return.

Get a key from [Google AI Studio](https://aistudio.google.com/apikey):

```bash
cp .env.example .env
# put GEMINI_API_KEY=... in .env
```

Restart the server after setting the key. Default model is `gemini-3.6-flash`.

## Docker

```bash
docker build -t jobfit .
docker run --rm -p 8000:8000 --env-file .env jobfit
```

`--env-file .env` is optional. Without a Gemini key the matcher still runs; Gemini assist stays off.
