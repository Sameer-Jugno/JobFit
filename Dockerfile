FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY web.py ./
COPY nlp_jobmatch ./nlp_jobmatch
COPY static ./static

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn web:app --host 0.0.0.0 --port ${PORT:-8000}"]
