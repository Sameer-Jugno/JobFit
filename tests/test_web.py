from fastapi.testclient import TestClient

from nlp_jobmatch.documents import DocumentError, read_upload
from web import app

client = TestClient(app)

MINIMAL_PDF = b"""%PDF-1.1
1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj
2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj
3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>endobj
4 0 obj<< /Length 68 >>stream
BT /F1 12 Tf 72 720 Td (Python SQL Docker machine learning engineer) Tj ET
endstream
endobj
5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000384 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
456
%%EOF
"""

JOB = "Need Python, SQL, Docker, and machine learning for this role."
RESUME = "Python developer. SQL and Docker. Machine learning intern on NLP."


def test_index_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "JobFit" in response.text
    assert "no-store" in response.headers.get("cache-control", "").lower()


def test_match_endpoint_returns_analysis():
    response = client.post("/api/match", json={"job": JOB, "resume": RESUME})
    assert response.status_code == 200
    data = response.json()
    assert data["skill_coverage"] >= 0.75
    assert "python" in data["matched_skills"]
    assert "suggestions" in data
    assert data["llm_used"] is False
    assert data["llm_summary"] is None
    assert "job_words" not in data


def test_match_rejects_short_text():
    response = client.post("/api/match", json={"job": "hi", "resume": "there"})
    assert response.status_code == 422


def test_match_rejects_identical_texts():
    response = client.post("/api/match", json={"job": JOB, "resume": JOB})
    assert response.status_code == 400
    assert "same" in response.json()["detail"].lower()


def test_extract_txt_upload():
    response = client.post(
        "/api/extract",
        files={"file": ("resume.txt", RESUME.encode(), "text/plain")},
    )
    assert response.status_code == 200
    assert "Python developer" in response.json()["text"]


def test_extract_rejects_unsupported_type():
    response = client.post(
        "/api/extract",
        files={"file": ("photo.png", b"not-a-pdf", "image/png")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_extract_rejects_binary_text():
    response = client.post(
        "/api/extract",
        files={"file": ("resume.txt", b"hello\x00world and more text here", "text/plain")},
    )
    assert response.status_code == 400


def test_extract_rejects_fake_pdf():
    response = client.post(
        "/api/extract",
        files={"file": ("job.pdf", b"this is not a pdf but named like one" + b" x" * 20, "application/pdf")},
    )
    assert response.status_code == 400


def test_read_upload_rejects_empty():
    try:
        read_upload("job.txt", b"")
        assert False, "expected DocumentError"
    except DocumentError as exc:
        assert "empty" in str(exc).lower()


def test_read_upload_strips_path_from_filename():
    text = read_upload("../../resume.txt", RESUME.encode())
    assert "Python developer" in text


def test_extract_pdf_upload():
    text = read_upload("job.pdf", MINIMAL_PDF)
    assert "python" in text.lower()
    response = client.post(
        "/api/extract",
        files={"file": ("job.pdf", MINIMAL_PDF, "application/pdf")},
    )
    assert response.status_code == 200
    assert "python" in response.json()["text"].lower()


def test_status_reports_llm_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert client.get("/api/status").json()["llm_ready"] is False
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert client.get("/api/status").json()["llm_ready"] is True


def test_match_llm_without_key_keeps_dictionary_result(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post(
        "/api/match",
        json={"job": JOB, "resume": RESUME, "use_llm": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["llm_used"] is False
    assert data["llm_error"]
    assert data["skill_coverage"] >= 0.75


def test_match_llm_adds_qualitative_gap(monkeypatch):
    def fake_post(prompt, api_key):
        assert api_key == "test-key"
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"summary": "Good Python overlap.", "gaps": ["Job wants 3 years; resume is internship-level."]}'
                            }
                        ]
                    }
                }
            ]
        }

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("nlp_jobmatch.llm._post_gemini", fake_post)
    response = client.post(
        "/api/match",
        json={"job": JOB, "resume": RESUME, "use_llm": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["llm_used"] is True
    assert data["llm_summary"] == "Good Python overlap."
    assert any("3 years" in item for item in data["suggestions"])
    assert data["skill_coverage"] >= 0.75
