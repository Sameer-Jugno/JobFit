import json

from nlp_jobmatch.llm import LlmReview, _parse_response, review_fit


def test_parse_ignores_non_list_gaps():
    raw = {
        "candidates": [
            {"content": {"parts": [{"text": '{"summary": "Partial fit.", "gaps": "not a list"}'}]}}
        ]
    }
    parsed = _parse_response(raw)
    assert parsed["summary"] == "Partial fit."
    assert parsed["gaps"] == []


def test_parse_empty_response_raises():
    try:
        _parse_response({"candidates": [{"content": {"parts": [{"text": ""}]}}]})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "empty" in str(exc).lower()


def test_parse_plain_and_fenced_json():
    payload = {"summary": "Partial match.", "gaps": ["Needs a graduate degree."]}
    plain = {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}
    fenced = {
        "candidates": [
            {"content": {"parts": [{"text": "```json\n" + json.dumps(payload) + "\n```"}]}}
        ]
    }
    assert _parse_response(plain) == payload
    assert _parse_response(fenced) == payload


def test_review_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    review = review_fit("Need Python for this backend role.", "Python intern.", [], ["python"])
    assert review == LlmReview("", [], "Gemini is off: set GEMINI_API_KEY to enable it.")


def test_review_uses_mock_and_skips_known_gaps(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake_post(prompt, api_key):
        assert "JOB:" in prompt
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "summary": "Resume has Python.",
                                        "gaps": [
                                            "Job asks for an MS or PhD; the resume does not list a graduate degree.",
                                            "Job wants 3 years of experience.",
                                        ],
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }

    known = ["Job asks for an MS or PhD; the resume does not list a graduate degree."]
    review = review_fit("Need Python and an MS.", "Python intern.", known, ["ms"], post=fake_post)
    assert review.error is None
    assert review.summary == "Resume has Python."
    assert review.gaps == ["Job wants 3 years of experience."]


def test_review_maps_http_failures(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake_post(prompt, api_key):
        raise RuntimeError("Gemini model was not found. Set GEMINI_MODEL in .env (try gemini-3.6-flash).")

    review = review_fit("Need Python for this backend role.", "Python intern.", [], ["python"], post=fake_post)
    assert "model was not found" in review.error

