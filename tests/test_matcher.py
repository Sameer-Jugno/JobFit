from nlp_jobmatch.matcher import match_texts, requirement_gaps
from nlp_jobmatch.skills import extract_skills


def test_extracts_multiword_and_single_skills():
    text = "I use Python, scikit-learn, and machine learning for NLP."
    skills = extract_skills(text)
    assert "scikit-learn" not in skills or "machine learning" in skills
    assert "python" in skills
    assert "machine learning" in skills
    assert "nlp" in skills


def test_aliases_map_cnn_and_llm_to_real_skills():
    skills = extract_skills(
        "Built a CNN-based model and fine-tuned a large language model chatbot."
    )
    assert "deep learning" in skills
    assert "nlp" in skills
    assert skills.count("nlp") == 1


def test_strong_resume_covers_most_job_skills():
    job = """
    Need Python, SQL, PyTorch, Docker, and Git.
    Machine learning experience required.
    """
    resume = """
    Python developer. Built PyTorch models, wrote SQL, used Docker and Git.
    Machine learning intern.
    """
    result = match_texts(job, resume)
    assert result.skill_coverage >= 0.8
    assert "python" in result.matched_skills
    assert result.similarity > 0.1
    assert result.overall_score >= result.skill_coverage * 0.5
    assert result.verdict in {"Strong match", "Partial match", "Weak match"}
    assert result.shared_keywords


def test_unrelated_resume_is_missing_job_skills():
    job = "Hiring for PyTorch, TensorFlow, and Kubernetes."
    resume = "Retail store manager. Excel and customer service."
    result = match_texts(job, resume)
    assert "pytorch" in result.missing_skills
    assert result.skill_coverage < 0.5
    assert match_texts(job, job).similarity > result.similarity


def test_related_terms_count_as_the_canonical_skill():
    job = """
    NLP intern. Strong knowledge in Deep Learning for Natural Language Processing.
    Experience with PyTorch or JAX. Pursuing a MS or PhD.
    Contribution to open-source projects. Excellent Python programming skills.
    """
    resume = """
    Python, SQL, Django, PyTorch, Git.
    Evaluated CNN and GNN designs. Assessed LLM performance.
    Fine-tuned a pre-trained large language model. CNN-based traffic signs.
    BS in Computer Science. Worked in 01/2024.
    """
    result = match_texts(job, resume)
    assert "python" in result.matched_skills
    assert "pytorch" in result.matched_skills
    assert "nlp" in result.matched_skills
    assert "deep learning" in result.matched_skills
    assert "nlp" not in result.missing_skills
    assert "deep learning" not in result.missing_skills
    assert "jax" in result.missing_skills
    keywords = {hit.term for hit in result.shared_keywords + result.job_keywords + result.resume_keywords}
    assert "01" not in keywords
    assert "using" not in keywords
    gaps = requirement_gaps(job, resume)
    assert any("MS or PhD" in item for item in gaps)
    assert any("open-source" in item for item in gaps)


def test_bachelors_or_masters_is_not_a_phd_requirement():
    job = "Completing your bachelor's or master's degree in Computer Science. Possible to join now."
    resume = "BS in Computer Science. Python and Django."
    gaps = requirement_gaps(job, resume)
    assert not any("MS or PhD" in item for item in gaps)
    assert not any("open-source" in item for item in gaps)


def test_github_as_a_tool_is_not_open_source_evidence():
    job = "Please include links to your open-source contributions."
    resume = "Used GitHub and Git for version control. Python developer."
    gaps = requirement_gaps(job, resume)
    assert any("open-source" in item for item in gaps)


def test_github_profile_url_counts_as_open_source_evidence():
    job = "Please include links to your open-source contributions."
    resume = "Python developer. github.com/example/project"
    gaps = requirement_gaps(job, resume)
    assert not any("open-source" in item for item in gaps)


def test_keyword_tab_drops_generic_unigrams():
    job = "We need Python students in Lahore Pakistan with leadership skills."
    resume = "Python intern in Lahore. Leadership skills and students club."
    result = match_texts(job, resume)
    keywords = {hit.term for hit in result.shared_keywords + result.job_keywords + result.resume_keywords}
    assert "python" in keywords
    assert "lahore" not in keywords
    assert "pakistan" not in keywords
    assert "skills" not in keywords
    assert "students" not in keywords


def test_empty_job_skills_are_called_out():
    job = "We need a creative storyteller with a can-do attitude and leadership."
    resume = "Python and Django developer."
    gaps = requirement_gaps(job, resume)
    assert any("catalog skills" in item.lower() for item in gaps)


def test_hardware_jd_marks_design_tools_missing_for_software_resume():
    job = "System Design Engineer intern. Cadence, OrCAD, and Multisim required."
    resume = "AI Engineer. Python, PyTorch, and Django."
    result = match_texts(job, resume)
    assert "cadence" in result.missing_skills
    assert "orcad" in result.missing_skills
    assert "python" in result.extra_skills
    assert result.skill_coverage == 0


def test_covers_web_devops_and_genai_job_families():
    web = extract_skills("React, Next.js, Node.js, TypeScript, PostgreSQL, REST APIs")
    devops = extract_skills("AWS, Kubernetes, Terraform, Docker, GitHub Actions, Linux")
    genai = extract_skills("LangChain, RAG, Qdrant, prompt engineering, agentic AI")
    assert {"react", "nextjs", "nodejs", "typescript", "sql", "rest api"} <= set(web)
    assert {"aws", "kubernetes", "terraform", "docker", "ci cd", "linux"} <= set(devops)
    assert {"agentic ai", "rag", "genai"} <= set(genai)

