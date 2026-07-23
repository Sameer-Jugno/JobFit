"""Canonical skills plus aliases for common software, AI, web, and DevOps jobs."""

from __future__ import annotations

import re

from nlp_jobmatch.preprocess import normalize

# Longer aliases must be listed first inside each group.
# This is a generic catalog, not a specific job or resume.
SKILL_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # --- AI / ML ---
    (
        "nlp",
        (
            "natural language processing",
            "large language models",
            "large language model",
            "language models",
            "language model",
            "chatbot",
            "llms",
            "llm",
            "nlp",
        ),
    ),
    (
        "genai",
        (
            "generative ai",
            "gen ai",
            "prompt engineering",
            "openai",
            "anthropic",
        ),
    ),
    (
        "agentic ai",
        (
            "agentic ai",
            "ai agents",
            "multi agent",
            "langgraph",
            "langchain",
            "llamaindex",
            "llama index",
            "crewai",
            "autogen",
            "mcp",
        ),
    ),
    (
        "rag",
        (
            "retrieval augmented generation",
            "retrieval augmented",
            "vector database",
            "vector db",
            "qdrant",
            "pinecone",
            "chromadb",
            "chroma",
            "weaviate",
            "faiss",
            "rag",
        ),
    ),
    (
        "deep learning",
        (
            "deep learning",
            "neural networks",
            "neural network",
            "transformers",
            "huggingface",
            "hugging face",
            "convolutional",
            "cnn",
            "gnn",
        ),
    ),
    ("machine learning", ("machine learning", "scikit-learn", "sklearn")),
    ("mlops", ("mlops", "mlflow", "wandb", "weights and biases")),
    ("computer vision", ("computer vision", "opencv", "image processing", "yolo")),
    ("data analysis", ("data analysis", "data-driven", "pandas", "numpy")),
    ("pytorch", ("pytorch", "torch")),
    ("tensorflow", ("tensorflow", "keras")),
    ("jax", ("jax",)),
    ("spacy", ("spacy",)),
    ("nltk", ("nltk",)),
    # --- Languages ---
    ("python", ("python",)),
    ("javascript", ("javascript",)),
    ("typescript", ("typescript",)),
    ("java", ("java",)),
    ("csharp", ("c#", "csharp", "dotnet", "asp.net", "asp net")),
    ("cpp", ("c++", "cpp")),
    ("golang", ("golang", "go lang")),
    ("rust", ("rust",)),
    ("kotlin", ("kotlin",)),
    ("swift", ("swift",)),
    ("php", ("php",)),
    ("ruby", ("ruby",)),
    ("scala", ("scala",)),
    ("sql", ("postgresql", "postgres", "mysql", "sql server", "sqlite", "sql")),
    # --- Web ---
    ("react", ("react native", "reactjs", "react.js", "react js", "react")),
    ("nextjs", ("next.js", "nextjs", "next js")),
    ("vue", ("vue.js", "vuejs", "vue")),
    ("angular", ("angular",)),
    ("nodejs", ("node.js", "nodejs", "node js")),
    ("express", ("express.js", "expressjs", "express js")),
    ("nestjs", ("nestjs", "nest.js")),
    ("django", ("django",)),
    ("flask", ("flask",)),
    ("fastapi", ("fastapi",)),
    ("spring", ("spring boot", "spring")),
    ("rails", ("ruby on rails", "rails")),
    ("graphql", ("graphql",)),
    ("rest api", ("rest apis", "rest api", "restful")),
    ("html", ("html",)),
    ("css", ("tailwind", "css")),
    # --- DevOps / Cloud ---
    ("docker", ("docker", "containers")),
    ("kubernetes", ("kubernetes", "k8s")),
    ("terraform", ("terraform", "iac")),
    ("ansible", ("ansible",)),
    ("ci cd", ("github actions", "gitlab ci", "circleci", "jenkins", "ci/cd", "ci cd", "cicd")),
    ("aws", ("aws", "amazon web services")),
    ("gcp", ("google cloud", "gcp")),
    ("azure", ("azure",)),
    ("linux", ("linux",)),
    ("git", ("github", "gitlab", "git")),
    ("prometheus", ("prometheus",)),
    ("grafana", ("grafana",)),
    ("helm", ("helm",)),
    # --- Data ---
    ("spark", ("pyspark", "spark")),
    ("airflow", ("airflow",)),
    ("kafka", ("kafka",)),
    ("dbt", ("dbt",)),
    ("snowflake", ("snowflake",)),
    ("mongodb", ("mongodb", "mongo")),
    ("redis", ("redis",)),
    ("elasticsearch", ("elasticsearch", "elastic search")),
    ("excel", ("excel",)),
    ("tableau", ("tableau",)),
    ("power bi", ("power bi", "powerbi")),
    # --- Testing ---
    ("pytest", ("pytest",)),
    ("jest", ("jest",)),
    ("cypress", ("cypress",)),
    ("selenium", ("selenium",)),
    # --- Hardware / embedded (still generic tools) ---
    ("cadence", ("cadence",)),
    ("orcad", ("orcad",)),
    ("multisim", ("multisim",)),
)

SKILLS: tuple[str, ...] = tuple(canonical for canonical, _aliases in SKILL_GROUPS)


def extract_skills(text: str) -> list[str]:
    """Return canonical skills found in `text`, without duplicates."""
    haystack = f" {normalize(text)} "
    found: list[str] = []
    seen: set[str] = set()
    used_spans: list[tuple[int, int]] = []

    for alias, canonical in _alias_table():
        pattern = rf"(?<![\w+#]){re.escape(alias)}(?![\w+#])"
        for match in re.finditer(pattern, haystack):
            span = match.span()
            if any(not (span[1] <= start or span[0] >= end) for start, end in used_spans):
                continue
            used_spans.append(span)
            if canonical not in seen:
                found.append(canonical)
                seen.add(canonical)
            break
    return found


def _alias_table() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for canonical, aliases in SKILL_GROUPS:
        for alias in aliases:
            key = normalize(alias)
            if not key or (key, canonical) in seen:
                continue
            seen.add((key, canonical))
            pairs.append((key, canonical))
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return pairs
