"""
Shared test setup. Tests never touch the internet or a real key: the Gemini and Claude SDKs are
pointed at tests/fake_llm.py through the *_BASE_URL hooks in llm_client.py.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Hermetic environment: set BEFORE llm_client / app are imported (load_dotenv never overrides these).
for name in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "DEFAULT_LLM_PROVIDER", "GEMINI_MODELS_FAST",
             "GEMINI_MODELS_BALANCED", "GEMINI_MODELS_BEST", "CLAUDE_MODELS_FAST", "CLAUDE_MODELS_BALANCED",
             "CLAUDE_MODELS_BEST", "APP_ENV", "RENDER", "TRUST_PROXY"):
    os.environ[name] = ""
os.environ["FLASK_SECRET_KEY"] = "test-secret-key-0123456789abcdef"
os.environ["LOG_LEVEL"] = "WARNING"

from fake_llm import FakeLLMServer  # noqa: E402

GEMINI_KEY = "AIzaSyFAKE_gemini_key_for_tests_0123456789"
CLAUDE_KEY = "sk-ant-api03-FAKE_claude_key_for_tests_0123456789"
GEMINI_HEADERS = {"X-Gemini-Api-Key": GEMINI_KEY}
CLAUDE_HEADERS = {"X-Anthropic-Api-Key": CLAUDE_KEY}

ANSWERS = [
    "BITS Pilani",
    "India",
    "B.Tech Computer Science",
    "I sleep around 11:30pm and wake up at 6:30am",
    "Breakfast at 7:30, lunch at 1pm, dinner at 8pm. Two cups of coffee in the morning.",
    "Mornings are my best time, about 5 hours a day",
    "Data Structures exam on 2026-11-20",
]


@pytest.fixture(scope="session")
def fake_llm():
    server = FakeLLMServer().start()
    os.environ["GEMINI_BASE_URL"] = server.base_url
    os.environ["ANTHROPIC_BASE_URL"] = server.base_url
    yield server
    server.stop()


@pytest.fixture(autouse=True)
def fresh_state(request, monkeypatch):
    """Every test starts with a clean fake provider, empty model-health caches and no real sleeping."""
    import llm_client
    llm_client.reset_caches()
    monkeypatch.setattr(llm_client, "_sleep", lambda seconds: None)
    if "fake_llm" in request.fixturenames:
        request.getfixturevalue("fake_llm").reset()


@pytest.fixture()
def gemini_creds():
    import llm_client
    return llm_client.Credentials(llm_client.GEMINI, GEMINI_KEY, "user")


@pytest.fixture()
def claude_creds():
    import llm_client
    return llm_client.Credentials(llm_client.CLAUDE, CLAUDE_KEY, "user")


@pytest.fixture()
def app_module(monkeypatch):
    import app as app_module
    from sessions import RateLimiter, SessionStore
    monkeypatch.setattr(app_module, "STORE", SessionStore())
    monkeypatch.setattr(app_module, "RATE", RateLimiter())
    return app_module


@pytest.fixture()
def client(app_module, fake_llm):
    return app_module.app.test_client()


def answer_all(client, headers, answers=ANSWERS):
    """Send the seven answers; returns the list of responses."""
    return [client.post("/api/chat", json={"message": a}, headers=headers) for a in answers]


def wait_for_schedule(client, timeout: float = 30.0):
    """Poll GET /api/schedule the way the results page does until the build settles."""
    deadline = time.monotonic() + timeout
    data = client.get("/api/schedule").get_json()
    while data["status"] in ("pending", "generating") and time.monotonic() < deadline:
        time.sleep(0.05)
        data = client.get("/api/schedule").get_json()
    return data
