"""API tests for Medhā FastAPI endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_api_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "phase" in data


def test_api_list_sessions() -> None:
    response = client.get("/api/sessions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_api_search_existing_session() -> None:
    response = client.post(
        "/api/search",
        json={
            "session_id": "3ec404e4f350",
            "query": "Where does the instructor explain loops?",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "3ec404e4f350"
    assert data["total_results"] == 3
    assert len(data["results"]) == 3

    top = data["results"][0]
    assert "chunk_id" in top
    assert "text" in top
    assert "start_time" in top
    assert "end_time" in top
    assert "timestamp" in top
    assert "score" in top
    assert top["score"] > 0.0


def test_api_search_nonexistent_session() -> None:
    response = client.post(
        "/api/search",
        json={
            "session_id": "does_not_exist_999",
            "query": "Where does the instructor explain loops?",
            "top_k": 3,
        },
    )
    assert response.status_code == 404


def test_api_search_empty_query() -> None:
    response = client.post(
        "/api/search",
        json={
            "session_id": "3ec404e4f350",
            "query": "   ",
            "top_k": 3,
        },
    )
    assert response.status_code in (400, 422)


def test_api_get_session_topics() -> None:
    response = client.get("/api/sessions/3ec404e4f350/topics")
    assert response.status_code == 200
    topics = response.json()
    assert isinstance(topics, list)
    assert len(topics) > 0
    first = topics[0]
    assert "topic_id" in first
    assert "name" in first
    assert "start_time" in first
    assert "end_time" in first
    assert "timestamp_range" in first
    assert "chunk_ids" in first
