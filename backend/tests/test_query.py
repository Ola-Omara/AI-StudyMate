import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.retrieval import RetrievalContext


class FakeCollection:
    def query(self, query_embeddings, n_results):
        return {"ids": [["adam_optimizer_p2_c0"]]}


class FakeEmbeddingModel:
    def encode(self, texts):
        return np.array([[1.0, 0.0, 0.0] for _ in texts])


class FakeBM25Index:
    def get_scores(self, query_tokens):
        return [1.0]


def build_fake_retrieval_context() -> RetrievalContext:
    chunk_id = "adam_optimizer_p2_c0"
    return RetrievalContext(
        embedding_model=FakeEmbeddingModel(),
        collection=FakeCollection(),
        bm25_index=FakeBM25Index(),
        chunk_ids=[chunk_id],
        chunk_id_to_text={
            chunk_id: "Adam combines momentum and adaptive learning rates using moment estimates."
        },
        chunk_id_to_metadata={
            chunk_id: {
                "source_id": "adam_optimizer",
                "title": "Adam: A Method for Stochastic Optimization",
                "authors": "Kingma, Ba",
                "page": "2",
                "section": "Algorithm",
                "source_url": "https://arxiv.org/pdf/1412.6980",
                "topic": "Optimization, gradient descent, Adam",
            }
        },
        chunk_id_to_embedding={chunk_id: [1.0, 0.0, 0.0]},
    )


class FakeOllamaClient:
    def chat(self, model, messages, options=None):
        system_content = messages[0]["content"]
        if "intent classifier" in system_content.lower():
            return {"message": {"content": "IN_SCOPE"}}
        return {
            "message": {
                "content": "Adam differs from SGD by using adaptive per-parameter learning rates [S1]."
            }
        }


@pytest.fixture
def test_client():
    settings = Settings(
        vector_store_path="/tmp/does-not-need-to-exist",
        ollama_host="http://fake-ollama:11434",
        frontend_origin="http://localhost:8501",
    )
    app = create_app(
        settings=settings,
        retrieval_context=build_fake_retrieval_context(),
        ollama_client=FakeOllamaClient(),
    )
    with TestClient(app) as client:
        yield client


def test_query_happy_path(test_client):
    response = test_client.post("/query", json={"question": "How does the Adam optimizer differ from SGD?"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "in_scope"
    assert body["is_refusal"] is False
    assert "Adam: A Method for Stochastic Optimization" in body["sources"]
    assert "[S1]" not in body["answer"]
    assert "Adam: A Method for Stochastic Optimization, page 2, Section: Algorithm" in body["answer"]


def test_query_invalid_input_returns_422(test_client):
    response = test_client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_missing_field_returns_422(test_client):
    response = test_client.post("/query", json={})
    assert response.status_code == 422


def test_health_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunk_count"] == 1


def test_chitchat_bypasses_retrieval(test_client):
    response = test_client.post("/query", json={"question": "Hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "chitchat"
    assert body["sources"] == []
