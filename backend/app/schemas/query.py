from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class SourceCitation(BaseModel):
    source_id: str
    title: str
    authors: str
    page: str
    section: str
    source_url: str
    topic: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    sources_detail: list[SourceCitation]
    intent: str
    citation_status: str
    is_refusal: bool


class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool
    chunk_count: int
    ollama_model: str
