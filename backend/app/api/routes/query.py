from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import Settings
from app.schemas.query import HealthResponse, QueryRequest, QueryResponse, SourceCitation
from app.services import generation
from app.services.retrieval import RetrievalContext
from app.utils.logging_config import get_logger

router = APIRouter()
logger = get_logger("routes.query")


def get_retrieval_context(request: Request) -> RetrievalContext:
    context = getattr(request.app.state, "retrieval_context", None)
    if context is None:
        raise HTTPException(status_code=503, detail="Retrieval service is not ready")
    return context


def get_app_settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=503, detail="Settings are not ready")
    return settings


def get_ollama_client(request: Request):
    return getattr(request.app.state, "ollama_client", None)


@router.get("/health", response_model=HealthResponse)
def health_check(
    retrieval_context: RetrievalContext = Depends(get_retrieval_context),
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        vector_store_loaded=True,
        chunk_count=len(retrieval_context.chunk_ids),
        ollama_model=settings.ollama_model,
    )


@router.post("/query", response_model=QueryResponse)
def run_query(
    payload: QueryRequest,
    retrieval_context: RetrievalContext = Depends(get_retrieval_context),
    settings: Settings = Depends(get_app_settings),
    ollama_client=Depends(get_ollama_client),
) -> QueryResponse:
    logger.info("Received query of length %d", len(payload.question))
    try:
        result = generation.answer_question(
            retrieval_context, settings, payload.question, client=ollama_client
        )
    except Exception as exc:
        logger.exception("Unexpected failure while answering query")
        raise HTTPException(status_code=500, detail="Internal error while generating an answer") from exc

    sources_detail = [
        SourceCitation(
            source_id=chunk["metadata"]["source_id"],
            title=chunk["metadata"]["title"],
            authors=chunk["metadata"]["authors"],
            page=str(chunk["metadata"]["page"]),
            section=chunk["metadata"]["section"],
            source_url=chunk["metadata"]["source_url"],
            topic=chunk["metadata"]["topic"],
        )
        for chunk in result["retrieved_chunks"]
    ]
    sources = [detail.title for detail in sources_detail]

    logger.info("Answered query with intent=%s is_refusal=%s", result["intent"], result["is_refusal"])

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        sources_detail=sources_detail,
        intent=result["intent"],
        citation_status=result["citation_status"],
        is_refusal=result["is_refusal"],
    )
