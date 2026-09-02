from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import Settings, get_settings
from app.services.retrieval import RetrievalContext, load_retrieval_context
from app.services.generation import get_ollama_client
from app.utils.logging_config import get_logger, setup_logging


def create_app(
    settings: Settings | None = None,
    retrieval_context: RetrievalContext | None = None,
    ollama_client=None,
) -> FastAPI:
    active_settings = settings or get_settings()
    setup_logging("INFO")
    logger = get_logger("main")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = active_settings

        if retrieval_context is not None:
            app.state.retrieval_context = retrieval_context
        else:
            logger.info("Loading persisted Chroma vector store and rebuilding BM25 index")
            app.state.retrieval_context = load_retrieval_context(active_settings)
            logger.info("Retrieval context ready with %d chunks", len(app.state.retrieval_context.chunk_ids))

        app.state.ollama_client = ollama_client or get_ollama_client(active_settings)

        yield

        app.state.retrieval_context = None
        app.state.ollama_client = None

    app = FastAPI(
        title="AI StudyMate RAG API",
        description="RAG-powered Machine Learning and Deep Learning study assistant backend",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.frontend_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(query_router)

    @app.get("/")
    def root():
        return {"service": "AI StudyMate RAG API", "docs": "/docs", "health": "/health"}

    return app


app = create_app()
