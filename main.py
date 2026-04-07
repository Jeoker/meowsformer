from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from loguru import logger
from app.services.rag_service import initialize_knowledge_base
from app.api.endpoints import router as api_router
from app.api.ws_endpoints import router as ws_router
import uvicorn

# 配置日志
logger.add("app.log", rotation="500 MB")

app = FastAPI(
    title="MeowTranslator",
    debug=settings.DEBUG_MODE,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    logger.info("Starting up MeowTranslator...")

    # Initialize RAG knowledge base
    try:
        initialize_knowledge_base()
        logger.info("Knowledge base initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize knowledge base: {e}")

    # Pre-load tagged samples for matching engine
    try:
        from app.services.sample_matcher import load_tagged_samples
        load_tagged_samples()
        logger.info("Tagged samples loaded for matching engine.")
    except Exception as e:
        logger.warning(f"Could not load tagged samples: {e}")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "app": "MeowTranslator"}


# Register routers
app.include_router(api_router, prefix="/api")  # /api/translate, /api/v1/translate
app.include_router(ws_router)            # /ws/translate

# Vue SPA (Docker / production): built to static/ui by Dockerfile multi-stage build
_ui_static = Path(__file__).resolve().parent / "static" / "ui"
if _ui_static.is_dir():
    app.mount("/", StaticFiles(directory=str(_ui_static), html=True), name="ui")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG_MODE)
