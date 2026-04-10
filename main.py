import asyncio
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from loguru import logger
from app.services.rag_service import initialize_knowledge_base
from app.api.endpoints import router as api_router
from app.api.ws_endpoints import router as ws_router
from app import request_stats
import uvicorn

# 配置日志
logger.add("app.log", rotation="500 MB")

app = FastAPI(
    title="MeowTranslator",
    debug=settings.DEBUG_MODE,
)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    request_stats.increment(request.url.path)
    return await call_next(request)


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

    asyncio.create_task(request_stats.periodic_save())
    logger.info(
        f"Stats loaded from {request_stats.STATS_FILE}: "
        f"{request_stats.snapshot_for_log()}"
    )


@app.on_event("shutdown")
async def shutdown_event():
    request_stats.save()
    logger.info("Stats saved on shutdown.")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "app": "MeowTranslator"}


@app.get("/stats")
async def get_stats():
    """请求计数统计；写入 stats.json，部署新镜像或 SIGKILL 可能丢失未落盘增量。"""
    return request_stats.stats_payload()


# Register routers
app.include_router(api_router, prefix="/api")  # /api/translate, /api/v1/translate
app.include_router(ws_router)            # /ws/translate

# Vue SPA (Docker / production): built to static/ui by Dockerfile multi-stage build
_ui_static = Path(__file__).resolve().parent / "static" / "ui"
if _ui_static.is_dir():
    app.mount("/", StaticFiles(directory=str(_ui_static), html=True), name="ui")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG_MODE)
