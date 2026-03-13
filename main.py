from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from loguru import logger
from app.services.rag_service import initialize_knowledge_base
from app.api.endpoints import router as api_router
from app.api.ws_endpoints import router as ws_router
from app.auth.router import router as auth_router
from app.auth.database import create_tables
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

    # Create auth DB tables
    try:
        create_tables()
        logger.info("Auth database tables created/verified.")
    except Exception as e:
        logger.error(f"Failed to create auth tables: {e}")

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
app.include_router(auth_router)          # /auth/register, /auth/login, /auth/me
app.include_router(api_router, prefix="/api")  # /api/translate, /api/v1/translate
app.include_router(ws_router)            # /ws/translate


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG_MODE)
