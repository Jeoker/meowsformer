from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from loguru import logger
from app.services.rag_service import initialize_knowledge_base
from app.api.endpoints import router as api_router
import uvicorn
import asyncio
import os

# 配置日志
logger.add("app.log", rotation="500 MB")

app = FastAPI(
    title="MeowTranslator",
    debug=settings.DEBUG_MODE,
)

# CORS 配置
origins = [
    "*", # 开发阶段允许所有来源，生产环境请修改
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Run startup tasks such as initializing the RAG database."""
    logger.info("Starting up MeowTranslator...")
    try:
        # Initialize the knowledge base with mock data if empty
        initialize_knowledge_base()
        logger.info("Knowledge base initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize knowledge base: {e}")

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "app": "MeowTranslator"}

# Register the main router under /api
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG_MODE)
