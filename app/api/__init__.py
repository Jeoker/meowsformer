from fastapi import APIRouter
from app.api.endpoints import router as translation_router

api_router = APIRouter()
api_router.include_router(translation_router, prefix="/v1", tags=["translation"])
