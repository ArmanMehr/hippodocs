from fastapi import APIRouter

from .ask import router as ask_router
from .workspaces import router as workspaces_router

router = APIRouter(prefix="/v1")
router.include_router(workspaces_router)
router.include_router(ask_router)
