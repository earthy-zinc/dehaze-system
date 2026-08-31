"""语音引擎注册表管理路由（管理端：Provider / Key / Model，权限 voice:engine:manage）"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.common import BasePageQuery, PageResult
from app.models.schema.voice_admin import (
    VoiceModelCreate,
    VoiceModelResult,
    VoiceModelUpdate,
    VoiceProviderCreate,
    VoiceProviderKeyCreate,
    VoiceProviderKeyResult,
    VoiceProviderKeyUpdate,
    VoiceProviderResult,
    VoiceProviderUpdate,
)
from app.service.voice.voice_admin_service import voice_admin_service

router = APIRouter(prefix="/api/v1/voice", tags=["语音交互-管理"])


class ProviderPageQuery(BasePageQuery):
    keyword: str | None = Query(default=None, description="关键字(按显示名称/引擎编码模糊搜索)")
    engine_type: str | None = Query(default=None, description="能力类型(asr/tts)")


# ==================== Provider ====================


@router.get("/providers", response_model=Result[PageResult[VoiceProviderResult]], summary="引擎分页列表")
@require_permission("voice:engine:manage")
async def list_providers(
    query: ProviderPageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await voice_admin_service.list_providers(
        db, query.pageNum, query.pageSize, query.keyword, query.engine_type
    )
    return success(result)


@router.get("/providers/enabled", response_model=Result[list[VoiceProviderResult]], summary="启用引擎列表")
@require_permission("voice:engine:manage")
async def list_enabled_providers(
    engine_type: str = Query(..., description="能力类型(asr/tts)"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await voice_admin_service.list_enabled(db, engine_type))


@router.post("/providers", response_model=Result[VoiceProviderResult], summary="新增引擎")
@require_permission("voice:engine:manage")
async def create_provider(
    form: VoiceProviderCreate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await voice_admin_service.create_provider(db, form))


@router.put("/providers/{provider_id}", response_model=Result[VoiceProviderResult], summary="更新引擎")
@require_permission("voice:engine:manage")
async def update_provider(
    provider_id: int,
    form: VoiceProviderUpdate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await voice_admin_service.update_provider(db, provider_id, form))


@router.delete("/providers/{provider_id}", response_model=Result[None], summary="删除引擎")
@require_permission("voice:engine:manage")
async def delete_provider(
    provider_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await voice_admin_service.delete_provider(db, provider_id)
    return success(msg="一切ok")


@router.post("/providers/{provider_id}/test-connection", response_model=Result[dict], summary="连通性测试")
@require_permission("voice:engine:manage")
async def test_provider_connection(
    provider_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await voice_admin_service.test_connection(db, provider_id))


# ==================== Key ====================


@router.get("/providers/{provider_id}/keys", response_model=Result[list[VoiceProviderKeyResult]], summary="引擎API Key列表")
@require_permission("voice:engine:manage")
async def list_keys(
    provider_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await voice_admin_service.list_keys(db, provider_id))


@router.post("/providers/{provider_id}/keys", response_model=Result[VoiceProviderKeyResult], summary="新增API Key")
@require_permission("voice:engine:manage")
async def create_key(
    provider_id: int,
    form: VoiceProviderKeyCreate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await voice_admin_service.create_key(db, provider_id, form))


@router.put("/providers/{provider_id}/keys/{key_id}", response_model=Result[VoiceProviderKeyResult], summary="更新API Key")
@require_permission("voice:engine:manage")
async def update_key(
    provider_id: int,
    key_id: int,
    form: VoiceProviderKeyUpdate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await voice_admin_service.update_key(db, provider_id, key_id, form))


@router.delete("/providers/{provider_id}/keys/{key_id}", response_model=Result[None], summary="删除API Key")
@require_permission("voice:engine:manage")
async def delete_key(
    provider_id: int,
    key_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await voice_admin_service.delete_key(db, provider_id, key_id)
    return success(msg="一切ok")


# ==================== Model ====================


@router.get("/models", response_model=Result[list[VoiceModelResult]], summary="模型/音色列表")
@require_permission("voice:engine:manage")
async def list_models(
    engine_type: str | None = Query(default=None, description="能力类型(asr/tts)"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await voice_admin_service.list_models(db, engine_type))


@router.post("/models", response_model=Result[VoiceModelResult], summary="新增模型/音色")
@require_permission("voice:engine:manage")
async def create_model(
    form: VoiceModelCreate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await voice_admin_service.create_model(db, form))


@router.put("/models/{model_id}", response_model=Result[VoiceModelResult], summary="更新模型/音色")
@require_permission("voice:engine:manage")
async def update_model(
    model_id: int,
    form: VoiceModelUpdate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await voice_admin_service.update_model(db, model_id, form))


@router.delete("/models/{model_id}", response_model=Result[None], summary="删除模型/音色")
@require_permission("voice:engine:manage")
async def delete_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await voice_admin_service.delete_model(db, model_id)
    return success(msg="一切ok")
