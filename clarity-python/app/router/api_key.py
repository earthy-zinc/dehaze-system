from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.api_key import ApiKeyCreate, ApiKeyResult
from app.service.api_key_service import api_key_service

router = APIRouter(prefix="/api/v1/auth/api-keys", tags=["认证中心"])


@router.post("", response_model=Result[ApiKeyResult], summary="创建API密钥")
async def create_api_key(
    form: ApiKeyCreate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await api_key_service.create_api_key(
        db,
        user.id,
        form.name,
        form.expiresAt,
        daily_quota=form.dailyQuota,
        monthly_quota=form.monthlyQuota,
        rpm_limit=form.rpmLimit,
        model_whitelist=form.modelWhitelist,
    )
    return success(result)


@router.get("", response_model=Result[list[ApiKeyResult]], summary="获取API密钥列表")
async def list_api_keys(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await api_key_service.list_api_keys(db, user.id)
    return success(result)


@router.delete("/{key_id}", response_model=Result[None], summary="删除API密钥")
async def delete_api_key(
    key_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    deleted = await api_key_service.delete_api_key(db, user.id, key_id)
    if not deleted:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "API Key 不存在")
    return success(msg="一切ok")
