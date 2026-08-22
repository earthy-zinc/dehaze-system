from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.ai_artifact import ArtifactPageQuery, ArtifactResult
from app.models.schema.common import PageResult
from app.service.ai_artifact_service import AiArtifactService

router = APIRouter(prefix="/api/v1/ai", tags=["AI对话"])


@router.get(
    "/conversations/{conv_id}/artifacts",
    response_model=Result[PageResult[ArtifactResult]],
    summary="会话产物分页列表",
)
async def list_conversation_artifacts(
    conv_id: int,
    query: ArtifactPageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await AiArtifactService.list_by_conversation(
        db, conv_id, user.id, query.pageNum, query.pageSize
    )
    return success(result)


@router.get(
    "/messages/{msg_id}/artifacts",
    response_model=Result[list[ArtifactResult]],
    summary="消息关联产物列表",
)
async def list_message_artifacts(
    msg_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await AiArtifactService.list_by_message(db, msg_id, user.id)
    return success(result)


@router.get(
    "/artifacts/by-ref",
    response_model=Result[list[ArtifactResult]],
    summary="按业务引用反查产物列表",
)
async def list_artifacts_by_ref(
    refType: str = Query(..., description="引用业务表"),
    refId: int = Query(..., description="引用业务表ID"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await AiArtifactService.list_by_ref(db, refType, refId, user.id)
    return success(result)


@router.get(
    "/artifacts/{artifact_id}/detail",
    response_model=Result[dict],
    summary="产物详情（含运行时图片URL）",
)
async def get_artifact_detail(
    artifact_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await AiArtifactService.get_detail(db, artifact_id, user.id)
    return success(result)
