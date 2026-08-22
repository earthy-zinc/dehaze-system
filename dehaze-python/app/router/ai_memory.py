from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.ai_memory import (
    MemoryClearQuery,
    MemoryCreate,
    MemoryPageQuery,
    MemoryResult,
    MemoryUpdate,
)
from app.models.schema.common import PageResult
from app.service.ai_memory_service import ai_memory_service

router = APIRouter(prefix="/api/v1/ai/memories", tags=["AI对话"])


@router.get("", response_model=Result[PageResult[MemoryResult]], summary="记忆分页列表")
async def list_memories(
    query: MemoryPageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_memory_service.list_memories(
        db, user.id, query.pageNum, query.pageSize, query.memoryType, query.source
    )
    return success(result)


@router.get(
    "/archived", response_model=Result[PageResult[MemoryResult]], summary="归档记忆分页列表"
)
async def list_archived(
    query: MemoryPageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_memory_service.list_archived(
        db, user.id, query.pageNum, query.pageSize, query.memoryType
    )
    return success(result)


@router.post("", response_model=Result[MemoryResult], summary="创建记忆")
async def create_memory(
    form: MemoryCreate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_memory_service.create_memory(db, user.id, form)
    return success(result)


@router.put("/{memory_id}", response_model=Result[MemoryResult], summary="更新记忆")
async def update_memory(
    memory_id: int,
    form: MemoryUpdate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_memory_service.update_memory(db, memory_id, user.id, form)
    return success(result)


@router.delete("/{memory_id}", response_model=Result[None], summary="删除记忆")
async def delete_memory(
    memory_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await ai_memory_service.delete_memory(db, memory_id, user.id)
    return success(msg="一切ok")


@router.get("/search", response_model=Result[list[MemoryResult]], summary="关键词搜索记忆")
async def search_memories(
    keyword: str,
    limit: int = 5,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_memory_service.search_memories(db, user.id, keyword, limit)
    return success(result)


@router.post("/clear", response_model=Result[int], summary="批量清空记忆")
async def clear_memories(
    query: MemoryClearQuery = Depends(),
    confirm: bool = Query(default=False, description="二次确认标识"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    count = await ai_memory_service.batch_clear(
        db, user.id, confirm, query.memoryType, query.start, query.end
    )
    return success(count, msg=f"已清空 {count} 条记忆（30 天内可恢复）")


@router.post("/restore", response_model=Result[int], summary="恢复软删记忆")
async def restore_memories(
    query: MemoryClearQuery = Depends(),
    confirm: bool = Query(default=False, description="二次确认标识"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    count = await ai_memory_service.restore_deleted(
        db, user.id, confirm, query.memoryType, query.start, query.end
    )
    return success(count, msg=f"已恢复 {count} 条记忆")


@router.get("/export", summary="导出全部记忆(JSON/Markdown)")
async def export_memories(
    fmt: str = Query(default="json", description="导出格式(json/markdown)"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    content_type, content = await ai_memory_service.export_memories(db, user.id, fmt)
    filename = f"memories.{'md' if fmt == 'markdown' else 'json'}"
    return StreamingResponse(
        iter([content]),
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
