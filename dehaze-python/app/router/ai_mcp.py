"""外部 MCP Server 管理路由（F-M08-006 §2.6.13）。

对齐 API 契约（API接口.md MCP Server 管理接口）：注册表 CRUD/启停/健康/凭据、
工具清单、命名空间、市场接入、调用审计。管理操作统一 `ai:mcp:manage`，
普通用户越权返回 403（A0301）。

服务分工：Server 注册表/凭据/健康/命名空间由 mcp-core 的 ai_mcp_server_service
承载；工具拉取/命名空间/市场/调用审计由 mcp-toolkit 的 mcp_manage_service 承载。
"""

from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.ai_mcp import (
    McpCallQuery,
    McpCallResult,
    McpCredentialForm,
    McpHealthResult,
    McpMarketPreset,
    McpNamespaceItem,
    McpServerCreate,
    McpServerQuery,
    McpServerResult,
    McpServerStatusForm,
    McpServerUpdate,
    McpToolResult,
    McpToolTestForm,
    McpToolTestResult,
)
from app.models.schema.common import PageResult
from app.service.ai_mcp.ai_mcp_server_service import ai_mcp_server_service
from app.service.ai_mcp.mcp_manage_service import mcp_manage_service

router = APIRouter(prefix="/api/v1/ai/mcp", tags=["AI对话"])

_MANAGE_PERMISSION = "ai:mcp:manage"


# ── Server 注册表 ────────────────────────────────────────────


@router.get("/servers", response_model=Result[PageResult[McpServerResult]], summary="MCP Server 列表")
@require_permission(_MANAGE_PERMISSION)
async def list_servers(
    query: McpServerQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    items, total = await ai_mcp_server_service.list_servers(
        db, query.pageNum, query.pageSize, query.keyword, query.status
    )
    return success(PageResult(list=items, total=total))


@router.post("/servers", response_model=Result[McpServerResult], summary="注册外部 MCP Server")
@require_permission(_MANAGE_PERMISSION)
async def create_server(
    form: McpServerCreate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_mcp_server_service.create_server(db, form)
    return success(result)


@router.get("/servers/{server_id}", response_model=Result[McpServerResult], summary="Server 详情")
@require_permission(_MANAGE_PERMISSION)
async def get_server(
    server_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await ai_mcp_server_service.get_server(db, server_id))


@router.put("/servers/{server_id}", response_model=Result[McpServerResult], summary="更新 Server")
@require_permission(_MANAGE_PERMISSION)
async def update_server(
    server_id: int,
    form: McpServerUpdate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await ai_mcp_server_service.update_server(db, server_id, form))


@router.delete("/servers/{server_id}", response_model=Result[None], summary="删除 Server")
@require_permission(_MANAGE_PERMISSION)
async def delete_server(
    server_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await ai_mcp_server_service.delete_server(db, server_id)
    return success(msg="一切ok")


@router.patch("/servers/{server_id}/status", response_model=Result[McpServerResult], summary="启停 Server")
@require_permission(_MANAGE_PERMISSION)
async def switch_server_status(
    server_id: int,
    form: McpServerStatusForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await ai_mcp_server_service.switch_server_status(db, server_id, form.status))


# ── 健康 / 工具 / 命名空间 / 凭据 ────────────────────────────


@router.get("/servers/{server_id}/health", response_model=Result[McpHealthResult], summary="Server 健康探测")
@require_permission(_MANAGE_PERMISSION)
async def probe_health(
    server_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await ai_mcp_server_service.probe_health(db, server_id))


@router.get("/servers/{server_id}/tools", response_model=Result[list[McpToolResult]], summary="Server 工具清单")
@require_permission(_MANAGE_PERMISSION)
async def get_tools(
    server_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await mcp_manage_service.get_tools(db, server_id))


@router.post("/servers/{server_id}/tools/test", response_model=Result[McpToolTestResult], summary="试调用 MCP 工具")
@require_permission(_MANAGE_PERMISSION)
async def test_tool(
    server_id: int,
    form: McpToolTestForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(
        await mcp_manage_service.test_tool(db, server_id, form.tool_name, form.arguments)
    )


@router.get("/servers/{server_id}/namespaces", response_model=Result[list[McpNamespaceItem]], summary="命名空间列表")
@require_permission(_MANAGE_PERMISSION)
async def list_namespaces(
    server_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await ai_mcp_server_service.list_namespaces(db, server_id))


@router.put("/servers/{server_id}/namespaces", response_model=Result[list[McpNamespaceItem]], summary="配置命名空间")
@require_permission(_MANAGE_PERMISSION)
async def update_namespaces(
    server_id: int,
    namespaces: list[McpNamespaceItem],
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await ai_mcp_server_service.update_namespaces(db, server_id, namespaces))


@router.put("/servers/{server_id}/credentials", response_model=Result[None], summary="配置外部服务凭据")
@require_permission(_MANAGE_PERMISSION)
async def update_credentials(
    server_id: int,
    form: McpCredentialForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await ai_mcp_server_service.update_credentials(db, server_id, form)
    return success(msg="一切ok")


# ── 市场 / 调用审计 ──────────────────────────────────────────


@router.get("/market", response_model=Result[list[McpMarketPreset]], summary="MCP 市场目录")
@require_permission(_MANAGE_PERMISSION)
async def get_market(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await mcp_manage_service.get_market(db))


@router.post("/market/{preset_id}/install", response_model=Result[McpServerResult], summary="从市场接入预设 Server")
@require_permission(_MANAGE_PERMISSION)
async def install_preset(
    preset_id: str,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    return success(await mcp_manage_service.install_preset(db, redis, preset_id))


@router.get("/calls", response_model=Result[PageResult[McpCallResult]], summary="MCP 调用审计")
@require_permission(_MANAGE_PERMISSION)
async def list_calls(
    query: McpCallQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await mcp_manage_service.list_calls(db, query))
