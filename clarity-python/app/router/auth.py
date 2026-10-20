import logging
from urllib.parse import quote

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.result import Result, success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import SESSION_COOKIE, SESSION_TTL, UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.user import (
    CaptchaData,
    CurrentUserVO,
    LoginData,
    LoginForm,
    PasswordChangeForm,
    RegisterForm,
)
from app.repository.user_repository import user_repository
from app.service.auth_service import auth_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["认证中心"])


def _set_session_cookie(response: Response, session_id: str, remember_me: bool):
    max_age = SESSION_TTL if remember_me else None
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        max_age=max_age,
        path=settings.SESSION_COOKIE_PATH,
        httponly=True,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite="lax",
    )


def _clear_session_cookie(response: Response):
    response.delete_cookie(SESSION_COOKIE, path=settings.SESSION_COOKIE_PATH)


@router.post(
    "/login",
    response_model=Result[LoginData],
    summary="用户登录",
)
async def login(
    request: LoginForm,
    req: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "")

    result = await auth_service.login(
        db,
        redis,
        request.username.lower().strip(),
        request.password,
        client_ip,
        request.captchaKey,
        request.captchaCode,
        user_agent,
        device_type=request.deviceType or "web",
    )
    remember_me = request.rememberMe if request.rememberMe is not None else False
    _set_session_cookie(response, result.get("sessionId", ""), remember_me)
    return success(result)


@router.post("/register", response_model=Result[LoginData], summary="用户注册")
async def register(
    request: RegisterForm,
    req: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    client_ip = req.client.host if req.client else "unknown"
    result = await auth_service.register(
        db,
        redis,
        request.username,
        request.password,
        request.nickname,
        request.captchaKey,
        request.captchaCode,
        client_ip,
    )
    _set_session_cookie(response, result.get("sessionId", ""), False)
    return success(result)


@router.post("/logout", response_model=Result[None], summary="用户注销")
async def logout(
    request: Request,
    response: Response,
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    session_id = request.cookies.get(SESSION_COOKIE) or request.headers.get(SESSION_COOKIE)
    if session_id:
        # 仅从索引移除当前会话元素，不影响其他端在线会话（F-AM-011）
        await redis.zrem(f"session:user:{user.id}", session_id)
        await redis.delete(f"session:{session_id}")

    _clear_session_cookie(response)
    return success(msg="一切ok")


@router.get("/captcha", response_model=Result[CaptchaData], summary="获取验证码")
async def get_captcha(
    redis: Redis = Depends(get_redis),
):
    result = await auth_service.get_captcha(redis)
    return success(result)


@router.get("/me", response_model=Result[CurrentUserVO], summary="获取当前用户信息")
async def get_current_user_info(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    sys_user = await user_repository.get_by_id(db, user.id)
    return success(
        {
            "userId": user.id,
            "username": user.username,
            "nickname": user.nickname,
            "avatar": sys_user.avatar if sys_user else None,
            "roles": user.roles,
            "perms": user.permissions if user.permissions else [],
        }
    )


@router.patch("/password", response_model=Result[None], summary="修改个人密码")
async def change_password(
    request: PasswordChangeForm,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    """个人安全操作（登录态即可）：旧密码校验失败返回 A0210，
    成功后踢出本人全部在线会话强制重新登录。管理员重置他人密码走
    PATCH /api/v1/users/{userId}/password（sys:user:password:reset）。"""
    await auth_service.change_password(db, redis, user, request.oldPassword, request.newPassword)
    return success(msg="密码修改成功，请重新登录")


@router.get("/login-logs", summary="登录日志查询（分页）")
async def list_login_logs(
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页数量"),
    username: str | None = Query(default=None, description="按用户名筛选"),
    ip: str | None = Query(default=None, description="按IP筛选"),
    status: int | None = Query(default=None, description="登录状态(1:成功;0:失败)"),
    deviceType: str | None = Query(
        default=None, description="设备类型(web/android/flutter/miniprogram)"
    ),
    startTime: str | None = Query(default=None, description="开始时间"),
    endTime: str | None = Query(default=None, description="结束时间"),
    user: UserContext = Depends(get_current_user),
):
    """登录日志查询。

    - 管理员（is_admin）查看全量日志
    - 普通用户仅查看本人日志（即便传入他人 username 也强制限定本人）
    """
    result = await auth_service.list_login_logs(
        pageNum,
        pageSize,
        username=username,
        ip=ip,
        status=status,
        device_type=deviceType,
        start_time=startTime,
        end_time=endTime,
        user=user,
    )
    return success(result)


@router.get("/login-logs/export", summary="登录日志导出（Excel）")
async def export_login_logs(
    username: str | None = Query(default=None, description="按用户名筛选"),
    ip: str | None = Query(default=None, description="按IP筛选"),
    status: int | None = Query(default=None, description="登录状态(1:成功;0:失败)"),
    deviceType: str | None = Query(
        default=None, description="设备类型(web/android/flutter/miniprogram)"
    ),
    startTime: str | None = Query(default=None, description="开始时间"),
    endTime: str | None = Query(default=None, description="结束时间"),
    user: UserContext = Depends(get_current_user),
):
    """按当前筛选条件导出登录日志（数据权限与分页查询一致：普通用户仅本人）。"""
    content = await auth_service.export_login_logs(
        username=username,
        ip=ip,
        status=status,
        device_type=deviceType,
        start_time=startTime,
        end_time=endTime,
        user=user,
    )
    filename = quote("登录日志.xlsx")
    return StreamingResponse(
        iter([content]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )


@router.get("/sessions", summary="在线会话列表（管理员）")
@require_permission("sys:auth:session:list")
async def list_sessions(
    username: str = Query(..., description="用户名（精确匹配）"),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    return success(await auth_service.list_sessions(redis, username))


@router.delete("/sessions/{sessionId}", summary="踢出指定在线会话（管理员）")
@require_permission("sys:auth:session:kick")
async def kick_session(
    sessionId: str,
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await auth_service.kick_session(redis, sessionId)
    return success(msg="会话已踢出")
