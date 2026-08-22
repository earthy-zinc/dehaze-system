import logging

from fastapi import APIRouter, Depends, Query, Request, Response
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import SESSION_COOKIE, SESSION_TTL, UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.user import CaptchaData, CurrentUserVO, LoginData, LoginForm, RegisterForm
from app.service.auth_service import AuthService

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

    result = await AuthService.login(
        db,
        redis,
        request.username.lower().strip(),
        request.password,
        client_ip,
        request.captchaKey,
        request.captchaCode,
        user_agent,
    )
    remember_me = request.rememberMe if request.rememberMe is not None else False
    _set_session_cookie(response, result.get("sessionId", ""), remember_me)
    return success(result)


@router.post("/register", response_model=Result[LoginData], summary="用户注册")
async def register(
    request: RegisterForm,
    response: Response,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    result = await AuthService.register(
        db,
        redis,
        request.username,
        request.password,
        request.nickname,
        request.captchaKey,
        request.captchaCode,
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
        # 清理多点登录索引
        if user.username:
            await redis.delete(f"session:user:{user.username}")
        await redis.delete(f"session:{session_id}")

    _clear_session_cookie(response)
    return success(msg="一切ok")


@router.get("/captcha", response_model=Result[CaptchaData], summary="获取验证码")
async def get_captcha(
    redis: Redis = Depends(get_redis),
):
    result = await AuthService.get_captcha(redis)
    return success(result)


@router.get("/me", response_model=Result[CurrentUserVO], summary="获取当前用户信息")
async def get_current_user_info(
    user: UserContext = Depends(get_current_user),
):
    return success(
        {
            "userId": user.id,
            "username": user.username,
            "nickname": user.nickname,
            "roles": user.roles,
            "perms": user.permissions if user.permissions else [],
        }
    )


@router.get("/login-logs", summary="登录日志查询（分页）")
async def list_login_logs(
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页数量"),
    username: str | None = Query(default=None, description="按用户名筛选"),
    ip: str | None = Query(default=None, description="按IP筛选"),
    status: int | None = Query(default=None, description="登录状态(1:成功;0:失败)"),
    startTime: str | None = Query(default=None, description="开始时间"),
    endTime: str | None = Query(default=None, description="结束时间"),
    user: UserContext = Depends(get_current_user),
):
    """登录日志查询。

    - 管理员（is_admin）查看全量日志
    - 普通用户仅查看本人日志（即便传入他人 username 也强制限定本人）
    """
    result = await AuthService.list_login_logs(
        pageNum,
        pageSize,
        username=username,
        ip=ip,
        status=status,
        start_time=startTime,
        end_time=endTime,
        user=user,
    )
    return success(result)
