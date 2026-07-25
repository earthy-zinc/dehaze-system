import logging

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.user import (CaptchaData, CurrentUserVO, LoginData,
                                    LoginForm)
from app.repository.login_log_repository import login_log_repository
from app.service.auth_service import AuthService
from app.utils.user_agent import parse_user_agent
from fastapi import APIRouter, Depends, Request, Response, status
from app.middleware.non_null_response import NonNullJSONResponse as JSONResponse
from jose import jwt
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["认证中心"])

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION = 900
REFRESH_TOKEN_MAX_AGE = 7 * 24 * 3600


def _set_refresh_token_cookies(response: Response, refresh_token: str, remember_me: bool):
    max_age = REFRESH_TOKEN_MAX_AGE if remember_me else -1
    response.set_cookie(
        "refreshToken", refresh_token, max_age=max_age, path="/",
        httponly=True, samesite="lax",
    )
    response.set_cookie(
        "rememberMe", str(remember_me).lower(), max_age=max_age, path="/",
        httponly=False, samesite="lax",
    )


def _clear_refresh_token_cookies(response: Response):
    response.delete_cookie("refreshToken", path="/")
    response.delete_cookie("rememberMe", path="/")


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
    browser, os_name = parse_user_agent(user_agent)

    captcha_valid = await AuthService.verify_captcha(redis, request.captchaKey, request.captchaCode)
    if not captcha_valid:
        stored_captcha = await redis.get(f"captcha:{request.captchaKey}")
        if stored_captcha is None:
            await login_log_repository.create_log(
                db, None, request.username, client_ip, 0,
                "验证码已过期", browser, os_name
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"code": ResultCode.VERIFY_CODE_TIMEOUT.code, "msg": ResultCode.VERIFY_CODE_TIMEOUT.msg, "data": None},
            )
        else:
            await login_log_repository.create_log(
                db, None, request.username, client_ip, 0,
                "验证码错误", browser, os_name
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"code": ResultCode.VERIFY_CODE_ERROR.code, "msg": ResultCode.VERIFY_CODE_ERROR.msg, "data": None},
            )

    username = request.username.lower().strip()
    lockout_key = f"login:lockout:{username}"
    attempts_key = f"login:attempts:{username}"

    is_locked = await redis.get(lockout_key)
    if is_locked:
        ttl = await redis.ttl(lockout_key)
        await login_log_repository.create_log(
            db, None, username, client_ip, 0,
            f"账户已锁定，剩余 {ttl // 60} 分钟", browser, os_name
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"code": "AUTH_001", "msg": f"账户已锁定，请 {ttl // 60} 分钟后重试", "data": None},
        )

    try:
        result = await AuthService.login(db, redis, username, request.password)
        await redis.delete(attempts_key)
        user_data = result.get("user", {})
        await login_log_repository.create_log(
            db, user_data.get("id"), username, client_ip, 1,
            "登录成功", browser, os_name
        )
        remember_me = request.rememberMe if request.rememberMe is not None else False
        _set_refresh_token_cookies(response, result.get("refreshToken", ""), remember_me)
        return success(result)
    except ValueError as e:
        attempts = await redis.incr(attempts_key)
        if attempts == 1:
            await redis.expire(attempts_key, LOCKOUT_DURATION)

        await login_log_repository.create_log(
            db, None, username, client_ip, 0,
            str(e), browser, os_name
        )

        if attempts >= MAX_LOGIN_ATTEMPTS:
            await redis.setex(lockout_key, LOCKOUT_DURATION, "1")
            await redis.delete(attempts_key)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"code": "AUTH_001", "msg": f"登录失败次数过多，账户已锁定 {LOCKOUT_DURATION // 60} 分钟", "data": None},
            )

        remaining = MAX_LOGIN_ATTEMPTS - attempts
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"code": ResultCode.USERNAME_OR_PASSWORD_ERROR.code, "msg": f"{str(e)}，剩余 {remaining} 次尝试机会", "data": None},
        )


@router.post("/logout", response_model=Result[None], summary="用户注销")
async def logout(
    request: Request,
    response: Response,
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS256"])
            jti = payload.get("jti")
            if jti:
                await redis.setex(
                    f"token:blacklist:{jti}",
                    settings.JWT_ACCESS_TOKEN_EXPIRES,
                    "1",
                )
        except Exception as e:
            logger.warning("logout 解码 Token 失败: %s", e)

    _clear_refresh_token_cookies(response)
    return success(msg="一切ok")


@router.get("/captcha", response_model=Result[CaptchaData], summary="获取验证码")
async def get_captcha(
    redis: Redis = Depends(get_redis),
):
    result = await AuthService.get_captcha(redis)
    return success(result)


@router.post("/refresh", response_model=Result[LoginData], summary="刷新访问令牌")
async def refresh_token(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    refresh_token_str = request.cookies.get("refreshToken", "")
    if not refresh_token_str:
        try:
            body = await request.json()
            refresh_token_str = body.get("refreshToken", "")
        except Exception:
            pass

    if not refresh_token_str:
        raise BusinessException(ResultCode.TOKEN_INVALID, "刷新令牌不能为空")

    result = await AuthService.refresh_token(db, refresh_token_str, redis)

    remember_me = request.cookies.get("rememberMe", "false") == "true"
    _set_refresh_token_cookies(response, result.get("refreshToken", ""), remember_me)
    return success(result)


@router.get("/me", response_model=Result[CurrentUserVO], summary="获取当前用户信息")
async def get_current_user_info(
    user: UserContext = Depends(get_current_user),
):
    return success({
        "userId": user.id,
        "username": user.username,
        "nickname": user.nickname,
        "roles": user.roles,
        "perms": user.permissions if user.permissions else [],
    })
