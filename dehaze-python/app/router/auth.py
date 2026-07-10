from app.config import settings
from app.core.code import ResultCode
from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.user import (CaptchaData, CurrentUserVO, LoginData,
                                    LoginForm)
from app.repository.login_log_repository import login_log_repository
from app.service.auth_service import AuthService
from app.utils.user_agent import parse_user_agent
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/v1/auth", tags=["认证中心"])

# 登录失败限制配置
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION = 900  # 15 分钟


@router.post(
    "/login",
    response_model=Result[LoginData],
    summary="用户登录",
    description="通过用户名和密码登录，返回 JWT Token。包含暴力破解防护：5 次失败后锁定 15 分钟",
)
async def login(
    request: LoginForm,
    req: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    # 获取客户端信息
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "")
    browser, os_name = parse_user_agent(user_agent)

    # 1. 验证码校验（先于账户锁定检查）
    captcha_valid = await AuthService.verify_captcha(redis, request.captchaKey, request.captchaCode)
    if not captcha_valid:
        # 检查验证码是否存在（区分过期和错误）
        stored_captcha = await redis.get(f"captcha:{request.captchaKey}")
        if stored_captcha is None:
            # 记录登录失败日志
            await login_log_repository.create_log(
                db, None, request.username, client_ip, 0,
                "验证码已过期", browser, os_name
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "code": ResultCode.VERIFY_CODE_TIMEOUT.code,
                    "msg": ResultCode.VERIFY_CODE_TIMEOUT.msg,
                    "data": None,
                },
            )
        else:
            # 记录登录失败日志
            await login_log_repository.create_log(
                db, None, request.username, client_ip, 0,
                "验证码错误", browser, os_name
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "code": ResultCode.VERIFY_CODE_ERROR.code,
                    "msg": ResultCode.VERIFY_CODE_ERROR.msg,
                    "data": None,
                },
            )

    # 2. 用户名规范化：转小写、去首尾空格
    username = request.username.lower().strip()

    # 3. 暴力破解防护：检查账户锁定状态
    lockout_key = f"login:lockout:{username}"
    attempts_key = f"login:attempts:{username}"

    is_locked = await redis.get(lockout_key)
    if is_locked:
        ttl = await redis.ttl(lockout_key)
        # 记录登录失败日志
        await login_log_repository.create_log(
            db, None, username, client_ip, 0,
            f"账户已锁定，剩余 {ttl // 60} 分钟", browser, os_name
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "code": "AUTH_001",
                "msg": f"账户已锁定，请 {ttl // 60} 分钟后重试",
                "data": None,
            },
        )

    try:
        result = await AuthService.login(db, username, request.password)
        # 登录成功：清除失败计数，记录日志
        await redis.delete(attempts_key)
        user_data = result.get("user", {})
        await login_log_repository.create_log(
            db, user_data.get("id"), username, client_ip, 1,
            "登录成功", browser, os_name
        )
        return success(result)
    except ValueError as e:
        # 登录失败：增加失败计数
        attempts = await redis.incr(attempts_key)
        if attempts == 1:
            await redis.expire(attempts_key, LOCKOUT_DURATION)

        # 记录登录失败日志
        await login_log_repository.create_log(
            db, None, username, client_ip, 0,
            str(e), browser, os_name
        )

        # 达到最大尝试次数：锁定账户
        if attempts >= MAX_LOGIN_ATTEMPTS:
            await redis.setex(lockout_key, LOCKOUT_DURATION, "1")
            await redis.delete(attempts_key)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "code": "AUTH_001",
                    "msg": f"登录失败次数过多，账户已锁定 {LOCKOUT_DURATION // 60} 分钟",
                    "data": None,
                },
            )

        remaining = MAX_LOGIN_ATTEMPTS - attempts
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "code": ResultCode.USERNAME_OR_PASSWORD_ERROR.code,
                "msg": f"{str(e)}，剩余 {remaining} 次尝试机会",
                "data": None,
            },
        )


@router.post("/logout", response_model=Result[None], summary="用户注销", description="将 Token 的 jti 加入黑名单，使其立即失效")
async def logout(
    request: Request,
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    # 从请求头获取 Token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        # 解码获取 jti
        from jose import jwt
        try:
            payload = jwt.decode(
                token, settings.JWT_SECRET_KEY, algorithms=["HS256"])
            jti = payload.get("jti")
            if jti:
                # 使用 jti 作为黑名单 key
                await redis.setex(
                    f"token:blacklist:{jti}",
                    settings.JWT_ACCESS_TOKEN_EXPIRES,
                    "1",
                )
        except Exception:
            pass  # Token 解码失败时忽略

    return success(msg="一切ok")


@router.get("/captcha", response_model=Result[CaptchaData], summary="获取验证码", description="返回验证码图片和 key")
async def get_captcha(
    redis: Redis = Depends(get_redis),
):
    result = await AuthService.get_captcha(redis)
    return success(result)


@router.get("/me", response_model=Result[CurrentUserVO], summary="获取当前用户信息", description="需要携带有效的 Bearer Token")
async def get_current_user_info(
    user: UserContext = Depends(get_current_user),
):
    return success(
        {
            "userId": user.id,
            "username": user.username,
            "nickname": user.nickname,
            "roles": user.roles,
            "permissions": user.permissions[:20] if user.permissions else [],
        }
    )


@router.post("/refresh", response_model=Result[LoginData], summary="刷新访问令牌", description="使用当前有效的 Token 获取新的访问令牌，原 Token 的 jti 会被加入黑名单")
async def refresh_token(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    try:
        result = await AuthService.refresh_token(db, user.id, redis)
        return success(result)
    except ValueError as e:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "code": ResultCode.TOKEN_INVALID.code,
                "msg": str(e),
                "data": None,
            },
        )
