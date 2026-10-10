"""调试工具库：多后端、多用户登录 + session 缓存。

对齐 dehaze-sdk-js/test/utils/auth.ts：
- 按 (backend, username) 缓存 sessionId，避免重复登录触发限流
- 验证码直接从 Redis 读（key: captcha_code:{captchaKey}），避免 OCR
- 不做本地文件缓存（避免与后端 Redis session 状态不同步）
"""
from __future__ import annotations

import re

import httpx

from . import config, redis


# (backend, username) -> session_id
_session_store: dict[tuple[str, str], str] = {}

# Java 端生产 profile 的验证码字符是算式（captcha.code.type=math，Hutool MathGenerator）：
# Redis 存的是算式本身（如 "2*4="，运算位数 >1 时数字后会补空格），校验方按算式求值再与
# 提交值比较 ⇒ 客户端必须提交计算结果，不能直接回填 Redis 原文。dev 的 random 与 Go/Python
# 端存的就是最终码，原样提交即可
_ARITHMETIC = re.compile(r"^(\d+)([+\-*])(\d+)=?$")


def _captcha_answer(raw: str) -> str:
    """Redis 中的验证码明文 → 实际要提交的验证码。"""
    compact = raw.replace(" ", "")
    matched = _ARITHMETIC.match(compact)
    if not matched:
        return compact
    left, operator, right = int(matched.group(1)), matched.group(2), int(matched.group(3))
    if operator == "+":
        return str(left + right)
    if operator == "-":
        return str(left - right)
    return str(left * right)


def login(
    username: str = config.DEFAULT_USERNAME,
    password: str = config.DEFAULT_PASSWORD,
    backend: str = config.DEFAULT_BACKEND,
) -> str:
    """登录指定后端，返回 X-Session-Id。

    流程：
    1. 命中缓存直接返回
    2. GET /api/v1/auth/captcha → 拿 captchaKey
    3. 从 Redis 读 captcha_code:{captchaKey} → 拿验证码（算式需作答）
    4. POST /api/v1/auth/login → 拿 sessionId
    5. 缓存并设置为当前后端的默认 session
    """
    key = (backend, username)
    if key in _session_store:
        return _session_store[key]

    backend_cfg = config.get_backend(backend)
    base = backend_cfg.base_url

    # 1. 获取验证码
    with httpx.Client(base_url=base, timeout=10) as client:
        captcha_resp = client.get("/api/v1/auth/captcha")
        captcha_resp.raise_for_status()
        captcha_data = captcha_resp.json()["data"]
        captcha_key = captcha_data["captchaKey"]

        # 2. 从 Redis 读验证码明文（Java 生产 profile 为算式，需按算式作答）
        raw_captcha = redis.get_captcha(captcha_key)
        if not raw_captcha:
            raise RuntimeError(
                f"无法从 Redis 读取验证码 (key=captcha_code:{captcha_key})，"
                f"请确认后端 {backend} 已正确写入 Redis"
            )
        captcha_code = _captcha_answer(raw_captcha)

        # 3. 登录
        login_resp = client.post(
            "/api/v1/auth/login",
            json={
                "username": username,
                "password": password,
                "captchaKey": captcha_key,
                "captchaCode": captcha_code,
                "rememberMe": False,
            },
        )
        login_resp.raise_for_status()
        resp_data = login_resp.json()
        if resp_data.get("code") != config.SUCCESS_CODE:
            raise RuntimeError(f"登录失败: {resp_data}")
        session_id = resp_data["data"]["sessionId"]

    # 4. 缓存（避免循环 import，延迟导入 api）
    from . import api
    _session_store[key] = session_id
    api.set_session(backend, session_id)

    return session_id


def logout(backend: str = config.DEFAULT_BACKEND) -> None:
    """登出指定后端，清除缓存。"""
    from . import api  # 延迟导入避免循环依赖
    for key in list(_session_store.keys()):
        if key[0] == backend:
            session_id = _session_store.pop(key, None)
            api.set_session(backend, None)
            if session_id:
                try:
                    api.post("/api/v1/auth/logout", backend=backend)
                except Exception:
                    pass  # 静默失败
            break


def get_current_session(backend: str = config.DEFAULT_BACKEND, username: str = config.DEFAULT_USERNAME) -> str:
    """获取已缓存的 session（未登录则自动登录）。"""
    return _session_store.get((backend, username)) or login(username=username, backend=backend)


def clear_cache() -> None:
    """清除所有缓存的 session（不从后端登出，仅清本地缓存）。"""
    from . import api  # 延迟导入避免循环依赖
    _session_store.clear()
    api.clear_sessions()


def get_user_id(username: str = config.DEFAULT_USERNAME) -> int:
    """查询用户 ID（从 MySQL）。"""
    from . import mysql
    user = mysql.get_user_by_username(username)
    if not user:
        raise RuntimeError(f"用户不存在: {username}")
    return user["id"]
