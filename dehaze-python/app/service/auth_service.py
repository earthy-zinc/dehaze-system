import asyncio
import base64
import json
import logging
import secrets
import string
import uuid
from datetime import UTC, datetime
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont
from redis.asyncio import Redis
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.auth import UserContext
from app.infrastructure.cache.cache import CacheService
from app.repository.login_log_repository import login_log_repository
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.member_repository import member_repository
from app.repository.role_repository import role_repository
from app.repository.user_repository import user_repository
from app.utils.password import check_password_async, hash_password_async

logger = logging.getLogger(__name__)

SESSION_PREFIX = "session:"
SESSION_USER_PREFIX = "session:user:"
SESSION_TTL = 7 * 24 * 3600
SESSION_COOKIE = "X-Session-Id"

DEVICE_TYPES = ("web", "android", "flutter", "miniprogram")

# 管理员（ROOT/ADMIN）不受等级权益约束，固定 10 台
ADMIN_MAX_DEVICES = 10
# 无会员记录（未初始化档案）时按 level_0 权益兜底
DEFAULT_MAX_DEVICES = 1

LOGIN_FAIL_PREFIX = "login:fail:"
LOGIN_FAIL_IP_PREFIX = "login:fail:ip:"


class AuthService:
    @staticmethod
    def _normalize_device_type(device_type: str | None) -> str:
        return device_type if device_type in DEVICE_TYPES else "web"

    async def login(
        self,
        db: AsyncSession,
        redis: Redis,
        username: str,
        password: str,
        client_ip: str = "unknown",
        captcha_key: str = "",
        captcha_code: str = "",
        user_agent: str = "",
        device_type: str = "web",
    ) -> dict:
        """登录（含审计：成功/失败均落 sys_login_log，异常路径也覆盖）"""
        from app.utils.user_agent import parse_user_agent

        browser, os_name = parse_user_agent(user_agent)
        device_type = self._normalize_device_type(device_type)
        try:
            result = await self._authenticate(
                db, redis, username, password, client_ip, captcha_key, captcha_code, device_type
            )
        except BusinessException as e:
            await login_log_repository.create_log(
                None,
                username,
                client_ip,
                0,
                e.message,
                browser,
                os_name,
                device_type=device_type,
            )
            raise
        user = result.get("user")
        if not user:
            # 登录成功但响应缺失用户结构属内部异常，不得静默记为匿名，显式暴露
            raise BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "登录成功但用户信息缺失")
        # 会员档案兜底：种子账号与后台创建的用户不走注册流程，登录时确保
        # sys_member 行存在（否则计费配额校验 fail-closed 误报"配额不足"）
        await member_repository.get_or_init_member(db, user.get("id"))
        await login_log_repository.create_log(
            user.get("id"),
            username,
            client_ip,
            1,
            "登录成功",
            browser,
            os_name,
            device_type=device_type,
        )
        return result

    async def _authenticate(
        self,
        db: AsyncSession,
        redis: Redis,
        username: str,
        password: str,
        client_ip: str,
        captcha_key: str,
        captcha_code: str,
        device_type: str = "web",
    ) -> dict:
        ip_fail_key = f"{LOGIN_FAIL_IP_PREFIX}{client_ip}"
        ip_fail_count_str = await redis.get(ip_fail_key)
        ip_fail_count = int(ip_fail_count_str) if ip_fail_count_str else 0
        if ip_fail_count >= settings.LOGIN_FAIL_MAX_ATTEMPTS:
            raise BusinessException(
                ResultCode.PASSWORD_ENTER_EXCEED_LIMIT,
                "IP登录失败次数过多，已临时锁定，请稍后重试",
            )

        fail_key = LOGIN_FAIL_PREFIX + username
        fail_count_str = await redis.get(fail_key)
        fail_count = int(fail_count_str) if fail_count_str else 0
        if fail_count >= settings.LOGIN_FAIL_MAX_ATTEMPTS:
            raise BusinessException(
                ResultCode.PASSWORD_ENTER_EXCEED_LIMIT,
                f"账号已被锁定，请{settings.LOGIN_FAIL_LOCK_MINUTES}分钟后再试",
            )

        # 验证码校验（下沉到 service 层，失败计入锁定计数）
        captcha_ok, captcha_expired = await self.verify_captcha_status(
            redis, captcha_key, captcha_code
        )
        if not captcha_ok:
            # 验证码失败计入锁定计数（T-AM-054），但错误码区分为 A0213/A0214
            code = (
                ResultCode.VERIFY_CODE_TIMEOUT if captcha_expired else ResultCode.VERIFY_CODE_ERROR
            )
            msg = "验证码已过期" if captcha_expired else "验证码错误"
            await self._fail_login(
                redis,
                fail_key,
                ip_fail_key,
                code=code,
                msg=msg,
            )

        user = await user_repository.get_by_username(db, username)

        if not user:
            await self._fail_login(redis, fail_key, ip_fail_key)
            raise BusinessException(ResultCode.USERNAME_OR_PASSWORD_ERROR, "用户名或密码错误")

        if user.password is None:
            await self._fail_login(redis, fail_key, ip_fail_key)
            raise BusinessException(ResultCode.USER_LOGIN_ERROR, "用户信息不完整")
        is_valid = await check_password_async(password, user.password)
        if not is_valid:
            await self._fail_login(redis, fail_key, ip_fail_key)

        if user.status != 1:
            raise BusinessException(ResultCode.USER_ACCOUNT_LOCKED, "用户已被禁用")

        roles = await user_repository.get_user_role_codes(db, user.id)

        from app.service.menu_service import menu_service

        data_scope = await role_repository.get_maximum_data_scope(db, roles)
        perms = await menu_service.list_role_perms(db, redis, set(roles))

        if user.username is None:
            raise BusinessException(ResultCode.USER_LOGIN_ERROR, "用户信息不完整")

        await redis.delete(fail_key)
        await redis.delete(ip_fail_key)

        session_id = str(uuid.uuid4())

        if settings.USE_MULTI_POINT:
            await self._enforce_device_limit(db, redis, session_id, user.id, roles)

        authorities = [f"ROLE_{r}" for r in roles] + list(perms)

        session_data = json.dumps(
            {
                "userId": user.id,
                "username": user.username,
                "nickname": user.nickname,
                "deptId": user.dept_id,
                "dataScope": data_scope,
                "authorities": authorities,
                "deviceType": device_type,
                "loginIp": client_ip,
                "loginTime": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                "lastAccessTime": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        await redis.setex(SESSION_PREFIX + session_id, SESSION_TTL, session_data)

        return {
            "sessionId": session_id,
            "user": {
                "id": user.id,
                "username": user.username,
                "nickname": user.nickname,
            },
        }

    async def _fail_login(
        self,
        redis: Redis,
        fail_key: str,
        ip_fail_key: str | None = None,
        code: ResultCode = ResultCode.USERNAME_OR_PASSWORD_ERROR,
        msg: str | None = None,
    ) -> None:
        """记录一次登录失败并递增锁定计数。

        默认以 A0210（用户名或密码错误）抛出；验证码类失败可传入
        code=A0213/A0214，使计数递增的同时返回对应错误码（T-AM-054 验证码错误计入失败计数）。
        """
        if ip_fail_key:
            ip_count = await redis.incr(ip_fail_key)
            if ip_count == 1:
                await redis.expire(ip_fail_key, settings.LOGIN_FAIL_LOCK_MINUTES * 60)

        count = await redis.incr(fail_key)
        if count == 1:
            await redis.expire(fail_key, settings.LOGIN_FAIL_LOCK_MINUTES * 60)
        remaining = settings.LOGIN_FAIL_MAX_ATTEMPTS - count
        if remaining <= 0:
            raise BusinessException(
                ResultCode.PASSWORD_ENTER_EXCEED_LIMIT,
                f"账号已被锁定，请{settings.LOGIN_FAIL_LOCK_MINUTES}分钟后再试",
            )
        raise BusinessException(
            code,
            msg or f"用户名或密码错误，剩余{remaining}次尝试机会",
        )

    async def _resolve_max_devices(self, db: AsyncSession, user_id: int) -> int:
        """按会员等级权益解析同时在线设备数上限（无会员记录按 level_0）。"""
        member = await member_repository.get_by_user_id(db, user_id)
        level_code = member.level_code if member else "level_0"
        benefit = await member_benefit_repository.get_by_level_code(db, level_code)
        return benefit.max_devices if benefit else DEFAULT_MAX_DEVICES

    async def _enforce_device_limit(
        self, db: AsyncSession, redis: Redis, session_id: str, user_id: int, roles: list[str]
    ) -> None:
        """多点登录控制（F-AM-011）：按同时在线设备数上限踢出最早登录的会话。

        会话索引 session:user:{userId} 为 ZSet（member=sessionId，score=登录 epoch 秒），
        三端共享同一 Redis 结构。超限时新会话保留，最早的若干会话被删除，其下一次请求
        因 session:{sessionId} 不存在而返回 401。
        """
        max_devices = (
            ADMIN_MAX_DEVICES
            if any(code in ("ROOT", "ADMIN") for code in roles)
            else await self._resolve_max_devices(db, user_id)
        )

        index_key = f"{SESSION_USER_PREFIX}{user_id}"
        await redis.zadd(index_key, {session_id: int(datetime.now(UTC).timestamp())})
        await redis.expire(index_key, SESSION_TTL)

        total = await redis.zcard(index_key)
        if total <= max_devices:
            return

        raw_members = await redis.zrange(index_key, 0, -1)
        members = [m.decode() if isinstance(m, bytes) else m for m in raw_members]
        # 排除本次新会话：同秒登录时 score 相同，按 member 字典序排序也可能把它排在前面
        candidates = [sid for sid in members if sid != session_id]
        evicted = candidates[: total - max_devices]
        if evicted:
            await redis.delete(*[f"{SESSION_PREFIX}{sid}" for sid in evicted])
            await redis.zrem(index_key, *evicted)

    async def register(
        self,
        db: AsyncSession,
        redis: Redis,
        username: str,
        password: str,
        nickname: str,
        captcha_key: str,
        captcha_code: str,
        client_ip: str = "unknown",
    ) -> dict:
        if not settings.REGISTER_ENABLED:
            raise BusinessException(ResultCode.OPERATION_NOT_ALLOW, "注册功能未开启")

        # 注册 IP 限流：60 秒内最多 10 次（用户注册设计.md §4.5）
        register_limit_key = f"rate:limit:register:{client_ip}"
        count = await redis.incr(register_limit_key)
        if count == 1:
            await redis.expire(register_limit_key, 60)
        if count > 10:
            raise BusinessException(
                ResultCode.REPEAT_SUBMIT_ERROR, "注册请求过于频繁，请60秒后再试"
            )

        stored_captcha = await redis.get(f"{settings.CAPTCHA_KEY_PREFIX}{captcha_key}")
        if not stored_captcha:
            raise BusinessException(ResultCode.VERIFY_CODE_TIMEOUT, "验证码已过期")
        if isinstance(stored_captcha, bytes):
            stored_captcha = stored_captcha.decode()
        if stored_captcha.lower() != captcha_code.lower():
            raise BusinessException(ResultCode.VERIFY_CODE_ERROR, "验证码错误")
        await redis.delete(f"{settings.CAPTCHA_KEY_PREFIX}{captcha_key}")

        username = username.lower().strip()
        from app.models.entity.sys_user import SysUser, SysUserRole

        if await user_repository.check_username_exists(db, username):
            raise BusinessException(ResultCode.DATA_EXISTS, "该用户名不可用")

        hashed = await hash_password_async(password)
        user = SysUser(
            username=username,
            nickname=nickname.strip(),
            password=hashed,
            gender=1,
            status=1,
            deleted=0,
        )
        db.add(user)
        try:
            await db.flush()
        except IntegrityError as e:
            # 先查后插存在并发窗口，数据库唯一键是最终防线，冲突按"用户名已注册"返回
            raise BusinessException(ResultCode.DATA_EXISTS, "该用户名不可用") from e

        guest_role = await role_repository.get_enabled_by_code(db, "GUEST")
        if guest_role:
            db.add(SysUserRole(user_id=user.id, role_id=guest_role.id))
            await db.flush()

        await member_repository.get_or_init_member(db, user.id)

        # 新用户注册赠送试用积分（AI 计费 F-MB-002 §2.2.3），同一事务保证余额与流水一致
        from app.service.billing.recharge_service import recharge_service

        await recharge_service.grant_trial_credits(db, user.id)

        data_scope = guest_role.data_scope if guest_role else 0

        session_id = str(uuid.uuid4())
        authorities = ["ROLE_GUEST"] if guest_role else []

        if settings.USE_MULTI_POINT:
            roles = [guest_role.code] if guest_role and guest_role.code else []
            await self._enforce_device_limit(db, redis, session_id, user.id, roles)

        session_data = json.dumps(
            {
                "userId": user.id,
                "username": user.username,
                "nickname": user.nickname,
                "deptId": None,
                "dataScope": data_scope,
                "authorities": authorities,
                "deviceType": "web",
                "loginIp": client_ip,
                "loginTime": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                "lastAccessTime": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        # 会话仅在用户记录落库后签发，避免事务失败时残留无效会话
        await db.commit()
        await redis.setex(SESSION_PREFIX + session_id, SESSION_TTL, session_data)

        return {
            "sessionId": session_id,
            "user": {"id": user.id, "username": user.username, "nickname": user.nickname},
        }

    @staticmethod
    def _parse_log_time(value: str | None) -> datetime | None:
        if not value:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
        return None

    @staticmethod
    def _log_query_kwargs(
        *,
        username: str | None,
        ip: str | None,
        status: int | None,
        device_type: str | None,
        start_time: str | None,
        end_time: str | None,
        user: UserContext | None,
    ) -> dict:
        """组装登录日志查询条件（分页查询与导出共用，含数据权限限定）。"""
        user_ids: list[int] | None = None
        if user is not None and not user.is_admin:
            # 普通用户仅可查看本人日志，即便传入他人 username 也强制限定本人（T-AM-117）
            user_ids = [user.id]
        return {
            "username": username,
            "ip": ip,
            "status": status,
            "device_type": device_type if device_type in DEVICE_TYPES else None,
            "start_time": AuthService._parse_log_time(start_time),
            "end_time": AuthService._parse_log_time(end_time),
            "user_ids": user_ids,
        }

    @staticmethod
    def _format_log_item(d: dict) -> dict:
        def _fmt_time(v) -> str:
            if isinstance(v, datetime):
                return v.strftime("%Y-%m-%d %H:%M:%S")
            return ""

        return {
            "id": str(d.get("_id", "")),
            "userId": d.get("user_id"),
            "username": d.get("username", ""),
            "ip": d.get("ip", ""),
            "location": d.get("location", ""),
            "browser": d.get("browser", ""),
            "os": d.get("os", ""),
            "deviceType": d.get("device_type") or "web",
            "status": d.get("status", 0),
            "message": d.get("message", ""),
            "loginTime": _fmt_time(d.get("create_time")),
        }

    async def list_login_logs(
        self,
        page_num: int,
        page_size: int,
        *,
        username: str | None = None,
        ip: str | None = None,
        status: int | None = None,
        device_type: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        user: UserContext | None = None,
    ) -> dict:
        """分页查询登录日志。

        - 普通用户仅能查看本人日志（user_ids 限定为当前用户）
        - 管理员（is_admin）可查看全量
        """
        kwargs = self._log_query_kwargs(
            username=username,
            ip=ip,
            status=status,
            device_type=device_type,
            start_time=start_time,
            end_time=end_time,
            user=user,
        )
        docs, total = await login_log_repository.page_logs(page_num, page_size, **kwargs)
        return {"list": [self._format_log_item(d) for d in docs], "total": total}

    async def export_login_logs(
        self,
        *,
        username: str | None = None,
        ip: str | None = None,
        status: int | None = None,
        device_type: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        user: "UserContext | None" = None,
    ) -> bytes:
        """按当前筛选条件导出登录日志 Excel（数据权限与分页查询一致）。"""
        import io

        from openpyxl import Workbook

        kwargs = self._log_query_kwargs(
            username=username,
            ip=ip,
            status=status,
            device_type=device_type,
            start_time=start_time,
            end_time=end_time,
            user=user,
        )
        docs = await login_log_repository.list_logs(**kwargs)

        headers = ["用户名", "IP", "登录时间", "状态", "提示信息", "设备类型", "浏览器", "操作系统"]
        wb = Workbook()
        ws = wb.active
        if ws is None:
            raise BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "导出登录日志失败")
        ws.title = "登录日志"
        ws.append(headers)
        for d in docs:
            item = self._format_log_item(d)
            ws.append(
                [
                    item["username"],
                    item["ip"],
                    item["loginTime"],
                    "成功" if item["status"] == 1 else "失败",
                    item["message"],
                    item["deviceType"],
                    item["browser"],
                    item["os"],
                ]
            )
        output = io.BytesIO()
        wb.save(output)
        return output.getvalue()

    async def list_sessions(self, redis: Redis, username: str) -> list[dict]:
        """在线会话列表（F-AM-011）：扫描 session:* 并按用户名精确过滤。"""
        sessions: list[dict] = []
        async for key in redis.scan_iter(match=f"{SESSION_PREFIX}*"):
            key_str = key.decode() if isinstance(key, bytes) else key
            if key_str.startswith(SESSION_USER_PREFIX):
                continue
            raw = await redis.get(key_str)
            if not raw:
                continue
            data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            if data.get("username") != username:
                continue
            login_time = data.get("loginTime", "")
            sessions.append(
                {
                    "sessionId": key_str[len(SESSION_PREFIX) :],
                    "username": data.get("username", ""),
                    "deviceType": data.get("deviceType") or "web",
                    "loginTime": login_time,
                    "ip": data.get("loginIp", ""),
                    "lastAccessTime": data.get("lastAccessTime") or login_time,
                }
            )
        sessions.sort(key=lambda s: s["loginTime"], reverse=True)
        return sessions

    async def kick_session(self, redis: Redis, session_id: str) -> None:
        """踢出指定在线会话（F-AM-011）。超级管理员会话不可被踢出。"""
        raw = await redis.get(SESSION_PREFIX + session_id)
        if not raw:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在或已过期")
        data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        if "ROLE_ROOT" in (data.get("authorities") or []):
            raise BusinessException(ResultCode.OPERATION_NOT_ALLOW, "超级管理员会话不可被踢出")

        await redis.delete(SESSION_PREFIX + session_id)
        user_id = data.get("userId")
        if user_id:
            await redis.zrem(f"{SESSION_USER_PREFIX}{int(user_id)}", session_id)

    async def kick_user_sessions(
        self,
        redis: Redis,
        user_id: int,
        role_codes: list[str] | None = None,
    ) -> None:
        """踢出目标用户全部在线会话并清理其角色权限缓存。

        供用户模块禁用/删除/重置密码联动调用：不依赖多点登录索引（索引仅在
        USE_MULTI_POINT 开启时维护），直接扫描会话空间匹配 userId；角色权限
        缓存一并失效，确保权限/凭据变更即时生效。
        """
        await self.kick_user_sessions_batch(redis, [user_id])
        for code in role_codes or []:
            await CacheService(redis).delete(f"role:perms:{code}")

    async def kick_user_sessions_batch(self, redis: Redis, user_ids: list[int]) -> int:
        """批量踢出多个用户的全部在线会话（单次扫描），返回踢出的会话数。

        超级管理员会话不可被踢出（与 kick_session 语义一致）。
        """
        target_ids = {int(uid) for uid in user_ids}
        if not target_ids:
            return 0

        session_keys: list[str] = []
        kicked_user_ids: set[int] = set()
        async for key in redis.scan_iter(match=f"{SESSION_PREFIX}*"):
            key_str = key.decode() if isinstance(key, bytes) else key
            if key_str.startswith(SESSION_USER_PREFIX):
                continue
            raw = await redis.get(key_str)
            if not raw:
                continue
            data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            user_id = data.get("userId")
            if user_id not in target_ids or "ROLE_ROOT" in (data.get("authorities") or []):
                continue
            session_keys.append(key_str)
            kicked_user_ids.add(int(user_id))

        if session_keys:
            await redis.delete(*session_keys)
        for uid in kicked_user_ids:
            await redis.delete(f"{SESSION_USER_PREFIX}{uid}")
        return len(session_keys)

    async def change_password(
        self,
        db: AsyncSession,
        redis: Redis,
        user: UserContext,
        old_password: str,
        new_password: str,
    ) -> None:
        """个人改密（登录态）：旧密码校验 + 新密码复杂度校验，成功后踢出本人全部在线会话。"""
        sys_user = await user_repository.get_by_id(db, user.id)
        if (
            sys_user is None
            or not sys_user.password
            or not await check_password_async(old_password, sys_user.password)
        ):
            raise BusinessException(ResultCode.USERNAME_OR_PASSWORD_ERROR, "旧密码错误")

        from app.service.user_service import validate_password_complexity

        is_valid, error_msg = validate_password_complexity(new_password)
        if not is_valid:
            raise BusinessException(ResultCode.PARAM_ERROR, error_msg)

        sys_user.password = await hash_password_async(new_password)

        role_codes = await user_repository.get_user_role_codes(db, user.id)
        await self.kick_user_sessions(redis, user.id, role_codes)

    async def get_captcha(self, redis: Redis) -> dict:
        captcha_text = "".join(
            secrets.choice(string.ascii_uppercase + string.digits)
            for _ in range(settings.CAPTCHA_LENGTH)
        )

        img_str = await asyncio.to_thread(self._generate_captcha_image, captcha_text)

        captcha_key = str(uuid.uuid4())

        await redis.setex(
            f"{settings.CAPTCHA_KEY_PREFIX}{captcha_key}", settings.CAPTCHA_EXPIRES, captcha_text
        )

        return {
            "captchaKey": captcha_key,
            "captchaBase64": f"data:image/jpeg;base64,{img_str}",
        }

    def _generate_captcha_image(self, captcha_text: str) -> str:
        image = Image.new(
            "RGB",
            (settings.CAPTCHA_WIDTH, settings.CAPTCHA_HEIGHT),
            color=(255, 255, 255),
        )
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", settings.CAPTCHA_FONT_SIZE)
        except OSError:
            # arial.ttf 在多数 Linux 服务端不存在，回退 PIL 内置字体属预期降级；
            # 记 debug 便于排查"验证码字体与预期不符"，不作为异常处理
            logger.debug("验证码字体 arial.ttf 不可用，回退 PIL 默认字体")
            font = ImageFont.load_default()

        draw.text((20, 10), captcha_text, fill=(0, 0, 0), font=font)

        for _ in range(settings.CAPTCHA_NOISE_LINES):
            x1 = secrets.randbelow(settings.CAPTCHA_WIDTH)
            y1 = secrets.randbelow(settings.CAPTCHA_HEIGHT)
            x2 = secrets.randbelow(settings.CAPTCHA_WIDTH)
            y2 = secrets.randbelow(settings.CAPTCHA_HEIGHT)
            draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=1)

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    async def verify_captcha_status(
        self, redis: Redis, captcha_key: str, captcha_code: str
    ) -> tuple[bool, bool]:
        """校验验证码，返回 (是否通过, 是否已过期)。

        用 GETDEL 原子取走验证码：并发提交同一 captchaKey 时仅一个请求能取到，
        杜绝验证码被并发重放；比对失败同样作废，需重新获取（与 Java 端"校验时
        无条件删除"语义一致，前端验证码错误后本就会刷新验证码）。
        expired=True 表示验证码 Key 不存在（未生成/已消费/超时），
        用于区分 A0214（验证码错误）与 A0213（验证码已过期）。
        """
        stored_captcha = await redis.getdel(f"{settings.CAPTCHA_KEY_PREFIX}{captcha_key}")

        if not stored_captcha:
            return False, True

        if isinstance(stored_captcha, bytes):
            stored_captcha = stored_captcha.decode()

        return stored_captcha.lower() == captcha_code.lower(), False


auth_service = AuthService()
