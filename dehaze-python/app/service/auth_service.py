import asyncio
import base64
import json
import secrets
import string
import uuid
from datetime import UTC, datetime
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.repository.user_repository import user_repository
from app.utils.password import check_password_async, hash_password_async

SESSION_PREFIX = "session:"
SESSION_USER_PREFIX = "session:user:"
SESSION_TTL = 7 * 24 * 3600
SESSION_COOKIE = "X-Session-Id"

LOGIN_FAIL_PREFIX = "login:fail:"
LOGIN_FAIL_IP_PREFIX = "login:fail:ip:"


class AuthService:
    async def login(self, 
        db: AsyncSession,
        redis: Redis,
        username: str,
        password: str,
        client_ip: str = "unknown",
        captcha_key: str = "",
        captcha_code: str = "",
        user_agent: str = "",
    ) -> dict:
        """登录（含审计：成功/失败均落 sys_login_log，异常路径也覆盖）"""
        from app.repository.login_log_repository import login_log_repository
        from app.utils.user_agent import parse_user_agent

        browser, os_name = parse_user_agent(user_agent)
        try:
            result = await self._authenticate(
                db, redis, username, password, client_ip, captcha_key, captcha_code
            )
        except BusinessException as e:
            await login_log_repository.create_log(
                db, None, username, client_ip, 0, e.message, browser, os_name
            )
            raise
        user = result.get("user")
        if not user:
            # 登录成功但响应缺失用户结构属内部异常，不得静默记为匿名，显式暴露
            raise BusinessException(
                ResultCode.SYSTEM_EXECUTION_ERROR, "登录成功但用户信息缺失"
            )
        await login_log_repository.create_log(
            db, user.get("id"), username, client_ip, 1, "登录成功", browser, os_name
        )
        return result

    async def _authenticate(self, 
        db: AsyncSession,
        redis: Redis,
        username: str,
        password: str,
        client_ip: str,
        captcha_key: str,
        captcha_code: str,
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
            code = ResultCode.VERIFY_CODE_TIMEOUT if captcha_expired else ResultCode.VERIFY_CODE_ERROR
            msg = "验证码已过期" if captcha_expired else "验证码错误"
            await self._fail_login(
                redis, fail_key, ip_fail_key, code=code, msg=msg,
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

        from app.repository.role_repository import role_repository
        from app.service.menu_service import menu_service

        data_scope = await role_repository.get_maximum_data_scope(db, roles)
        perms = await menu_service.list_role_perms(db, redis, set(roles))

        if user.username is None:
            raise BusinessException(ResultCode.USER_LOGIN_ERROR, "用户信息不完整")

        await redis.delete(fail_key)
        await redis.delete(ip_fail_key)

        session_id = str(uuid.uuid4())

        if settings.USE_MULTI_POINT:
            await self._handle_multi_point_session(redis, session_id, user.username)

        authorities = [f"ROLE_{r}" for r in roles] + list(perms)

        session_data = json.dumps(
            {
                "userId": user.id,
                "username": user.username,
                "nickname": user.nickname,
                "deptId": user.dept_id,
                "dataScope": data_scope,
                "authorities": authorities,
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

    async def _fail_login(self, 
        redis: Redis,
        fail_key: str,
        ip_fail_key: str = None,
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

    async def _handle_multi_point_session(self, redis: Redis, new_session_id: str, username: str) -> None:
        """多点登录控制：删除同一用户名下的旧 Session，仅保留最新。"""
        user_session_key = f"{SESSION_USER_PREFIX}{username}"
        old_session_id = await redis.get(user_session_key)
        if old_session_id:
            old_session_id_str = (
                old_session_id.decode() if isinstance(old_session_id, bytes) else old_session_id
            )
            if old_session_id_str:
                await redis.delete(f"{SESSION_PREFIX}{old_session_id_str}")
        await redis.setex(user_session_key, SESSION_TTL, new_session_id)

    async def register(self, 
        db: AsyncSession,
        redis: Redis,
        username: str,
        password: str,
        nickname: str,
        captcha_key: str,
        captcha_code: str,
    ) -> dict:
        from app.core.code import ResultCode
        from app.core.exceptions import BusinessException

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
        from app.repository.role_repository import role_repository
        from app.repository.user_repository import user_repository

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
        await db.flush()

        guest_role = await role_repository.get_enabled_by_code(db, "GUEST")
        if guest_role:
            db.add(SysUserRole(user_id=user.id, role_id=guest_role.id))
            await db.flush()

        from app.repository.member_repository import member_repository

        # upsert 会员记录：冲突时复活（降级为 level_0、清空月度配额；保留 total_consumption）
        await member_repository.get_or_init_member(db, user.id)

        # 新用户注册赠送试用积分（AI 计费 F-MB-002 §2.2.3），同一事务保证余额与流水一致
        from app.service.billing.recharge_service import recharge_service

        await recharge_service.grant_trial_credits(db, user.id)

        data_scope = guest_role.data_scope if guest_role else 0

        session_id = str(uuid.uuid4())
        authorities = ["ROLE_GUEST"] if guest_role else []

        session_data = json.dumps(
            {
                "userId": user.id,
                "username": user.username,
                "nickname": user.nickname,
                "deptId": None,
                "dataScope": data_scope,
                "authorities": authorities,
            }
        )

        await redis.setex(f"session:{session_id}", SESSION_TTL, session_data)

        await db.commit()

        return {
            "sessionId": session_id,
            "user": {"id": user.id, "username": user.username, "nickname": user.nickname},
        }

    async def list_login_logs(self, 
        page_num: int,
        page_size: int,
        *,
        username: str | None = None,
        ip: str | None = None,
        status: int | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        user: "UserContext | None" = None,
    ) -> dict:
        """分页查询登录日志。

        - 普通用户仅能查看本人日志（user_ids 限定为当前用户）
        - 管理员（is_admin）可查看全量
        """
        from app.repository.login_log_repository import login_log_repository

        def _parse_dt(value: str | None) -> datetime | None:
            if not value:
                return None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(value, fmt).replace(tzinfo=UTC)
                except ValueError:
                    continue
            return None

        user_ids: list[int] | None = None
        if user is not None and not user.is_admin:
            # 普通用户仅可查看本人日志，即便传入他人 username 也强制限定本人（T-AM-117）
            user_ids = [user.id]

        docs, total = await login_log_repository.page_logs(
            page_num,
            page_size,
            username=username,
            ip=ip,
            status=status,
            start_time=_parse_dt(start_time),
            end_time=_parse_dt(end_time),
            user_ids=user_ids,
        )

        def _fmt_time(v) -> str:
            if isinstance(v, datetime):
                return v.strftime("%Y-%m-%d %H:%M:%S")
            return ""

        items = []
        for d in docs:
            items.append(
                {
                    "id": str(d.get("_id", "")),
                    "userId": d.get("user_id"),
                    "username": d.get("username", ""),
                    "ip": d.get("ip", ""),
                    "location": d.get("location", ""),
                    "browser": d.get("browser", ""),
                    "os": d.get("os", ""),
                    "status": d.get("status", 0),
                    "message": d.get("message", ""),
                    "loginTime": _fmt_time(d.get("create_time")),
                }
            )
        return {"list": items, "total": total}

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

    async def verify_captcha(self, redis: Redis, captcha_key: str, captcha_code: str) -> bool:
        stored_captcha = await redis.get(f"{settings.CAPTCHA_KEY_PREFIX}{captcha_key}")

        if not stored_captcha:
            return False

        if isinstance(stored_captcha, bytes):
            stored_captcha = stored_captcha.decode()

        result = stored_captcha.lower() == captcha_code.lower()

        if result:
            await redis.delete(f"{settings.CAPTCHA_KEY_PREFIX}{captcha_key}")

        return result

    async def verify_captcha_status(self, 
        redis: Redis, captcha_key: str, captcha_code: str
    ) -> tuple[bool, bool]:
        """校验验证码，返回 (是否通过, 是否已过期)。

        expired=True 表示验证码 Key 不存在（未生成/已消费/超时），
        用于区分 A0214（验证码错误）与 A0213（验证码已过期）。
        """
        stored_captcha = await redis.get(f"{settings.CAPTCHA_KEY_PREFIX}{captcha_key}")

        if not stored_captcha:
            return False, True

        if isinstance(stored_captcha, bytes):
            stored_captcha = stored_captcha.decode()

        if stored_captcha.lower() != captcha_code.lower():
            return False, False

        await redis.delete(f"{settings.CAPTCHA_KEY_PREFIX}{captcha_key}")
        return True, False


auth_service = AuthService()
