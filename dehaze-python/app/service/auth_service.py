import asyncio
import base64
import json
import secrets
import string
import uuid
from io import BytesIO

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.repository.user_repository import user_repository
from app.utils.password import check_password_async, hash_password_async
from PIL import Image, ImageDraw, ImageFont
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

SESSION_PREFIX = "session:"
SESSION_USER_PREFIX = "session:user:"
SESSION_TTL = 7 * 24 * 3600
SESSION_COOKIE = "X-Session-Id"

LOGIN_FAIL_PREFIX = "login:fail:"
LOGIN_FAIL_IP_PREFIX = "login:fail:ip:"


class AuthService:
    @staticmethod
    async def login(
        db: AsyncSession,
        redis: Redis,
        username: str,
        password: str,
        client_ip: str = "unknown",
        captcha_key: str = "",
        captcha_code: str = "",
    ) -> dict:
        # IP 纬度锁定检查
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
        if not await AuthService.verify_captcha(redis, captcha_key, captcha_code):
            await AuthService._fail_login(redis, fail_key, ip_fail_key)
            raise BusinessException(ResultCode.VERIFY_CODE_ERROR, "验证码错误")

        user = await user_repository.get_by_username(db, username)

        if not user:
            await AuthService._fail_login(redis, fail_key, ip_fail_key)
            raise BusinessException(ResultCode.USERNAME_OR_PASSWORD_ERROR, "用户名或密码错误")

        if user.password is None:
            await AuthService._fail_login(redis, fail_key, ip_fail_key)
            raise BusinessException(ResultCode.USER_LOGIN_ERROR, "用户信息不完整")
        is_valid = await check_password_async(password, user.password)
        if not is_valid:
            await AuthService._fail_login(redis, fail_key, ip_fail_key)

        if user.status != 1:
            raise BusinessException(ResultCode.USER_ACCOUNT_LOCKED, "用户已被禁用")

        roles = await user_repository.get_user_role_codes(db, user.id)

        from app.repository.role_repository import role_repository
        from app.service.menu_service import MenuService
        data_scope = await role_repository.get_maximum_data_scope(db, roles)
        perms = await MenuService.list_role_perms(db, redis, set(roles))

        if user.username is None:
            raise BusinessException(ResultCode.USER_LOGIN_ERROR, "用户信息不完整")

        await redis.delete(fail_key)
        await redis.delete(ip_fail_key)

        session_id = str(uuid.uuid4())

        if settings.USE_MULTI_POINT:
            await AuthService._handle_multi_point_session(redis, session_id, user.username)

        authorities = [f"ROLE_{r}" for r in roles] + list(perms)

        session_data = json.dumps({
            "userId": user.id,
            "username": user.username,
            "nickname": user.nickname,
            "deptId": user.dept_id,
            "dataScope": data_scope,
            "authorities": authorities,
        })

        await redis.setex(SESSION_PREFIX + session_id, SESSION_TTL, session_data)

        return {
            "sessionId": session_id,
            "user": {
                "id": user.id,
                "username": user.username,
                "nickname": user.nickname,
            },
        }

    @staticmethod
    async def _fail_login(redis: Redis, fail_key: str, ip_fail_key: str = None) -> None:
        # 递增 IP 纬度计数
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
            ResultCode.USERNAME_OR_PASSWORD_ERROR,
            f"用户名或密码错误，剩余{remaining}次尝试机会",
        )

    @staticmethod
    async def _handle_multi_point_session(redis: Redis, new_session_id: str, username: str) -> None:
        """多点登录控制：删除同一用户名下的旧 Session，仅保留最新。"""
        user_session_key = f"{SESSION_USER_PREFIX}{username}"
        old_session_id = await redis.get(user_session_key)
        if old_session_id:
            old_session_id_str = old_session_id.decode() if isinstance(old_session_id, bytes) else old_session_id
            if old_session_id_str:
                await redis.delete(f"{SESSION_PREFIX}{old_session_id_str}")
        await redis.setex(user_session_key, SESSION_TTL, new_session_id)

    @staticmethod
    async def register(
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
        from sqlalchemy import select, exists
        from app.models.entity.sys_user import SysRole, SysUserRole, SysUser
        dup_result = await db.execute(
            select(exists().where(SysUser.username == username))
        )
        if dup_result.scalar():
            raise BusinessException(ResultCode.DATA_EXISTS, "该用户名不可用")

        hashed = await hash_password_async(password)
        user = SysUser(username=username, nickname=nickname.strip(), password=hashed,
                       gender=1, status=1, deleted=0)
        db.add(user)
        await db.flush()

        guest_result = await db.execute(
            select(SysRole).where(SysRole.code == "GUEST", SysRole.status == 1, SysRole.deleted == 0)
        )
        guest_role = guest_result.scalar()
        if guest_role:
            db.add(SysUserRole(user_id=user.id, role_id=guest_role.id))
            await db.flush()

        from app.repository.member_repository import member_repository
        # upsert 会员记录：冲突时复活（降级为 level_0、清空月度配额；保留 total_consumption）
        await member_repository.get_or_init_member(db, user.id)

        data_scope = guest_role.data_scope if guest_role else 0

        session_id = str(uuid.uuid4())
        authorities = ["ROLE_GUEST"] if guest_role else []

        session_data = json.dumps({
            "userId": user.id,
            "username": user.username,
            "nickname": user.nickname,
            "deptId": None,
            "dataScope": data_scope,
            "authorities": authorities,
        })

        await redis.setex(f"session:{session_id}", SESSION_TTL, session_data)

        await db.commit()

        return {
            "sessionId": session_id,
            "user": {"id": user.id, "username": user.username, "nickname": user.nickname},
        }

    @staticmethod
    async def get_captcha(redis: Redis) -> dict:
        captcha_text = "".join(
            secrets.choice(string.ascii_uppercase + string.digits)
            for _ in range(settings.CAPTCHA_LENGTH)
        )

        img_str = await asyncio.to_thread(
            AuthService._generate_captcha_image, captcha_text
        )

        captcha_key = str(uuid.uuid4())

        await redis.setex(f"{settings.CAPTCHA_KEY_PREFIX}{captcha_key}", settings.CAPTCHA_EXPIRES, captcha_text)

        return {
            "captchaKey": captcha_key,
            "captchaBase64": f"data:image/jpeg;base64,{img_str}",
        }

    @staticmethod
    def _generate_captcha_image(captcha_text: str) -> str:
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

    @staticmethod
    async def verify_captcha(redis: Redis, captcha_key: str, captcha_code: str) -> bool:
        stored_captcha = await redis.get(f"{settings.CAPTCHA_KEY_PREFIX}{captcha_key}")

        if not stored_captcha:
            return False

        if isinstance(stored_captcha, bytes):
            stored_captcha = stored_captcha.decode()

        result = stored_captcha.lower() == captcha_code.lower()

        if result:
            await redis.delete(f"{settings.CAPTCHA_KEY_PREFIX}{captcha_key}")

        return result
