import asyncio
import base64
import json
import secrets
import string
import uuid
from io import BytesIO

from app.config import settings
from app.repository.user_repository import user_repository
from app.utils.password import check_password_async
from PIL import Image, ImageDraw, ImageFont
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

SESSION_PREFIX = "session:"
SESSION_TTL = 7 * 24 * 3600
SESSION_COOKIE = "X-Session-Id"


class AuthService:
    @staticmethod
    async def login(
        db: AsyncSession,
        redis: Redis,
        username: str,
        password: str,
    ) -> dict:
        user = await user_repository.get_by_username(db, username)

        if not user:
            raise ValueError("用户名或密码错误")

        if user.password is None:
            raise ValueError("用户密码未设置")
        is_valid = await check_password_async(password, user.password)
        if not is_valid:
            raise ValueError("用户名或密码错误")

        if user.status != 1:
            raise ValueError("用户已被禁用")

        roles = await user_repository.get_user_role_codes(db, user.id)

        from app.repository.role_repository import role_repository
        from app.service.menu_service import MenuService
        data_scope = await role_repository.get_maximum_data_scope(db, roles)
        perms = await MenuService.list_role_perms(db, redis, set(roles))

        if user.username is None:
            raise ValueError("用户信息不完整")

        session_id = str(uuid.uuid4())

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
    async def get_captcha(redis: Redis) -> dict:
        captcha_text = "".join(
            secrets.choice(string.ascii_uppercase + string.digits)
            for _ in range(settings.CAPTCHA_LENGTH)
        )

        img_str = await asyncio.to_thread(
            AuthService._generate_captcha_image, captcha_text
        )

        captcha_key = str(uuid.uuid4())

        await redis.setex(f"captcha:{captcha_key}", settings.CAPTCHA_EXPIRES, captcha_text)

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
        stored_captcha = await redis.get(f"captcha:{captcha_key}")

        if not stored_captcha:
            return False

        if isinstance(stored_captcha, bytes):
            stored_captcha = stored_captcha.decode()

        result = stored_captcha.lower() == captcha_code.lower()

        if result:
            await redis.delete(f"captcha:{captcha_key}")

        return result
