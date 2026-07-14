"""
认证服务

提供用户登录、验证码生成等功能
"""

import asyncio
import base64
import secrets
import string
import uuid
from io import BytesIO

from app.config import settings
from app.repository.user_repository import user_repository
from app.utils.jwt import JWTUtils
from app.utils.password import check_password_async
from PIL import Image, ImageDraw, ImageFont
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession


class AuthService:
    @staticmethod
    async def login(
        db: AsyncSession,
        redis: Redis,
        username: str,
        password: str,
    ) -> dict:
        """
        用户登录

        Args:
            db: 异步数据库会话
            redis: Redis 客户端（用于权限缓存）
            username: 用户名
            password: 密码

        Returns:
            登录结果，包含 token

        Raises:
            ValueError: 用户名或密码错误
        """
        # 查询用户
        user = await user_repository.get_by_username(db, username)

        if not user:
            raise ValueError("用户名或密码错误")

        # 验证密码（异步执行，避免阻塞事件循环）
        if user.password is None:
            raise ValueError("用户密码未设置")
        is_valid = await check_password_async(password, user.password)
        if not is_valid:
            raise ValueError("用户名或密码错误")

        # 检查用户状态
        if user.status != 1:
            raise ValueError("用户已被禁用")

        # 查询用户角色和权限（权限走 Redis 缓存）
        roles = await user_repository.get_user_role_codes(db, user.id)
        from app.service.menu_service import MenuService
        permissions = await MenuService.list_role_perms(db, redis, roles)

        # 使用 JWT 工具类生成 Token
        if user.username is None or user.nickname is None:
            raise ValueError("用户信息不完整")
        access_token = JWTUtils.create_access_token(
            user_id=user.id,
            username=user.username,
            nickname=user.nickname,
            roles=roles,
            permissions=permissions,
        )

        return {
            "tokenType": "Bearer",
            "accessToken": access_token,
            "user": {
                "id": user.id,
                "username": user.username,
                "nickname": user.nickname,
            },
        }

    @staticmethod
    async def get_captcha(redis: Redis) -> dict:
        """
        获取验证码

        Args:
            redis: Redis 异步客户端

        Returns:
            验证码信息
        """
        # 生成验证码文本（使用 secrets 确保随机性）
        captcha_text = "".join(
            secrets.choice(string.ascii_uppercase + string.digits)
            for _ in range(settings.CAPTCHA_LENGTH)
        )

        # 生成验证码图片（PIL 是同步 CPU 密集型操作，移至线程池避免阻塞事件循环）
        img_str = await asyncio.to_thread(
            AuthService._generate_captcha_image, captcha_text
        )

        # 生成验证码 key
        captcha_key = str(uuid.uuid4())

        # 存储到 Redis
        await redis.setex(f"captcha:{captcha_key}", settings.CAPTCHA_EXPIRES, captcha_text)

        return {
            "captchaKey": captcha_key,
            "captchaBase64": f"data:image/jpeg;base64,{img_str}",
        }

    @staticmethod
    def _generate_captcha_image(captcha_text: str) -> str:
        """同步生成验证码图片并返回 base64 字符串（供 asyncio.to_thread 调用）"""
        image = Image.new(
            "RGB",
            (settings.CAPTCHA_WIDTH, settings.CAPTCHA_HEIGHT),
            color=(255, 255, 255),
        )
        draw = ImageDraw.Draw(image)

        # 使用默认字体
        try:
            font = ImageFont.truetype("arial.ttf", settings.CAPTCHA_FONT_SIZE)
        except OSError:
            font = ImageFont.load_default()

        draw.text((20, 10), captcha_text, fill=(0, 0, 0), font=font)

        # 添加干扰线（使用 secrets 生成随机坐标）
        for _ in range(settings.CAPTCHA_NOISE_LINES):
            x1 = secrets.randbelow(settings.CAPTCHA_WIDTH)
            y1 = secrets.randbelow(settings.CAPTCHA_HEIGHT)
            x2 = secrets.randbelow(settings.CAPTCHA_WIDTH)
            y2 = secrets.randbelow(settings.CAPTCHA_HEIGHT)
            draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=1)

        # 转换为 base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    @staticmethod
    async def verify_captcha(redis: Redis, captcha_key: str, captcha_code: str) -> bool:
        """
        验证验证码

        Args:
            redis: Redis 异步客户端
            captcha_key: 验证码 key
            captcha_code: 用户输入的验证码

        Returns:
            验证结果
        """
        stored_captcha = await redis.get(f"captcha:{captcha_key}")

        if not stored_captcha:
            return False

        # 比较验证码（不区分大小写）
        if isinstance(stored_captcha, bytes):
            stored_captcha = stored_captcha.decode()

        result = stored_captcha.lower() == captcha_code.lower()

        # 验证后删除
        if result:
            await redis.delete(f"captcha:{captcha_key}")

        return result

    @staticmethod
    async def refresh_token(
        db: AsyncSession,
        user_id: int,
        redis: Redis,
    ) -> dict:
        """
        刷新访问令牌

        Args:
            db: 异步数据库会话
            user_id: 用户ID
            redis: Redis 客户端

        Returns:
            新的访问令牌

        Raises:
            ValueError: 用户不存在或已禁用
        """
        # 验证用户状态
        user = await user_repository.get_by_id(db, user_id)
        if not user:
            raise ValueError("用户不存在")
        if user.status != 1:
            raise ValueError("用户已被禁用")

        # 查询用户角色和权限（权限走 Redis 缓存）
        roles = await user_repository.get_user_role_codes(db, user.id)
        from app.service.menu_service import MenuService
        permissions = await MenuService.list_role_perms(db, redis, roles)

        # 使用 JWT 工具类生成 Token
        if user.username is None or user.nickname is None:
            raise ValueError("用户信息不完整")
        access_token = JWTUtils.create_access_token(
            user_id=user.id,
            username=user.username,
            nickname=user.nickname,
            roles=roles,
            permissions=permissions,
        )

        return {
            "tokenType": "Bearer",
            "accessToken": access_token,
            "user": {
                "id": user.id,
                "username": user.username,
                "nickname": user.nickname,
            },
        }
