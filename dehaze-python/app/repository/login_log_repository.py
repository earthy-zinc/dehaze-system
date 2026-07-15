"""
登录日志 Repository
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_log import SysLoginLog
from app.repository.base import BaseRepository


class LoginLogRepository(BaseRepository[SysLoginLog]):
    """登录日志 Repository"""

    model = SysLoginLog

    async def create_log(
        self,
        db: AsyncSession,
        user_id: int | None,
        username: str,
        ip: str,
        status: int,
        message: str,
        browser: str = "",
        os: str = "",
        location: str = "",
    ) -> SysLoginLog:
        """
        创建登录日志

        Args:
            db: 数据库会话
            user_id: 用户ID（登录失败时可能为 None）
            username: 登录用户名
            ip: 登录IP
            status: 登录状态(1:成功;0:失败)
            message: 登录消息
            browser: 浏览器类型
            os: 操作系统
            location: 登录地点

        Returns:
            创建的日志记录
        """
        log = SysLoginLog(
            user_id=user_id,
            username=username,
            ip=ip,
            status=status,
            message=message,
            browser=browser,
            os=os,
            location=location,
        )
        return await self.create(db, log)



login_log_repository = LoginLogRepository()
