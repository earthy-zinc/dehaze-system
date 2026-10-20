"""
用户服务

提供用户 CRUD 功能，支持角色分配等
"""

import re
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.auth import UserContext
from app.models.base import get_current_user_id
from app.models.entity.sys_member import QUOTA_TASK_TYPES
from app.models.entity.sys_user import SysUser
from app.repository.dept_repository import DeptRepository, dept_repository
from app.repository.member_repository import member_repository
from app.repository.mongo_audit_log_repository import (
    MongoAuditLogRepository,
    mongo_audit_log_repository,
)
from app.repository.user_repository import UserRepository, user_repository
from app.utils.password import hash_password_async


def validate_password_complexity(password: str) -> tuple[bool, str]:
    """
    验证密码复杂度（8-20 位，至少包含字母和数字）

    Args:
        password: 待验证的密码

    Returns:
        (是否通过, 错误信息)
    """
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        return False, f"密码长度不能少于 {settings.PASSWORD_MIN_LENGTH} 位"

    if len(password) > settings.PASSWORD_MAX_LENGTH:
        return False, f"密码长度不能超过 {settings.PASSWORD_MAX_LENGTH} 位"

    if settings.PASSWORD_REQUIRE_COMPLEXITY:
        has_letter = bool(re.search(r"[a-zA-Z]", password))
        has_digit = bool(re.search(r"\d", password))

        if not (has_letter and has_digit):
            return False, "密码必须包含字母和数字"

    return True, ""


class UserService:
    """用户服务"""

    def __init__(
        self,
        repo: UserRepository = user_repository,
        dept_repo: DeptRepository = dept_repository,
        audit_repo: MongoAuditLogRepository = mongo_audit_log_repository,
    ):
        self.repo = repo
        self.dept_repo = dept_repo
        self.audit_repo = audit_repo

    async def get_user_list(
        self,
        db: AsyncSession,
        page: int,
        page_size: int,
        keywords: str | None = None,
        status: int | None = None,
        dept_id: int | None = None,
        create_time_start: str | None = None,
        create_time_end: str | None = None,
        current_user: UserContext | None = None,
    ) -> tuple[list[dict], int]:
        """
        获取用户列表（分页）

        Args:
            db: 异步数据库会话
            page: 页码
            page_size: 每页数量
            keywords: 关键词搜索
            status: 状态筛选
            dept_id: 部门ID筛选（包含下级部门）
            create_time_start: 创建时间开始
            create_time_end: 创建时间结束
            current_user: 当前登录用户（用于行级数据权限过滤）

        Returns:
            (用户列表, 总数)
        """
        dept_ids = None
        if dept_id:
            dept_ids = await self.dept_repo.get_children_ids(db, dept_id)

        users, total = await self.repo.get_user_list(
            db,
            page=page,
            page_size=page_size,
            keywords=keywords,
            status=status,
            dept_ids=dept_ids,
            create_time_start=create_time_start,
            create_time_end=create_time_end,
            current_user=current_user,
        )

        # 会员字段批量聚合（in user_ids 单次查询，禁止 N+1）
        member_map = await member_repository.get_by_user_ids(db, [u["id"] for u in users])
        for u in users:
            member = member_map.get(u["id"])
            u["memberLevel"] = member.level_code if member else None
            u["memberExpireTime"] = (
                member.expire_time.strftime("%Y-%m-%d %H:%M:%S")
                if member and member.expire_time
                else None
            )
            if member:
                used = sum(getattr(member, f"monthly_{t}_used") for t in QUOTA_TASK_TYPES)
                quota = sum(getattr(member, f"monthly_{t}_quota") for t in QUOTA_TASK_TYPES)
                u["quotaUsage"] = f"{used}/{quota}"
            else:
                u["quotaUsage"] = "0/0"

        return users, total

    async def get_user_form_data(self, db: AsyncSession, user_id: int) -> dict[str, Any] | None:
        """
        获取用户表单数据

        Args:
            db: 异步数据库会话
            user_id: 用户ID

        Returns:
            用户表单数据
        """
        user = await self.repo.get_by_id(db, user_id)
        if not user:
            return None

        role_ids = await self.repo.get_user_role_ids(db, user_id)

        return {
            "id": user.id,
            "username": user.username,
            "nickname": user.nickname,
            "gender": user.gender,
            "deptId": user.dept_id,
            "mobile": user.mobile,
            "email": user.email,
            "status": user.status,
            "avatar": user.avatar,
            "userType": user.user_type,
            "roleIds": role_ids,
        }

    async def create_user_with_roles(
        self,
        db: AsyncSession,
        data: dict[str, Any],
    ) -> SysUser:
        """
        创建新用户并关联角色

        Args:
            db: 异步数据库会话
            data: 用户数据

        Returns:
            创建的用户对象

        Raises:
            BusinessException: 用户名为空或用户名已存在
        """
        username = data.get("username")
        nickname = data.get("nickname", username)
        gender = data.get("gender")
        dept_id = data.get("deptId")
        mobile = data.get("mobile")
        email = data.get("email")
        status = data.get("status", 1)
        user_type = data.get("userType") or "personal"
        role_ids = data.get("roleIds", [])

        if not username:
            raise BusinessException("用户名不能为空")

        existing_user = await self.repo.get_by_username_include_deleted(db, username)
        if existing_user:
            raise BusinessException(ResultCode.DATA_EXISTS, "该用户名不可用")

        plain_password = settings.DEFAULT_PASSWORD
        hashed_password = await hash_password_async(plain_password)

        user = SysUser(
            username=username,
            nickname=nickname,
            gender=gender,
            dept_id=dept_id,
            mobile=mobile,
            email=email,
            password=hashed_password,
            status=status,
            user_type=user_type,
        )

        return await self.repo.create_user(db, user, role_ids)

    async def update_user_with_roles(
        self,
        db: AsyncSession,
        user_id: int,
        data: dict[str, Any],
    ) -> None:
        """
        更新用户信息并关联角色

        Args:
            db: 异步数据库会话
            user_id: 用户ID
            data: 用户数据

        Raises:
            BusinessException: 用户不存在或用户名已存在
        """
        user = await self.repo.get_by_id(db, user_id)
        if not user:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "用户不存在")

        username = data.get("username")
        nickname = data.get("nickname")
        gender = data.get("gender")
        dept_id = data.get("deptId")
        mobile = data.get("mobile")
        email = data.get("email")
        user_type = data.get("userType")
        role_ids = data.get("roleIds", [])
        status = data.get("status")

        # 用户名字段只读，不可修改（与角色编码创建后不可修改保持一致）
        if username is not None and username != user.username:
            raise BusinessException(ResultCode.OPERATION_NOT_ALLOW, "用户名不可修改")

        if nickname is not None:
            user.nickname = nickname
        if gender is not None:
            user.gender = gender
        if dept_id is not None:
            user.dept_id = dept_id
        if mobile is not None:
            user.mobile = mobile
        if email is not None:
            user.email = email
        if user_type is not None:
            user.user_type = user_type
        if status is not None:
            user.status = status

        await db.flush()
        await self.repo.replace_user_roles(db, user_id, role_ids)

    async def _kick_user_sessions(self, db: AsyncSession, redis: Redis, user_id: int) -> None:
        """踢出目标用户全部在线会话并清理其角色权限缓存（禁用/删除/重置密码联动）。

        局部导入避免与 auth_service 的模块级循环依赖。
        """
        from app.service.auth_service import auth_service

        role_codes = await self.repo.get_user_role_codes(db, user_id)
        await auth_service.kick_user_sessions(redis, user_id, role_codes)

    async def update_user_status(
        self,
        db: AsyncSession,
        redis: Redis,
        user_id: int,
        status: int,
        current_user: UserContext | None = None,
    ) -> None:
        """
        更新用户状态

        Args:
            db: 异步数据库会话
            redis: Redis 客户端（禁用时踢出目标用户在线会话）
            user_id: 用户ID
            status: 状态（1-正常，0-禁用）
            current_user: 当前登录用户（用于自禁保护校验）

        Raises:
            BusinessException: 用户不存在、禁用超级管理员或禁用自己
        """
        user = await self.repo.get_by_id(db, user_id)
        if not user:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "用户不存在")

        # 超级管理员不可禁用（防自锁），启用不受限
        if user.username == "root" and status == 0:
            raise BusinessException(ResultCode.ROOT_USER_PROTECTED, "超级管理员不可禁用")

        # 任何用户不可禁用自己（文档 T-UM-042）
        if current_user is not None and current_user.id == user_id:
            raise BusinessException(ResultCode.OPERATION_NOT_ALLOW, "不可禁用自己")

        user.status = status

        # 禁用后实时踢出该用户全部在线会话（文档 §3.6.3）
        if status == 0:
            await self._kick_user_sessions(db, redis, user_id)

    async def update_password(
        self,
        db: AsyncSession,
        redis: Redis,
        user_id: int,
        new_password: str,
    ) -> None:
        """
        重置用户密码（管理员操作）

        Args:
            db: 异步数据库会话
            redis: Redis 客户端（重置后踢出目标用户在线会话）
            user_id: 用户ID
            new_password: 新密码

        Raises:
            BusinessException: 用户不存在或密码复杂度不符合要求
        """
        is_valid, error_msg = validate_password_complexity(new_password)
        if not is_valid:
            raise BusinessException(ResultCode.PARAM_ERROR, error_msg)

        user = await self.repo.get_by_id(db, user_id)
        if not user:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "用户不存在")

        hashed_password = await hash_password_async(new_password)
        user.password = hashed_password

        self.audit_repo.create_audit_async(
            operator_id=get_current_user_id(),
            target_type="user",
            target_id=user_id,
            action="password_change",
            module="user",
        )

        # 重置后踢出该用户全部在线会话，强制重新登录（文档 §3.5.3）
        await self._kick_user_sessions(db, redis, user_id)

    async def delete_users(
        self,
        db: AsyncSession,
        redis: Redis,
        ids: str,
        current_user: UserContext | None = None,
    ) -> dict[str, int]:
        """
        删除用户（逻辑删除，支持批量）

        Args:
            db: 异步数据库会话
            redis: Redis 客户端（删除后踢出目标用户在线会话）
            ids: 用户ID，多个以英文逗号分隔
            current_user: 当前登录用户（用于自删保护校验）

        Returns:
            删除统计 {"deleted_count": int}

        Raises:
            BusinessException: 未指定要删除的用户、不可删除自己或超级管理员不可删除
        """
        user_ids = [int(id_str.strip()) for id_str in ids.split(",") if id_str.strip()]

        if not user_ids:
            raise BusinessException("未指定要删除的用户")

        # 不可删除自己
        if current_user is not None and current_user.id in user_ids:
            raise BusinessException(ResultCode.OPERATION_NOT_ALLOW, "不可删除自己")

        # 超级管理员受保护，不可删除
        protected_ids = await self.repo.get_protected_user_ids(db, user_ids)
        if protected_ids:
            raise BusinessException(ResultCode.ROOT_USER_PROTECTED, "超级管理员不可删除")

        await self.repo.soft_delete_by_ids(db, user_ids)

        self.audit_repo.create_audit_async(
            operator_id=get_current_user_id(),
            target_type="user",
            target_id=ids,
            action="delete",
            module="user",
        )

        # 删除后踢出目标用户全部在线会话（文档 §3.4.3）
        for user_id in user_ids:
            await self._kick_user_sessions(db, redis, user_id)

        return {"deleted_count": len(user_ids)}


user_service = UserService()
