"""
用户导入处理器
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.entity.sys_user import SysUser
from app.repository.user_repository import user_repository
from app.service.import_export.models import (
    ImportError,
    ImportFieldConfig,
    ImportOptions,
    ImportResult,
)
from app.service.import_export.registry import ImportHandler
from app.utils.password import hash_password_async


class UserImportHandler(ImportHandler):
    def get_module(self) -> str:
        return "user"

    def get_field_configs(self) -> list[ImportFieldConfig]:
        return [
            ImportFieldConfig(field="username", label="用户名", required=True, max_length=64),
            ImportFieldConfig(field="nickname", label="昵称", required=True, max_length=64),
            ImportFieldConfig(field="password", label="密码", max_length=64),
            ImportFieldConfig(field="email", label="邮箱", max_length=128),
            ImportFieldConfig(field="mobile", label="手机号", max_length=20),
            ImportFieldConfig(field="gender", label="性别(男/女)"),
            ImportFieldConfig(field="dept_id", label="部门ID"),
            ImportFieldConfig(field="role_ids", label="角色ID(多个用英文逗号分隔)"),
        ]

    def get_template_sample_data(self) -> list[dict]:
        return [
            {
                "username": "zhangsan",
                "nickname": "张三",
                "password": "",
                "email": "zhangsan@example.com",
                "mobile": "13800138000",
                "gender": "男",
                "dept_id": "1",
                "role_ids": "1",
            }
        ]

    async def import_batch(
        self,
        db: AsyncSession,
        rows: list[dict],
        options: ImportOptions,
        progress_cb,
        cancel_cb,
    ) -> ImportResult:
        errors: list[ImportError] = []
        success_count = 0
        failure_count = 0
        all_usernames = [str(r.get("username", "")).strip() for r in rows if r.get("username")]
        existing_usernames = await user_repository.get_existing_usernames(db, all_usernames)
        seen: set[str] = set()
        default_dept_id = options.extra.get("deptId")

        for idx, row in enumerate(rows, start=2):
            try:
                username = str(row.get("username") or "").strip()
                nickname = str(row.get("nickname") or "").strip()
                if not username or not nickname:
                    errors.append(ImportError(row=idx, message="用户名或昵称为空"))
                    failure_count += 1
                    continue
                if username in existing_usernames or username in seen:
                    errors.append(ImportError(row=idx, field="username", message="用户名已存在"))
                    failure_count += 1
                    continue
                seen.add(username)

                password = str(row.get("password") or "").strip() or settings.DEFAULT_PASSWORD
                hashed = await hash_password_async(password)

                gender_str = str(row.get("gender") or "").strip()
                if gender_str and gender_str not in ("男", "女"):
                    errors.append(
                        ImportError(row=idx, field="gender", message="性别取值无效（应为 男/女）")
                    )
                    failure_count += 1
                    continue
                gender_value = 2 if gender_str == "女" else 1

                dept_id_raw = row.get("dept_id")
                dept_id = (
                    int(dept_id_raw)
                    if dept_id_raw not in (None, "", "None")
                    else (default_dept_id or None)
                )

                role_ids: list[int] = []
                role_ids_raw = row.get("role_ids")
                if role_ids_raw:
                    role_ids = [
                        int(rid.strip()) for rid in str(role_ids_raw).split(",") if rid.strip()
                    ]

                user = SysUser(
                    username=username,
                    nickname=nickname,
                    password=hashed,
                    email=str(row.get("email") or "").strip() or None,
                    mobile=str(row.get("mobile") or "").strip() or None,
                    gender=gender_value,
                    dept_id=dept_id,
                    status=1,
                )
                await user_repository.create_user(db, user, role_ids)
                success_count += 1
            except Exception as e:
                errors.append(ImportError(row=idx, message=str(e)))
                failure_count += 1
            if idx % 100 == 0:
                await progress_cb(idx, len(rows))
                if await cancel_cb():
                    break

        return ImportResult(
            total_rows=len(rows),
            success_count=success_count,
            failure_count=failure_count,
            skipped_count=0,
            errors=errors,
        )
