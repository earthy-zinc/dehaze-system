"""用户导入处理器（模板字段与 Java 权威实现对齐：角色按编码关联，密码统一使用默认密码）"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.entity.sys_user import SysUser
from app.repository.role_repository import role_repository
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
            ImportFieldConfig(field="email", label="邮箱", max_length=128),
            ImportFieldConfig(field="mobile", label="手机号", max_length=20),
            ImportFieldConfig(field="gender", label="性别(男/女)"),
            ImportFieldConfig(field="roleCodes", label="角色编码(多个用英文逗号分隔)"),
        ]

    def get_template_sample_data(self) -> list[dict]:
        return [
            {
                "username": "zhangsan",
                "nickname": "张三",
                "email": "zhangsan@example.com",
                "mobile": "13800138000",
                "gender": "男",
                "roleCodes": "guest",
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

        role_codes: list[str] = []
        for row in rows:
            raw = str(row.get("roleCodes") or "").strip()
            role_codes.extend(code.strip() for code in raw.split(",") if code.strip())
        role_code_map = await role_repository.get_role_code_id_map(db, role_codes)

        hashed_password = await hash_password_async(settings.DEFAULT_PASSWORD)

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

                gender_str = str(row.get("gender") or "").strip()
                if gender_str and gender_str not in ("男", "女"):
                    errors.append(
                        ImportError(row=idx, field="gender", message="性别取值无效（应为 男/女）")
                    )
                    failure_count += 1
                    continue
                gender_value = 2 if gender_str == "女" else 1

                role_ids: list[int] = []
                role_ids_raw = str(row.get("roleCodes") or "").strip()
                if role_ids_raw:
                    for code in (c.strip() for c in role_ids_raw.split(",")):
                        if not code:
                            continue
                        role_id = role_code_map.get(code)
                        if role_id is None:
                            raise ValueError(f"角色编码不存在或已停用: {code}")
                        role_ids.append(role_id)

                user = SysUser(
                    username=username,
                    nickname=nickname,
                    password=hashed_password,
                    email=str(row.get("email") or "").strip() or None,
                    mobile=str(row.get("mobile") or "").strip() or None,
                    gender=gender_value,
                    dept_id=default_dept_id,
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
