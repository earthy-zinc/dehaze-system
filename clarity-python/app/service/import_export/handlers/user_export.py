"""
用户导出处理器
"""

from __future__ import annotations

import io

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import BATCH_SIZE
from app.repository.dept_repository import dept_repository
from app.repository.user_repository import user_repository
from app.service.import_export.file_generator import write_csv, write_excel
from app.service.import_export.models import ExportContext, ExportFieldConfig
from app.service.import_export.registry import ExportHandler


class UserExportHandler(ExportHandler):
    def get_module(self) -> str:
        return "user"

    async def estimate_count(self, db: AsyncSession, query_params: dict) -> int:
        dept_id = query_params.get("deptId")
        dept_ids = await dept_repository.get_children_ids(db, dept_id) if dept_id else None
        _, total = await user_repository.get_user_list(
            db,
            page=1,
            page_size=1,
            keywords=query_params.get("keywords"),
            status=query_params.get("status"),
            dept_ids=dept_ids,
            create_time_start=query_params.get("startTime"),
            create_time_end=query_params.get("endTime"),
        )
        return int(total)

    async def export(
        self,
        db: AsyncSession,
        ctx: ExportContext,
        output: io.BytesIO,
        progress_cb,
        cancel_cb,
    ) -> None:
        params = ctx.query_params
        dept_id = params.get("deptId")
        dept_ids = await dept_repository.get_children_ids(db, dept_id) if dept_id else None
        total = ctx.total_count or await self.estimate_count(db, params)

        page = 1
        page_size = BATCH_SIZE
        all_rows: list[dict] = []
        while True:
            users, _ = await user_repository.get_user_list(
                db,
                page=page,
                page_size=page_size,
                keywords=params.get("keywords"),
                status=params.get("status"),
                dept_ids=dept_ids,
                create_time_start=params.get("startTime"),
                create_time_end=params.get("endTime"),
            )
            if not users:
                break
            all_rows.extend(_user_to_row(u) for u in users)
            processed = page * page_size
            await progress_cb(min(processed, total), total)
            if await cancel_cb():
                break
            page += 1

        fields = self.filter_fields(ctx.selected_fields)
        if ctx.format == "csv":
            write_csv(fields, all_rows, output)
        else:
            write_excel(fields, all_rows, output)

    def get_field_configs(self) -> list[ExportFieldConfig]:
        return [
            ExportFieldConfig(field="id", label="ID", order=1),
            ExportFieldConfig(field="username", label="用户名", order=2),
            ExportFieldConfig(field="nickname", label="昵称", order=3),
            ExportFieldConfig(field="email", label="邮箱", order=4),
            ExportFieldConfig(field="mobile", label="手机号", order=5),
            ExportFieldConfig(field="gender_label", label="性别", order=6),
            ExportFieldConfig(field="status_label", label="状态", order=7),
            ExportFieldConfig(field="dept_name", label="部门", order=8),
            ExportFieldConfig(field="role_names", label="角色", order=9),
            ExportFieldConfig(
                field="create_time", label="创建时间", order=10, date_format="%Y-%m-%d %H:%M:%S"
            ),
        ]


def _user_to_row(u: dict) -> dict:
    gender = int(u.get("gender") or 0)
    status = int(u.get("status") or 0)
    return {
        "id": u.get("id"),
        "username": u.get("username") or "",
        "nickname": u.get("nickname") or "",
        "email": u.get("email") or "",
        "mobile": u.get("mobile") or "",
        "gender_label": {1: "男", 2: "女"}.get(gender, "未知"),
        "status_label": "正常" if status == 1 else "禁用",
        "dept_name": u.get("deptName") or "",
        "role_names": u.get("roleNames") or "",
        "create_time": u.get("create_time"),
    }
