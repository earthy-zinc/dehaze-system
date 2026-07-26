"""
用户导入导出处理器单元测试
"""
from __future__ import annotations

import io
from unittest.mock import AsyncMock, patch

import pytest
from openpyxl import load_workbook

from app.service.import_export.handlers.user_export import UserExportHandler
from app.service.import_export.handlers.user_import import UserImportHandler
from app.service.import_export.models import ExportContext, ImportOptions


class TestUserExportHandler:
    def test_get_module(self):
        assert UserExportHandler().get_module() == "user"

    def test_get_field_configs(self):
        handler = UserExportHandler()
        fields = handler.get_field_configs()
        assert len(fields) == 10
        assert fields[0].field == "id"
        assert fields[1].field == "username"
        assert fields[2].field == "nickname"
        assert fields[3].field == "email"
        # 按 order 排序
        orders = [f.order for f in fields]
        assert orders == sorted(orders)
        # create_time 配置日期格式
        create_time_field = next(f for f in fields if f.field == "create_time")
        assert create_time_field.date_format == "%Y-%m-%d %H:%M:%S"

    @pytest.mark.asyncio
    async def test_estimate_count_without_dept_id(self):
        handler = UserExportHandler()
        with patch(
            "app.service.import_export.handlers.user_export.user_repository"
        ) as user_repo, patch(
            "app.service.import_export.handlers.user_export.dept_repository"
        ) as dept_repo:
            user_repo.get_user_list = AsyncMock(return_value=([], 42))
            count = await handler.estimate_count(None, {"keywords": "张"})
            assert count == 42
            dept_repo.get_children_ids.assert_not_called()
            user_repo.get_user_list.assert_awaited_once()
            call_kwargs = user_repo.get_user_list.call_args.kwargs
            assert call_kwargs["keywords"] == "张"
            assert call_kwargs["page"] == 1
            assert call_kwargs["page_size"] == 1

    @pytest.mark.asyncio
    async def test_estimate_count_with_dept_id(self):
        handler = UserExportHandler()
        with patch(
            "app.service.import_export.handlers.user_export.user_repository"
        ) as user_repo, patch(
            "app.service.import_export.handlers.user_export.dept_repository"
        ) as dept_repo:
            dept_repo.get_children_ids = AsyncMock(return_value=[1, 2, 3])
            user_repo.get_user_list = AsyncMock(return_value=([], 100))
            count = await handler.estimate_count(None, {"deptId": 1})
            assert count == 100
            dept_repo.get_children_ids.assert_awaited_once()
            call_kwargs = user_repo.get_user_list.call_args.kwargs
            assert call_kwargs["dept_ids"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_export_writes_excel_with_rows(self):
        handler = UserExportHandler()
        users = [
            {
                "id": 1,
                "username": "u1",
                "nickname": "n1",
                "email": "u1@x.com",
                "mobile": "13800138000",
                "gender": 1,
                "status": 1,
                "deptName": "研发部",
                "roleNames": "管理员",
                "create_time": None,
            }
        ]
        with patch(
            "app.service.import_export.handlers.user_export.user_repository"
        ) as user_repo, patch(
            "app.service.import_export.handlers.user_export.dept_repository"
        ):
            user_repo.get_user_list = AsyncMock(side_effect=[(users, 1), ([], 0)])
            output = io.BytesIO()
            ctx = ExportContext(
                task_id="t1",
                module="user",
                format="excel",
                query_params={},
                total_count=1,
            )
            await handler.export(
                None, ctx, output,
                AsyncMock(), AsyncMock(return_value=False),
            )
            output.seek(0)
            wb = load_workbook(output)
            ws = wb.active
            rows = list(ws.iter_rows(values_only=True))
            # 表头 10 列
            assert len(rows[0]) == 10
            assert rows[1][1] == "u1"  # username
            assert rows[1][2] == "n1"  # nickname
            assert rows[1][6] == "正常"  # status_label
            assert rows[1][7] == "研发部"  # dept_name

    @pytest.mark.asyncio
    async def test_export_csv_format(self):
        handler = UserExportHandler()
        with patch(
            "app.service.import_export.handlers.user_export.user_repository"
        ) as user_repo, patch(
            "app.service.import_export.handlers.user_export.dept_repository"
        ):
            user_repo.get_user_list = AsyncMock(side_effect=[([], 0), ([], 0)])
            output = io.BytesIO()
            ctx = ExportContext(
                task_id="t1", module="user", format="csv", query_params={}, total_count=0
            )
            await handler.export(
                None, ctx, output,
                AsyncMock(), AsyncMock(return_value=False),
            )
            content = output.getvalue()
            assert content.startswith("\ufeff".encode("utf-8"))

    @pytest.mark.asyncio
    async def test_export_stops_on_cancel(self):
        handler = UserExportHandler()
        with patch(
            "app.service.import_export.handlers.user_export.user_repository"
        ) as user_repo, patch(
            "app.service.import_export.handlers.user_export.dept_repository"
        ):
            users = [{"id": 1, "username": "u1", "nickname": "n1", "gender": 1, "status": 1}]
            user_repo.get_user_list = AsyncMock(return_value=(users, 1))
            cancel_cb = AsyncMock(return_value=True)
            output = io.BytesIO()
            ctx = ExportContext(
                task_id="t1", module="user", format="excel", query_params={}, total_count=1
            )
            await handler.export(None, ctx, output, AsyncMock(), cancel_cb)
            # 取消后只调用一次 get_user_list
            assert user_repo.get_user_list.await_count == 1


class TestUserImportHandler:
    def test_get_module(self):
        assert UserImportHandler().get_module() == "user"

    def test_get_field_configs(self):
        handler = UserImportHandler()
        fields = handler.get_field_configs()
        assert len(fields) == 8
        username_field = next(f for f in fields if f.field == "username")
        assert username_field.required is True
        assert username_field.max_length == 64
        nickname_field = next(f for f in fields if f.field == "nickname")
        assert nickname_field.required is True

    def test_get_template_sample_data(self):
        handler = UserImportHandler()
        samples = handler.get_template_sample_data()
        assert len(samples) == 1
        sample = samples[0]
        assert sample["username"] == "zhangsan"
        assert sample["nickname"] == "张三"
        assert sample["mobile"] == "13800138000"

    @pytest.mark.asyncio
    async def test_import_batch_success(self):
        handler = UserImportHandler()
        rows = [
            {"username": "u1", "nickname": "n1", "gender": "男", "mobile": "13800138000"},
        ]
        with patch(
            "app.service.import_export.handlers.user_import.user_repository"
        ) as user_repo, patch(
            "app.service.import_export.handlers.user_import.hash_password_async",
            new=AsyncMock(return_value="hashed-pwd"),
        ):
            user_repo.get_existing_usernames = AsyncMock(return_value=set())
            user_repo.create_user = AsyncMock()

            result = await handler.import_batch(
                None, rows, ImportOptions(mode="all"),
                AsyncMock(), AsyncMock(return_value=False),
            )
            assert result.total_rows == 1
            assert result.success_count == 1
            assert result.failure_count == 0
            user_repo.create_user.assert_awaited_once()
            created_user = user_repo.create_user.call_args.args[1]
            assert created_user.username == "u1"
            assert created_user.nickname == "n1"
            assert created_user.password == "hashed-pwd"

    @pytest.mark.asyncio
    async def test_import_batch_duplicate_username_skipped(self):
        handler = UserImportHandler()
        rows = [
            {"username": "dup", "nickname": "n1", "mobile": "13800138000"},
            {"username": "new", "nickname": "n2", "mobile": "13900139000"},
        ]
        with patch(
            "app.service.import_export.handlers.user_import.user_repository"
        ) as user_repo, patch(
            "app.service.import_export.handlers.user_import.hash_password_async",
            new=AsyncMock(return_value="hashed-pwd"),
        ):
            user_repo.get_existing_usernames = AsyncMock(return_value={"dup"})
            user_repo.create_user = AsyncMock()

            result = await handler.import_batch(
                None, rows, ImportOptions(mode="partial"),
                AsyncMock(), AsyncMock(return_value=False),
            )
            assert result.total_rows == 2
            assert result.success_count == 1
            assert result.failure_count == 1
            assert len(result.errors) == 1
            assert result.errors[0].row == 2
            user_repo.create_user.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_import_batch_blank_username_recorded_as_error(self):
        handler = UserImportHandler()
        rows = [{"username": "", "nickname": "n1"}]
        with patch(
            "app.service.import_export.handlers.user_import.user_repository"
        ) as user_repo, patch(
            "app.service.import_export.handlers.user_import.hash_password_async",
            new=AsyncMock(return_value="hashed-pwd"),
        ):
            user_repo.get_existing_usernames = AsyncMock(return_value=set())
            user_repo.create_user = AsyncMock()

            result = await handler.import_batch(
                None, rows, ImportOptions(mode="partial"),
                AsyncMock(), AsyncMock(return_value=False),
            )
            assert result.failure_count == 1
            assert "为空" in result.errors[0].message
            user_repo.create_user.assert_not_called()

    @pytest.mark.asyncio
    async def test_import_batch_duplicate_in_batch_skipped(self):
        handler = UserImportHandler()
        rows = [
            {"username": "dup", "nickname": "n1", "mobile": "13800138000"},
            {"username": "dup", "nickname": "n2", "mobile": "13900139000"},
        ]
        with patch(
            "app.service.import_export.handlers.user_import.user_repository"
        ) as user_repo, patch(
            "app.service.import_export.handlers.user_import.hash_password_async",
            new=AsyncMock(return_value="hashed-pwd"),
        ):
            user_repo.get_existing_usernames = AsyncMock(return_value=set())
            user_repo.create_user = AsyncMock()

            result = await handler.import_batch(
                None, rows, ImportOptions(mode="partial"),
                AsyncMock(), AsyncMock(return_value=False),
            )
            assert result.success_count == 1
            assert result.failure_count == 1
            assert result.errors[0].row == 3

    @pytest.mark.asyncio
    async def test_import_batch_default_dept_id_from_options(self):
        handler = UserImportHandler()
        rows = [{"username": "u1", "nickname": "n1", "mobile": "13800138000"}]
        with patch(
            "app.service.import_export.handlers.user_import.user_repository"
        ) as user_repo, patch(
            "app.service.import_export.handlers.user_import.hash_password_async",
            new=AsyncMock(return_value="hashed-pwd"),
        ):
            user_repo.get_existing_usernames = AsyncMock(return_value=set())
            user_repo.create_user = AsyncMock()

            await handler.import_batch(
                None, rows, ImportOptions(mode="all", extra={"deptId": 100}),
                AsyncMock(), AsyncMock(return_value=False),
            )
            created_user = user_repo.create_user.call_args.args[1]
            assert created_user.dept_id == 100

    @pytest.mark.asyncio
    async def test_import_batch_create_exception_recorded(self):
        handler = UserImportHandler()
        rows = [{"username": "u1", "nickname": "n1", "mobile": "13800138000"}]
        with patch(
            "app.service.import_export.handlers.user_import.user_repository"
        ) as user_repo, patch(
            "app.service.import_export.handlers.user_import.hash_password_async",
            new=AsyncMock(return_value="hashed-pwd"),
        ):
            user_repo.get_existing_usernames = AsyncMock(return_value=set())
            user_repo.create_user = AsyncMock(side_effect=RuntimeError("DB 错误"))

            result = await handler.import_batch(
                None, rows, ImportOptions(mode="partial"),
                AsyncMock(), AsyncMock(return_value=False),
            )
            assert result.failure_count == 1
            assert "DB 错误" in result.errors[0].message

    @pytest.mark.asyncio
    async def test_import_batch_gender_female(self):
        handler = UserImportHandler()
        rows = [{"username": "u1", "nickname": "n1", "gender": "女", "mobile": "13800138000"}]
        with patch(
            "app.service.import_export.handlers.user_import.user_repository"
        ) as user_repo, patch(
            "app.service.import_export.handlers.user_import.hash_password_async",
            new=AsyncMock(return_value="hashed-pwd"),
        ):
            user_repo.get_existing_usernames = AsyncMock(return_value=set())
            user_repo.create_user = AsyncMock()

            await handler.import_batch(
                None, rows, ImportOptions(mode="all"),
                AsyncMock(), AsyncMock(return_value=False),
            )
            created_user = user_repo.create_user.call_args.args[1]
            assert created_user.gender == 0

    @pytest.mark.asyncio
    async def test_import_batch_role_ids_parsed(self):
        handler = UserImportHandler()
        rows = [
            {
                "username": "u1",
                "nickname": "n1",
                "mobile": "13800138000",
                "role_ids": "1, 2, 3",
            }
        ]
        with patch(
            "app.service.import_export.handlers.user_import.user_repository"
        ) as user_repo, patch(
            "app.service.import_export.handlers.user_import.hash_password_async",
            new=AsyncMock(return_value="hashed-pwd"),
        ):
            user_repo.get_existing_usernames = AsyncMock(return_value=set())
            user_repo.create_user = AsyncMock()

            await handler.import_batch(
                None, rows, ImportOptions(mode="all"),
                AsyncMock(), AsyncMock(return_value=False),
            )
            role_ids = user_repo.create_user.call_args.args[2]
            assert role_ids == [1, 2, 3]
