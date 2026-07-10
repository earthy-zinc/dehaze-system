"""
字典服务测试

测试 DictService 和 DictTypeService 的核心功能
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.service.dict_service import DictService, DictTypeService
from app.core.exceptions import BusinessException


@pytest.mark.unit
class TestDictService:
    """字典服务测试"""

    @pytest.mark.asyncio
    async def test_get_dict_page(self):
        """测试获取字典分页列表"""
        mock_db = AsyncMock()

        with patch("app.service.dict_service.dict_repository") as mock_repo:
            mock_repo.get_page = AsyncMock(return_value=([], 0))

            result, total = await DictService.get_dict_page(
                db=mock_db,
                page=1,
                page_size=10,
            )

            assert result == []
            assert total == 0

    @pytest.mark.asyncio
    async def test_get_dict_form(self):
        """测试获取字典表单数据"""
        mock_db = AsyncMock()

        with patch("app.service.dict_service.dict_repository") as mock_repo:
            mock_repo.get_form_by_id = AsyncMock(return_value={"id": 1, "name": "测试"})

            result = await DictService.get_dict_form(mock_db, 1)

            assert result is not None
            assert result["name"] == "测试"

    @pytest.mark.asyncio
    async def test_create_dict_success(self):
        """测试创建字典项成功"""
        mock_db = AsyncMock()
        mock_dict_type = MagicMock()
        mock_dict_type.code = "status"

        with patch("app.service.dict_service.dict_repository") as mock_repo, \
             patch("app.service.dict_service.dict_type_repository") as mock_type_repo:
            mock_type_repo.get_by_code = AsyncMock(return_value=mock_dict_type)
            mock_repo.get_by_type_code_and_value = AsyncMock(return_value=None)
            mock_repo.create = AsyncMock(return_value=True)

            # Mock 缓存清除
            with patch.object(DictService, "_invalidate_options_cache", AsyncMock()):
                result = await DictService.create_dict(
                    db=mock_db,
                    data={"name": "测试", "value": "test", "typeCode": "status"},
                )

                assert result is True

    @pytest.mark.asyncio
    async def test_create_dict_type_not_exist(self):
        """测试创建字典项时类型不存在"""
        mock_db = AsyncMock()

        with patch("app.service.dict_service.dict_type_repository") as mock_type_repo:
            mock_type_repo.get_by_code = AsyncMock(return_value=None)

            with pytest.raises(BusinessException, match="字典类型不存在"):
                await DictService.create_dict(
                    db=mock_db,
                    data={"name": "测试", "value": "test", "typeCode": "nonexistent"},
                )

    @pytest.mark.asyncio
    async def test_create_dict_value_duplicate(self):
        """测试创建字典项时值重复"""
        mock_db = AsyncMock()
        mock_dict_type = MagicMock()
        mock_dict_type.code = "status"
        mock_existing = MagicMock()

        with patch("app.service.dict_service.dict_repository") as mock_repo, \
             patch("app.service.dict_service.dict_type_repository") as mock_type_repo:
            mock_type_repo.get_by_code = AsyncMock(return_value=mock_dict_type)
            mock_repo.get_by_type_code_and_value = AsyncMock(return_value=mock_existing)

            with pytest.raises(BusinessException, match="该类型下字典值已存在"):
                await DictService.create_dict(
                    db=mock_db,
                    data={"name": "测试", "value": "test", "typeCode": "status"},
                )

    @pytest.mark.asyncio
    async def test_update_dict_success(self):
        """测试更新字典项成功"""
        mock_db = AsyncMock()
        mock_old_dict = MagicMock()
        mock_old_dict.id = 1
        mock_old_dict.type_code = "status"
        mock_old_dict.value = "old_value"

        with patch("app.service.dict_service.dict_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_old_dict)
            mock_repo.get_by_type_code_and_value = AsyncMock(return_value=None)
            mock_repo.update_by_id = AsyncMock(return_value=True)

            # Mock 缓存清除
            with patch.object(DictService, "_invalidate_options_cache", AsyncMock()):
                result = await DictService.update_dict(
                    db=mock_db,
                    dict_id=1,
                    data={"name": "更新测试"},
                )

                assert result is True

    @pytest.mark.asyncio
    async def test_update_dict_not_found(self):
        """测试更新字典项时不存在"""
        mock_db = AsyncMock()

        with patch("app.service.dict_service.dict_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=None)

            with pytest.raises(BusinessException, match="字典不存在"):
                await DictService.update_dict(
                    db=mock_db,
                    dict_id=999,
                    data={"name": "更新测试"},
                )

    @pytest.mark.asyncio
    async def test_delete_dict_success(self):
        """测试删除字典项成功"""
        mock_db = AsyncMock()

        with patch("app.service.dict_service.dict_repository") as mock_repo:
            mock_repo.get_type_codes_by_ids = AsyncMock(return_value=["status"])
            mock_repo.delete_by_ids = AsyncMock(return_value=True)

            # Mock 缓存清除
            with patch.object(DictService, "_invalidate_options_cache", AsyncMock()):
                result = await DictService.delete_dict(
                    db=mock_db,
                    dict_ids=[1, 2, 3],
                )

                assert result is True

    @pytest.mark.asyncio
    async def test_list_dict_options(self):
        """测试获取字典下拉列表"""
        mock_db = AsyncMock()

        with patch("app.service.dict_service.dict_repository") as mock_repo:
            mock_repo.list_options_by_type = AsyncMock(return_value=[{"label": "启用", "value": "1"}])

            # Mock 缓存
            with patch.object(DictService, "_get_options_from_cache", AsyncMock(return_value=None)), \
                 patch.object(DictService, "_set_options_to_cache", AsyncMock()):
                result = await DictService.list_dict_options(mock_db, "status")

                assert len(result) == 1
                assert result[0]["label"] == "启用"


@pytest.mark.unit
class TestDictTypeService:
    """字典类型服务测试"""

    @pytest.mark.asyncio
    async def test_get_dict_type_page(self):
        """测试获取字典类型分页列表"""
        mock_db = AsyncMock()

        with patch("app.service.dict_service.dict_type_repository") as mock_repo:
            mock_repo.get_page = AsyncMock(return_value=([], 0))

            result, total = await DictTypeService.get_dict_type_page(
                db=mock_db,
                page=1,
                page_size=10,
            )

            assert result == []
            assert total == 0

    @pytest.mark.asyncio
    async def test_get_dict_type_form(self):
        """测试获取字典类型表单数据"""
        mock_db = AsyncMock()

        with patch("app.service.dict_service.dict_type_repository") as mock_repo:
            mock_repo.get_form_by_id = AsyncMock(return_value={"id": 1, "name": "状态", "code": "status"})

            result = await DictTypeService.get_dict_type_form(mock_db, 1)

            assert result is not None
            assert result["name"] == "状态"

    @pytest.mark.asyncio
    async def test_create_dict_type_success(self):
        """测试创建字典类型成功"""
        mock_db = AsyncMock()

        with patch("app.service.dict_service.dict_type_repository") as mock_repo:
            mock_repo.get_by_code = AsyncMock(return_value=None)
            mock_repo.create = AsyncMock(return_value=True)

            result = await DictTypeService.create_dict_type(
                db=mock_db,
                data={"name": "状态", "code": "status"},
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_create_dict_type_duplicate_code(self):
        """测试创建字典类型时编码重复"""
        mock_db = AsyncMock()
        mock_existing = MagicMock()

        with patch("app.service.dict_service.dict_type_repository") as mock_repo:
            mock_repo.get_by_code = AsyncMock(return_value=mock_existing)

            with pytest.raises(BusinessException, match="字典类型编码已存在"):
                await DictTypeService.create_dict_type(
                    db=mock_db,
                    data={"name": "状态", "code": "status"},
                )

    @pytest.mark.asyncio
    async def test_update_dict_type_success(self):
        """测试更新字典类型成功"""
        mock_db = AsyncMock()
        mock_old_type = MagicMock()
        mock_old_type.id = 1
        mock_old_type.code = "status"

        with patch("app.service.dict_service.dict_type_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_old_type)
            mock_repo.get_by_code = AsyncMock(return_value=None)
            mock_repo.update_by_id = AsyncMock(return_value=True)

            result = await DictTypeService.update_dict_type(
                db=mock_db,
                type_id=1,
                data={"name": "更新状态"},
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_update_dict_type_not_found(self):
        """测试更新字典类型时不存在"""
        mock_db = AsyncMock()

        with patch("app.service.dict_service.dict_type_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=None)

            with pytest.raises(BusinessException, match="字典类型不存在"):
                await DictTypeService.update_dict_type(
                    db=mock_db,
                    type_id=999,
                    data={"name": "更新状态"},
                )

    @pytest.mark.asyncio
    async def test_delete_dict_types_success(self):
        """测试删除字典类型成功"""
        mock_db = AsyncMock()
        mock_dict_type = MagicMock()
        mock_dict_type.id = 1
        mock_dict_type.code = "status"
        mock_dict_type.name = "状态"

        with patch("app.service.dict_service.dict_type_repository") as mock_type_repo, \
             patch("app.service.dict_service.dict_repository") as mock_dict_repo:
            mock_type_repo.get_by_id = AsyncMock(return_value=mock_dict_type)
            mock_dict_repo.count_by_type_code = AsyncMock(return_value=0)
            mock_type_repo.delete_by_ids = AsyncMock(return_value=True)

            result = await DictTypeService.delete_dict_types(
                db=mock_db,
                type_ids=[1, 2],
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_delete_dict_types_has_data(self):
        """测试删除字典类型时存在关联数据"""
        mock_db = AsyncMock()
        mock_dict_type = MagicMock()
        mock_dict_type.id = 1
        mock_dict_type.code = "status"
        mock_dict_type.name = "状态"

        with patch("app.service.dict_service.dict_type_repository") as mock_type_repo, \
             patch("app.service.dict_service.dict_repository") as mock_dict_repo:
            mock_type_repo.get_by_id = AsyncMock(return_value=mock_dict_type)
            mock_dict_repo.count_by_type_code = AsyncMock(return_value=5)

            with pytest.raises(BusinessException, match="存在.*关联数据"):
                await DictTypeService.delete_dict_types(
                    db=mock_db,
                    type_ids=[1],
                )
