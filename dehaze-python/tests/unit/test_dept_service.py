"""
部门服务测试

测试 DeptService 的核心功能
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.service.dept_service import DeptService
from app.core.exceptions import BusinessException


@pytest.mark.unit
class TestDeptService:
    """部门服务测试"""

    @pytest.mark.asyncio
    async def test_get_dept_list(self):
        """测试获取部门列表"""
        mock_db = AsyncMock()

        with patch("app.service.dept_service.dept_repository") as mock_repo:
            mock_repo.get_dept_list = AsyncMock(return_value=[])

            result = await DeptService.get_dept_list(mock_db)

            assert result == []

    @pytest.mark.asyncio
    async def test_get_dept_options(self):
        """测试获取部门选项"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.dept_service.dept_repository") as mock_repo:
            with patch("app.service.dept_service.CacheService") as mock_cache_class:
                mock_cache = AsyncMock()
                mock_cache.get_json = AsyncMock(return_value=None)
                mock_cache.set_json = AsyncMock()
                mock_cache_class.return_value = mock_cache

                mock_repo.get_dept_options_tree = AsyncMock(return_value=[])

                result = await DeptService.get_dept_options(mock_db, mock_redis)

                assert result == []

    @pytest.mark.asyncio
    async def test_get_dept_options_from_cache(self):
        """测试从缓存获取部门选项"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        cached_data = [{"value": 1, "label": "测试部门", "children": []}]

        with patch("app.service.dept_service.CacheService") as mock_cache_class:
            mock_cache = AsyncMock()
            mock_cache.get_json = AsyncMock(return_value=cached_data)
            mock_cache_class.return_value = mock_cache

            result = await DeptService.get_dept_options(mock_db, mock_redis)

            assert result == cached_data

    @pytest.mark.asyncio
    async def test_create_dept_success(self):
        """测试创建部门成功"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_dept = MagicMock()
        mock_dept.id = 2

        with patch("app.service.dept_service.dept_repository") as mock_repo:
            mock_repo.check_name_exists = AsyncMock(return_value=False)
            mock_repo.generate_tree_path = AsyncMock(return_value="0,1,2")

            mock_db.add = MagicMock()
            mock_db.commit = AsyncMock()
            mock_db.refresh = AsyncMock()

            with patch.object(DeptService, "_clear_cache", new_callable=AsyncMock):
                dept_id = await DeptService.create_dept(
                    db=mock_db,
                    redis=mock_redis,
                    data={"name": "测试部门", "parentId": 0},
                )

            mock_repo.check_name_exists.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_dept_duplicate_name(self):
        """测试创建部门时名称已存在"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.dept_service.dept_repository") as mock_repo:
            mock_repo.check_name_exists = AsyncMock(return_value=True)

            with pytest.raises(BusinessException, match="同一层级下部门名称已存在"):
                await DeptService.create_dept(
                    db=mock_db,
                    redis=mock_redis,
                    data={"name": "已存在部门", "parentId": 0},
                )

    @pytest.mark.asyncio
    async def test_create_dept_parent_not_found(self):
        """测试创建部门时上级部门不存在"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.dept_service.dept_repository") as mock_repo:
            mock_repo.check_name_exists = AsyncMock(return_value=False)
            mock_repo.get_by_id = AsyncMock(return_value=None)

            with pytest.raises(BusinessException, match="上级部门不存在"):
                await DeptService.create_dept(
                    db=mock_db,
                    redis=mock_redis,
                    data={"name": "测试部门", "parentId": 999},
                )

    @pytest.mark.asyncio
    async def test_update_dept_success(self):
        """测试更新部门成功"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_dept = MagicMock()
        mock_dept.id = 2
        mock_dept.name = "原部门"
        mock_dept.parent_id = 1

        with patch("app.service.dept_service.dept_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_dept)
            mock_repo.check_name_exists = AsyncMock(return_value=False)

            mock_db.commit = AsyncMock()

            with patch.object(DeptService, "_clear_cache", new_callable=AsyncMock):
                await DeptService.update_dept(
                    db=mock_db,
                    redis=mock_redis,
                    dept_id=2,
                    data={"name": "更新部门"},
                )

    @pytest.mark.asyncio
    async def test_update_dept_not_found(self):
        """测试更新部门时部门不存在"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.dept_service.dept_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=None)

            with pytest.raises(BusinessException, match="部门不存在"):
                await DeptService.update_dept(
                    db=mock_db,
                    redis=mock_redis,
                    dept_id=999,
                    data={"name": "测试"},
                )

    @pytest.mark.asyncio
    async def test_update_root_dept_parent_id(self):
        """测试修改根部门的上级部门"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_dept = MagicMock()
        mock_dept.id = 1
        mock_dept.name = "根部门"
        mock_dept.parent_id = 0

        with patch("app.service.dept_service.dept_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_dept)

            with pytest.raises(BusinessException, match="根部门不可修改上级部门"):
                await DeptService.update_dept(
                    db=mock_db,
                    redis=mock_redis,
                    dept_id=1,
                    data={"parentId": 2},
                )

    @pytest.mark.asyncio
    async def test_update_dept_circular_reference(self):
        """测试循环引用检测"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_dept = MagicMock()
        mock_dept.id = 2
        mock_dept.name = "测试部门"
        mock_dept.parent_id = 1

        with patch("app.service.dept_service.dept_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_dept)
            mock_repo.get_children_ids = AsyncMock(return_value=[2, 3, 4])

            with pytest.raises(BusinessException, match="不能将部门移动到子部门下"):
                await DeptService.update_dept(
                    db=mock_db,
                    redis=mock_redis,
                    dept_id=2,
                    data={"parentId": 3},
                )

    @pytest.mark.asyncio
    async def test_delete_depts_success(self):
        """测试删除部门成功"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_dept = MagicMock()
        mock_dept.id = 2
        mock_dept.name = "测试部门"

        with patch("app.service.dept_service.dept_repository") as mock_repo:
            with patch("app.service.dept_service.user_repository") as mock_user_repo:
                mock_repo.get_by_id = AsyncMock(return_value=mock_dept)
                mock_repo.count_children = AsyncMock(return_value=0)
                mock_repo.delete_depts = AsyncMock()
                mock_user_repo.count_users_by_dept = AsyncMock(return_value=0)

                mock_db.commit = AsyncMock()

                with patch.object(DeptService, "_clear_cache", new_callable=AsyncMock):
                    await DeptService.delete_depts(
                        db=mock_db,
                        redis=mock_redis,
                        dept_ids=[2],
                    )

    @pytest.mark.asyncio
    async def test_delete_depts_empty_ids(self):
        """测试删除部门时 ID 为空"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with pytest.raises(BusinessException, match="未指定要删除的部门"):
            await DeptService.delete_depts(
                db=mock_db,
                redis=mock_redis,
                dept_ids=[],
            )

    @pytest.mark.asyncio
    async def test_delete_root_dept(self):
        """测试删除根部门"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with pytest.raises(BusinessException, match="根部门不可删除"):
            await DeptService.delete_depts(
                db=mock_db,
                redis=mock_redis,
                dept_ids=[1],
            )

    @pytest.mark.asyncio
    async def test_delete_dept_with_users(self):
        """测试删除有用户的部门"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_dept = MagicMock()
        mock_dept.id = 2
        mock_dept.name = "测试部门"

        with patch("app.service.dept_service.dept_repository") as mock_repo:
            with patch("app.service.dept_service.user_repository") as mock_user_repo:
                mock_repo.get_by_id = AsyncMock(return_value=mock_dept)
                mock_user_repo.count_users_by_dept = AsyncMock(return_value=5)

                with pytest.raises(BusinessException, match="下存在 5 个用户"):
                    await DeptService.delete_depts(
                        db=mock_db,
                        redis=mock_redis,
                        dept_ids=[2],
                    )

    @pytest.mark.asyncio
    async def test_delete_dept_with_children(self):
        """测试删除有子部门的部门"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_dept = MagicMock()
        mock_dept.id = 2
        mock_dept.name = "测试部门"

        with patch("app.service.dept_service.dept_repository") as mock_repo:
            with patch("app.service.dept_service.user_repository") as mock_user_repo:
                mock_repo.get_by_id = AsyncMock(return_value=mock_dept)
                mock_repo.count_children = AsyncMock(return_value=3)
                mock_user_repo.count_users_by_dept = AsyncMock(return_value=0)

                with pytest.raises(BusinessException, match="下存在子部门"):
                    await DeptService.delete_depts(
                        db=mock_db,
                        redis=mock_redis,
                        dept_ids=[2],
                    )


@pytest.mark.unit
class TestDeptTreeBuilding:
    """部门树构建测试"""

    def test_build_empty_tree(self):
        """测试构建空树"""
        result = DeptService._build_dept_tree([])
        assert result == []

    def test_build_single_level_tree(self):
        """测试构建单层树"""
        # 创建真实的 mock 对象，而非 MagicMock
        class MockDept:
            def __init__(self, id, name, parent_id, tree_path, sort, status, deleted, create_time, update_time):
                self.id = id
                self.name = name
                self.parent_id = parent_id
                self.tree_path = tree_path
                self.sort = sort
                self.status = status
                self.deleted = deleted
                self.create_time = create_time
                self.update_time = update_time

        mock_depts = [
            MockDept(1, "根部门", 0, "0,1", 1, 1, 0, None, None),
        ]

        result = DeptService._build_dept_tree(mock_depts)

        assert len(result) == 1
        assert result[0]["name"] == "根部门"

    def test_build_multi_level_tree(self):
        """测试构建多层树"""
        class MockDept:
            def __init__(self, id, name, parent_id, tree_path, sort, status, deleted, create_time, update_time):
                self.id = id
                self.name = name
                self.parent_id = parent_id
                self.tree_path = tree_path
                self.sort = sort
                self.status = status
                self.deleted = deleted
                self.create_time = create_time
                self.update_time = update_time

        mock_depts = [
            MockDept(1, "根部门", 0, "0,1", 1, 1, 0, None, None),
            MockDept(2, "子部门1", 1, "0,1,2", 1, 1, 0, None, None),
            MockDept(3, "子部门2", 1, "0,1,3", 2, 1, 0, None, None),
        ]

        result = DeptService._build_dept_tree(mock_depts)

        assert len(result) == 1
        assert result[0]["name"] == "根部门"
        assert len(result[0]["children"]) == 2
