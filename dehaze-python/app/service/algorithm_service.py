"""
算法服务

提供算法 CRUD 功能
"""

import os
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import BusinessException
from app.models.entity.sys_algorithm import SysAlgorithm
from app.repository.algorithm_repository import algorithm_repository
from app.utils.datetime_utils import format_time
from app.utils.file import get_file_size


class AlgorithmService:
    """算法服务（异步版本）"""

    @staticmethod
    def _build_algorithm_tree(algorithms: list[SysAlgorithm]) -> list[dict[str, Any]]:
        """构建算法树形结构"""
        algorithm_dict = {
            algorithm.id: {
                "id": algorithm.id,
                "parentId": algorithm.parent_id,
                "type": algorithm.type,
                "name": algorithm.name,
                "path": algorithm.path,
                "size": algorithm.size,
                "img": algorithm.img,
                "params": algorithm.params,
                "flops": algorithm.flops,
                "importPath": algorithm.import_path,
                "description": algorithm.description,
                "status": algorithm.status,
                "createTime": format_time(algorithm.create_time),
                "updateTime": format_time(algorithm.update_time),
                "children": [],
            }
            for algorithm in algorithms
        }

        root_algorithms = []
        for algorithm in algorithm_dict.values():
            if algorithm["parentId"] == 0:
                root_algorithms.append(algorithm)
            else:
                parent = algorithm_dict.get(algorithm["parentId"])
                if parent:
                    parent["children"].append(algorithm)

        return root_algorithms

    @staticmethod
    async def get_algorithm_list(db: AsyncSession, keywords: str | None = None) -> list[dict[str, Any]]:
        """
        获取算法树形表格

        Args:
            db: 异步数据库会话
            keywords: 搜索关键词

        Returns:
            算法列表（树形结构）
        """
        algorithms = await algorithm_repository.get_list_with_keywords(db, keywords)
        return AlgorithmService._build_algorithm_tree(algorithms)

    @staticmethod
    async def get_algorithm_options(db: AsyncSession) -> list[dict[str, Any]]:
        """获取模型下拉选项列表"""
        return await algorithm_repository.get_algorithm_options(db)

    @staticmethod
    async def get_algorithm_by_id(db: AsyncSession, algorithm_id: int) -> dict[str, Any] | None:
        """根据 ID 获取算法信息"""
        algorithm = await algorithm_repository.get_by_id(db, algorithm_id)

        if not algorithm:
            return None

        return {
            "id": algorithm.id,
            "parentId": algorithm.parent_id,
            "type": algorithm.type,
            "name": algorithm.name,
            "path": algorithm.path,
            "size": algorithm.size,
            "img": algorithm.img,
            "params": algorithm.params,
            "flops": algorithm.flops,
            "importPath": algorithm.import_path,
            "description": algorithm.description,
            "status": algorithm.status,
            "createTime": format_time(algorithm.create_time),
            "updateTime": format_time(algorithm.update_time),
        }

    @staticmethod
    async def create_algorithm(db: AsyncSession, data: dict[str, Any]) -> int:
        """
        新增算法

        Args:
            db: 异步数据库会话
            data: 算法数据

        Returns:
            创建的算法ID
        """
        algorithm = SysAlgorithm(
            parent_id=data.get("parentId", 0),
            type=data.get("type", ""),
            name=data.get("name", ""),
            path=data.get("path", ""),
            import_path=data.get("importPath", ""),
            description=data.get("description", ""),
            status=data.get("status", 1),
        )

        # 如果路径是有效文件，获取文件大小
        if "path" in data and os.path.isfile(data["path"]):
            algorithm.size = get_file_size(data["path"])

        created = await algorithm_repository.create(db, algorithm)
        return created.id

    @staticmethod
    async def update_algorithm(db: AsyncSession, algorithm_id: int, data: dict[str, Any]) -> None:
        """
        修改算法

        Args:
            db: 异步数据库会话
            algorithm_id: 算法ID
            data: 算法数据

        Raises:
            BusinessException: 算法不存在
        """
        algorithm = await algorithm_repository.get_by_id(db, algorithm_id)

        if not algorithm:
            raise BusinessException("算法不存在")

        update_data = {}
        if "parentId" in data:
            update_data["parent_id"] = data["parentId"]
        if "type" in data:
            update_data["type"] = data["type"]
        if "name" in data:
            update_data["name"] = data["name"]
        if "path" in data:
            update_data["path"] = data["path"]
            if os.path.isfile(data["path"]):
                update_data["size"] = get_file_size(data["path"])
        if "importPath" in data:
            update_data["import_path"] = data["importPath"]
        if "description" in data:
            update_data["description"] = data["description"]
        if "status" in data:
            update_data["status"] = data["status"]

        await algorithm_repository.update(db, algorithm, update_data)

    @staticmethod
    async def delete_algorithms(db: AsyncSession, algorithm_ids: list[int]) -> int:
        """
        删除算法（包含子算法）

        Args:
            db: 异步数据库会话
            algorithm_ids: 算法ID列表

        Returns:
            删除的数量
        """
        ids_to_delete = await algorithm_repository.get_with_children_ids(db, algorithm_ids)
        if ids_to_delete:
            return await algorithm_repository.delete_by_ids(db, ids_to_delete)
        return 0
