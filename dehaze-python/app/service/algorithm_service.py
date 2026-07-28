"""
算法服务

提供算法 CRUD、状态机、审核、版本控制、导入/导出、监控功能
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import BusinessException
from app.models.entity.sys_algorithm import SysAlgorithm, SysAlgorithmVersion
from app.repository.algorithm_repository import (
    algorithm_repository,
    AlgorithmStatus,
)
from app.utils.datetime_utils import format_time
from app.utils.file import convert_size
from algorithm.model_loader import check_model_exists


class AlgorithmService:
    """算法服务（异步版本）"""

    @staticmethod
    def _to_vo(algorithm: SysAlgorithm) -> dict[str, Any]:
        """算法实体转 VO（统一字段映射，消除重复）"""
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
            "version": algorithm.version,
            "auditBy": algorithm.audit_by,
            "auditTime": format_time(algorithm.audit_time) if algorithm.audit_time else None,
            "auditRemark": algorithm.audit_remark,
            "createTime": format_time(algorithm.create_time),
            "updateTime": format_time(algorithm.update_time),
        }

    @staticmethod
    def _build_algorithm_tree(algorithms: list[SysAlgorithm]) -> list[dict[str, Any]]:
        """构建算法树形结构"""
        algorithm_dict = {
            algorithm.id: {**AlgorithmService._to_vo(algorithm), "children": []}
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
        """获取算法树形表格"""
        algorithms = await algorithm_repository.get_list_with_keywords(db, keywords)
        return AlgorithmService._build_algorithm_tree(algorithms)

    @staticmethod
    async def get_algorithm_options(db: AsyncSession) -> list[dict[str, Any]]:
        """获取模型下拉选项列表"""
        return await algorithm_repository.get_algorithm_options(db)

    @staticmethod
    async def list_all_algorithms(db: AsyncSession) -> list[dict[str, Any]]:
        """获取所有算法扁平列表（不构建树形结构），用于前端下拉选择"""
        algorithms = await algorithm_repository.get_list_with_keywords(db, None)
        return [AlgorithmService._to_vo(algo) for algo in algorithms]

    @staticmethod
    async def get_algorithm_by_id(db: AsyncSession, algorithm_id: int) -> dict[str, Any]:
        """根据 ID 获取算法信息"""
        algorithm = await algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException("当前算法不存在")
        return AlgorithmService._to_vo(algorithm)

    @staticmethod
    async def create_algorithm(db: AsyncSession, data: dict[str, Any]) -> int:
        """新增算法"""
        algorithm = SysAlgorithm(
            parent_id=data.get("parentId", 0),
            type=data.get("type", ""),
            name=data.get("name", ""),
            path=data.get("path", ""),
            import_path=data.get("importPath", ""),
            description=data.get("description", ""),
            status=data.get("status", AlgorithmStatus.DRAFT),
            version=data.get("version"),
        )

        # path 指向具体文件时，通过 Nginx 静态服务校验可访问性并回填 size
        path_value = data.get("path", "")
        if path_value:
            size_bytes = await asyncio.to_thread(check_model_exists, path_value)
            if size_bytes is not None:
                algorithm.size = convert_size(size_bytes)

        created = await algorithm_repository.create(db, algorithm)
        return created.id

    @staticmethod
    async def update_algorithm(db: AsyncSession, algorithm_id: int, data: dict[str, Any]) -> None:
        """修改算法"""
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
            path_value = data["path"]
            if path_value:
                size_bytes = await asyncio.to_thread(check_model_exists, path_value)
                if size_bytes is not None:
                    update_data["size"] = convert_size(size_bytes)
        if "importPath" in data:
            update_data["import_path"] = data["importPath"]
        if "description" in data:
            update_data["description"] = data["description"]
        if "version" in data:
            update_data["version"] = data["version"]
        if "status" in data:
            update_data["status"] = data["status"]

        await algorithm_repository.update(db, algorithm, update_data)

    @staticmethod
    async def delete_algorithm_single(db: AsyncSession, algorithm_id: int) -> int:
        """删除单个算法（含子算法）"""
        return await AlgorithmService.delete_algorithms(db, [algorithm_id])

    @staticmethod
    async def delete_algorithms(db: AsyncSession, algorithm_ids: list[int]) -> int:
        """批量删除算法（包含子算法），对齐 Java deleteAlgorithms

        Java: removeByIds 返回 false 时 Result.judge(false) → Result.failed()
        Python: 无匹配记录时抛出 BusinessException（等价于 Result.failed）
        """
        ids_to_delete = await algorithm_repository.get_with_children_ids(db, algorithm_ids)
        if not ids_to_delete:
            raise BusinessException("删除失败，算法不存在")
        count = await algorithm_repository.delete_by_ids(db, ids_to_delete)
        return count

    # ── 状态机 ──────────────────────────────────────

    @staticmethod
    async def update_status(
        db: AsyncSession,
        algorithm_id: int,
        target_status: int,
    ) -> None:
        """
        修改算法状态（对齐 Java validateStatusTransition 逻辑）

        - 校验目标状态是合法枚举值 (1-6)
        - 终态(已发布4/已停用5)不允许变更，已归档(6)除外
        - 不允许直接跳转到已发布(4)，必须从待审核(3)流转
        """
        algorithm = await algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException("算法不存在")

        # 校验目标状态是合法值
        valid_statuses = {
            AlgorithmStatus.DRAFT, AlgorithmStatus.TESTING,
            AlgorithmStatus.PENDING_AUDIT, AlgorithmStatus.PUBLISHED,
            AlgorithmStatus.DISABLED, AlgorithmStatus.ARCHIVED,
        }
        if target_status not in valid_statuses:
            raise BusinessException(f"无效的状态值: {target_status}")

        current_status = algorithm.status

        # 终态校验：已发布/已停用不允许变更（已归档可重新启用）
        final_statuses = {AlgorithmStatus.PUBLISHED, AlgorithmStatus.DISABLED}
        if current_status in final_statuses:
            raise BusinessException("终态算法不允许修改状态")

        # 不允许直接跳转到已发布
        if target_status == AlgorithmStatus.PUBLISHED and current_status != AlgorithmStatus.PENDING_AUDIT:
            raise BusinessException("算法必须经过审核才能发布")

        await algorithm_repository.update_status(db, algorithm_id, target_status)

    # ── 审核 ──────────────────────────────────────

    @staticmethod
    async def audit_algorithm(
        db: AsyncSession,
        algorithm_id: int,
        audit_by: int,
        passed: bool,
        remark: Optional[str] = None,
    ) -> None:
        """
        审核算法

        - passed=True: 通过，状态变为已发布
        - passed=False: 驳回，必须填 remark，状态回到测试中
        """
        algorithm = await algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException("算法不存在")

        if algorithm.status != AlgorithmStatus.PENDING_AUDIT:
            raise BusinessException("仅待审核状态的算法可审核")

        if not passed and not remark:
            raise BusinessException("驳回时必须填写原因")

        await algorithm_repository.audit(
            db=db,
            algorithm_id=algorithm_id,
            audit_by=audit_by,
            passed=passed,
            remark=remark,
        )

    # ── 版本控制 ──────────────────────────────────────

    @staticmethod
    async def create_version(
        db: AsyncSession,
        algorithm_id: int,
        version: str,
        change_log: Optional[str] = None,
        status: Optional[int] = None,
        config_json: Optional[str] = None,
        model_file_id: Optional[int] = None,
        is_active: int = 0,
    ) -> int:
        """
        新增版本 (对齐 Java SysAlgorithmVersion 字段)

        - 校验版本号唯一（版本历史表）
        - 将当前版本归档到版本历史表
        - 更新算法主表的 version

        注: 预测缓存失效由调用方（router）负责
        """
        algorithm = await algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException("算法不存在")

        if await algorithm_repository.check_version_exists(db, algorithm_id, version):
            raise BusinessException(f"版本号 {version} 已存在")

        # 归档当前版本到历史表
        if not await algorithm_repository.check_version_exists(db, algorithm_id, algorithm.version):
            await algorithm_repository.create_version(
                db=db,
                algorithm_id=algorithm_id,
                version=algorithm.version,
                change_log=f"自动归档: 升级到 {version} 前",
                status=algorithm.status,
            )

        # 更新算法主表 version
        await algorithm_repository.update(db, algorithm, {"version": version})

        # 记录新版本到历史表
        await algorithm_repository.create_version(
            db=db,
            algorithm_id=algorithm_id,
            version=version,
            change_log=change_log,
            status=status,
            config_json=config_json,
            model_file_id=model_file_id,
            is_active=is_active,
        )

        return algorithm_id

    @staticmethod
    async def list_versions(db: AsyncSession, algorithm_id: int) -> list[dict[str, Any]]:
        """查询算法版本历史"""
        versions = await algorithm_repository.list_versions(db, algorithm_id)
        return [
            {
                "id": v.id,
                "algorithmId": v.algorithm_id,
                "version": v.version,
                "changeLog": v.change_log,
                "status": v.status,
                "configJson": v.config_json,
                "modelFileId": v.model_file_id,
                "isActive": v.is_active,
                "createTime": format_time(v.create_time),
                "updateTime": format_time(v.update_time),
            }
            for v in versions
        ]

    @staticmethod
    async def rollback_version(
        db: AsyncSession,
        algorithm_id: int,
        version_id: int,
    ) -> None:
        """回滚到指定版本"""
        algorithm = await algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException("算法不存在")

        # 仅已停用/已发布状态可回滚
        if algorithm.status not in (AlgorithmStatus.DISABLED, AlgorithmStatus.PUBLISHED):
            raise BusinessException("仅已停用/已发布状态的算法可回滚")

        result = await algorithm_repository.rollback_to_version(db, algorithm_id, version_id)
        if not result:
            raise BusinessException("版本不存在或不属于该算法")

    # ── 导入/导出 ──────────────────────────────────────

    @staticmethod
    async def export_algorithm(db: AsyncSession, algorithm_id: int) -> str:
        """
        导出单个算法为 JSON 字符串（对齐 Java exportAlgorithmJson）
        """
        algorithm = await algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException("算法不存在")

        # 获取父算法名称
        parent_name = ""
        if algorithm.parent_id and algorithm.parent_id > 0:
            parent = await algorithm_repository.get_by_id(db, algorithm.parent_id)
            parent_name = parent.name if parent else ""

        export_data = {
            "formatVersion": "1.0",
            "name": algorithm.name,
            "type": algorithm.type,
            "parentName": parent_name,
            "version": algorithm.version,
            "description": algorithm.description,
            "importPath": algorithm.import_path,
            "flops": algorithm.flops,
            "params": algorithm.params,
            "status": algorithm.status,
            "exportTime": datetime.now(timezone.utc).isoformat(),
        }

        return json.dumps(export_data, ensure_ascii=False, indent=2)

    @staticmethod
    async def validate_import_package(file_bytes: bytes, filename: str = "") -> str:
        """
        校验导入包格式（对齐 Java validateImport：解析 JSON，返回校验消息字符串）

        Returns:
            校验通过的消息字符串
        Raises:
            BusinessException: 校验失败
        """
        if not file_bytes:
            raise BusinessException("导入文件不能为空")

        if not filename.lower().endswith(".json"):
            raise BusinessException("仅支持 .json 格式的算法导出文件")

        try:
            root = json.loads(file_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise BusinessException(f"导入文件解析失败: {e}")

        name = root.get("name")
        if not name or not str(name).strip():
            raise BusinessException("导入文件缺少必填字段: name")

        type_ = root.get("type")
        if not type_ or not str(type_).strip():
            raise BusinessException("导入文件缺少必填字段: type")

        return f"校验通过: 算法名称={name}, 类型={type_}"

    @staticmethod
    async def import_algorithm(db: AsyncSession, file_bytes: bytes, filename: str = "") -> int:
        """
        导入算法包（对齐 Java importAlgorithm：解析 JSON 文件）
        """
        if not file_bytes:
            raise BusinessException("导入文件不能为空")

        if not filename.lower().endswith(".json"):
            raise BusinessException("仅支持 .json 格式的算法导出文件")

        root = json.loads(file_bytes.decode("utf-8"))

        name = root.get("name")
        if not name or not str(name).strip():
            raise BusinessException("导入失败: 缺少算法名称")

        type_ = root.get("type", "")
        description = root.get("description", "")
        import_path = root.get("importPath", "")
        version = root.get("version", "0.0.1")

        # 名称唯一性校验
        existing = await algorithm_repository.get_list_with_keywords(db, name)
        for algo in existing:
            if algo.name == name:
                raise BusinessException(f"算法名称 '{name}' 已存在")

        # 创建算法记录（对齐 Java：parentId=0，status=DRAFT）
        algorithm = SysAlgorithm(
            parent_id=0,
            type=type_,
            name=name,
            import_path=import_path,
            description=description,
            status=AlgorithmStatus.DRAFT,
            version=version,
        )

        created = await algorithm_repository.create(db, algorithm)
        return created.id

    # ── 监控 ──────────────────────────────────────

    @staticmethod
    async def get_monitor_data(db: AsyncSession, algorithm_id: int) -> dict[str, Any]:
        """获取算法监控数据（对齐 Java AlgorithmMonitorVO 字段）"""
        algorithm = await algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException("算法不存在")

        stats, today_calls = await asyncio.gather(
            algorithm_repository.get_monitor_stats(db, algorithm_id),
            algorithm_repository.get_today_call_count(db, algorithm_id),
        )

        total_calls = stats["totalCalls"]
        # 对齐 Java: totalCalls=0 时 successRate=100.0
        if total_calls > 0:
            rate = stats["successRate"]
            success_rate = rate * 100 if rate <= 1 else rate
        else:
            success_rate = 100.0

        return {
            "callCount": total_calls,
            "avgTime": round(stats["avgTime"], 2),
            "successRate": round(success_rate, 2),
            "todayCallCount": today_calls,
        }

    @staticmethod
    async def get_monitor_stats_report(db: AsyncSession, algorithm_id: int) -> dict[str, Any]:
        """获取算法监控统计报表（对齐 Java：直接返回 getMonitorData）"""
        return await AlgorithmService.get_monitor_data(db, algorithm_id)
