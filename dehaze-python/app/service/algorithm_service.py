"""
算法服务

提供算法 CRUD、状态机、审核、版本控制、监控功能
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from algorithm.model_loader import check_model_exists
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_algorithm import SysAlgorithm
from app.repository.algorithm_repository import (
    AlgorithmStatus,
    algorithm_repository,
)
from app.repository.favorite_repository import favorite_repository
from app.utils.datetime_utils import format_time
from app.utils.file import convert_size


class AlgorithmService:
    """算法服务"""

    def __init__(self, algorithm_repository=algorithm_repository):
        self.algorithm_repository = algorithm_repository

    def _to_vo(self, algorithm: SysAlgorithm) -> dict[str, Any]:
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

    def _build_algorithm_tree(self, algorithms: list[SysAlgorithm]) -> list[dict[str, Any]]:
        """构建算法树形结构"""
        algorithm_dict = {
            algorithm.id: {**self._to_vo(algorithm), "children": []} for algorithm in algorithms
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

    async def get_algorithm_list(
        self, db: AsyncSession, keywords: str | None = None
    ) -> list[dict[str, Any]]:
        """获取算法树形表格"""
        algorithms = await self.algorithm_repository.get_list_with_keywords(db, keywords)
        return self._build_algorithm_tree(algorithms)

    async def get_algorithm_options(self, db: AsyncSession) -> list[dict[str, Any]]:
        """获取模型下拉选项列表"""
        return await self.algorithm_repository.get_algorithm_options(db)

    async def list_all_algorithms(self, db: AsyncSession) -> list[dict[str, Any]]:
        """获取所有算法扁平列表（不构建树形结构），用于前端下拉选择"""
        algorithms = await self.algorithm_repository.get_list_with_keywords(db, None)
        return [self._to_vo(algo) for algo in algorithms]

    async def get_algorithm_by_id(self, db: AsyncSession, algorithm_id: int) -> dict[str, Any]:
        """根据 ID 获取算法信息"""
        algorithm = await self.algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在")
        return self._to_vo(algorithm)

    async def create_algorithm(self, db: AsyncSession, data: dict[str, Any]) -> int:
        """新增算法"""
        name = data.get("name", "")
        # 名称唯一性（A0501，仅活跃行；软删行不占键位可重建）
        if name:
            existing = await self.algorithm_repository.get_list_with_keywords(db, name)
            if any(algo.name == name for algo in existing):
                raise BusinessException(ResultCode.DATA_EXISTS, f"算法名称 '{name}' 已存在")

        algorithm = SysAlgorithm(
            parent_id=data.get("parentId", 0),
            type=data.get("type", ""),
            name=name,
            path=data.get("path", ""),
            import_path=data.get("importPath", ""),
            description=data.get("description", ""),
            status=data.get("status", AlgorithmStatus.DRAFT),
        )

        # path 指向具体文件时，通过 Nginx 静态服务校验可访问性并回填 size
        path_value = data.get("path", "")
        if path_value:
            size_bytes = await asyncio.to_thread(check_model_exists, path_value)
            if size_bytes is not None:
                algorithm.size = convert_size(size_bytes)

        created = await self.algorithm_repository.create(db, algorithm)
        return created.id

    async def update_algorithm(
        self, db: AsyncSession, algorithm_id: int, data: dict[str, Any]
    ) -> None:
        """修改算法"""
        algorithm = await self.algorithm_repository.get_by_id(db, algorithm_id)

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
        if "status" in data:
            update_data["status"] = data["status"]

        await self.algorithm_repository.update(db, algorithm, update_data)

    async def delete_algorithm_single(self, db: AsyncSession, algorithm_id: int) -> int:
        """删除单个算法（含子算法）"""
        return await self.delete_algorithms(db, [algorithm_id])

    async def delete_algorithms(self, db: AsyncSession, algorithm_ids: list[int]) -> int:
        """批量删除算法（软删，级联子孙算法），对齐 Java deleteAlgorithms

        Java/Python/Go: 任一算法不存在时抛 RESOURCE_NOT_FOUND；
        仅草稿/已停用/已归档状态可删除（A0502，级联子算法一并校验）；
        软删 deleted=行 id（对齐 Java @TableLogic delval=id），释放唯一键位支持删后重建。
        """
        all_algorithms = await self.algorithm_repository.get_list_with_keywords(db)
        id_to_algo = {a.id: a for a in all_algorithms}
        for algorithm_id in algorithm_ids:
            if algorithm_id not in id_to_algo:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在")

        ids_to_delete = await self.algorithm_repository.get_with_children_ids(db, algorithm_ids)
        for deleted_id in ids_to_delete:
            algo = id_to_algo.get(deleted_id)
            if algo and algo.status not in AlgorithmStatus.DELETABLE_STATUSES:
                raise BusinessException(
                    ResultCode.DATA_STATE_NOT_ALLOW,
                    f"算法[{algo.name}]当前状态不允许删除，请先停用或归档",
                )
        count = await self.algorithm_repository.soft_delete_by_ids(db, ids_to_delete)
        # 失效联动：标记相关收藏为已失效（对齐 Java SysAlgorithmServiceImpl.deleteAlgorithms）
        await favorite_repository.mark_invalid(db, "algorithm", ids_to_delete)
        return count

    # ── 状态机 ──────────────────────────────────────

    async def update_status(
        self,
        db: AsyncSession,
        algorithm_id: int,
        target_status: int,
    ) -> None:
        """修改算法状态（三端统一状态流转白名单，
        对齐 Java validateStatusTransition / Go CanTransitionTo）"""
        algorithm = await self.algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在")

        allowed = AlgorithmStatus.ALLOWED_TRANSITIONS.get(algorithm.status)
        if allowed is None or target_status not in allowed:
            raise BusinessException(
                ResultCode.DATA_STATE_NOT_ALLOW,
                f"不允许将算法状态从 {algorithm.status} 变更为 {target_status}",
            )

        await self.algorithm_repository.update_status(db, algorithm_id, target_status)

    # ── 审核 ──────────────────────────────────────

    async def audit_algorithm(
        self,
        db: AsyncSession,
        algorithm_id: int,
        audit_by: int,
        passed: bool,
        remark: str | None = None,
    ) -> None:
        """
        审核算法

        - passed=True: 通过，状态变为已发布
        - passed=False: 驳回，必须填 remark，状态回到测试中
        """
        algorithm = await self.algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException("算法不存在")

        if algorithm.status != AlgorithmStatus.PENDING_AUDIT:
            raise BusinessException("仅待审核状态的算法可审核")

        if not passed and not remark:
            raise BusinessException("驳回时必须填写原因")

        await self.algorithm_repository.audit(
            db=db,
            algorithm_id=algorithm_id,
            audit_by=audit_by,
            passed=passed,
            remark=remark,
        )

    # ── 版本控制 ──────────────────────────────────────

    async def create_version(
        self,
        db: AsyncSession,
        algorithm_id: int,
        version: str,
        change_log: str | None = None,
        config_json: str | None = None,
        model_file_id: int | None = None,
    ) -> int:
        """
        新增版本 (对齐 Java SysAlgorithmVersionServiceImpl.addVersion)

        - 校验版本号唯一（活跃行；uk_algo_version 含 deleted，软删行不占键位）
        - is_active 单活跃版本管理：旧活跃版本置非活跃，新版本为活跃
        - 更新算法主表的 version
        """
        algorithm = await self.algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在")

        if await self.algorithm_repository.check_version_exists(db, algorithm_id, version):
            raise BusinessException(f"版本号 {version} 已存在")

        await self.algorithm_repository.deactivate_active_versions(db, algorithm_id)
        await self.algorithm_repository.create_version(
            db=db,
            algorithm_id=algorithm_id,
            version=version,
            change_log=change_log,
            status=algorithm.status,
            config_json=config_json,
            model_file_id=model_file_id,
            is_active=1,
        )
        await self.algorithm_repository.update(db, algorithm, {"version": version})
        return algorithm_id

    async def list_versions(self, db: AsyncSession, algorithm_id: int) -> list[dict[str, Any]]:
        """查询算法版本历史"""
        versions = await self.algorithm_repository.list_versions(db, algorithm_id)
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

    async def rollback_version(
        self,
        db: AsyncSession,
        algorithm_id: int,
        version_id: int,
    ) -> None:
        """回滚到指定版本（对齐 Java rollbackToVersion：is_active 切换 + 防重复回滚）"""
        algorithm = await self.algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在")

        version = await self.algorithm_repository.get_version_by_id(db, version_id)
        if not version or version.algorithm_id != algorithm_id:
            raise BusinessException("版本不存在或不属于该算法")

        if version.is_active:
            raise BusinessException("当前已是该版本，无需回滚")

        await self.algorithm_repository.deactivate_active_versions(db, algorithm_id)
        version.is_active = 1
        await db.flush()
        await self.algorithm_repository.update(db, algorithm, {"version": version.version})

    # ── 监控 ──────────────────────────────────────

    async def get_monitor_data(self, db: AsyncSession, algorithm_id: int) -> dict[str, Any]:
        """获取算法监控数据（对齐 Java AlgorithmMonitorVO 字段）"""
        algorithm = await self.algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException("算法不存在")

        stats, today_calls = await asyncio.gather(
            self.algorithm_repository.get_monitor_stats(db, algorithm_id),
            self.algorithm_repository.get_today_call_count(db, algorithm_id),
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

    async def get_monitor_stats_report(
        self, db: AsyncSession, algorithm_id: int, days: int = 7
    ) -> list[dict[str, Any]]:
        """获取算法监控统计报表（对齐 Java：最近 days 天每天一条，含无数据天）"""
        algorithm = await self.algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException("算法不存在")
        by_date = await self.algorithm_repository.get_monitor_stats_by_date(db, algorithm_id, days)
        today = datetime.now().date()
        result: list[dict[str, Any]] = []
        for i in range(days - 1, -1, -1):
            day = today - timedelta(days=i)
            key = str(day)
            row = by_date.get(key)
            count = int(row.call_count) if row else 0
            avg_time = round(float(row.avg_time or 0), 2) if row else 0.0
            success_count = int(row.success_count) if row else 0
            success_rate = round(success_count / count * 100, 2) if count > 0 else 0.0
            result.append(
                {
                    "date": key,
                    "callCount": count,
                    "avgTime": avg_time,
                    "successRate": success_rate,
                }
            )
        return result


algorithm_service = AlgorithmService()
