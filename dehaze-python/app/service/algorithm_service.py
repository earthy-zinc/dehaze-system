"""
算法服务

提供算法 CRUD、状态机、审核、版本控制、导入/导出、监控功能
"""

import io
import json
import os
import zipfile
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import BusinessException
from app.core.code import ResultCode
from app.models.entity.sys_algorithm import SysAlgorithm, SysAlgorithmVersion
from app.repository.algorithm_repository import (
    algorithm_repository,
    AlgorithmStatus,
    can_transition,
)
from app.utils.datetime_utils import format_time
from app.utils.file import get_file_size


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
    async def get_algorithm_by_id(db: AsyncSession, algorithm_id: int) -> dict[str, Any] | None:
        """根据 ID 获取算法信息"""
        algorithm = await algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            return None
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

        # 如果路径是有效文件，获取文件大小
        if "path" in data and os.path.isfile(data["path"]):
            algorithm.size = get_file_size(data["path"])

        created = await algorithm_repository.create(db, algorithm)
        return created.id

    @staticmethod
    async def update_algorithm(db: AsyncSession, algorithm_id: int, data: dict[str, Any]) -> None:
        """修改算法"""
        algorithm = await algorithm_repository.get_by_id(db, algorithm_id)

        if not algorithm:
            raise BusinessException("算法不存在", ResultCode.ALGORITHM_NOT_FOUND.code)

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
        if "version" in data:
            update_data["version"] = data["version"]

        await algorithm_repository.update(db, algorithm, update_data)

    @staticmethod
    async def delete_algorithm_single(db: AsyncSession, algorithm_id: int) -> int:
        """删除单个算法（含子算法）"""
        return await AlgorithmService.delete_algorithms(db, [algorithm_id])

    @staticmethod
    async def delete_algorithms(db: AsyncSession, algorithm_ids: list[int]) -> int:
        """批量删除算法（包含子算法），仅草稿/已停用状态可删除"""
        for aid in algorithm_ids:
            algorithm = await algorithm_repository.get_by_id(db, aid)
            if not algorithm:
                raise BusinessException(f"算法不存在: id={aid}", ResultCode.ALGORITHM_NOT_FOUND.code)
            if algorithm.status not in (AlgorithmStatus.DRAFT, AlgorithmStatus.DISABLED):
                raise BusinessException(
                    f"算法 {algorithm.name} 当前状态不允许删除",
                    ResultCode.ALGORITHM_STATUS_NOT_ALLOWED.code,
                )
        ids_to_delete = await algorithm_repository.get_with_children_ids(db, algorithm_ids)
        if ids_to_delete:
            return await algorithm_repository.delete_by_ids(db, ids_to_delete)
        return 0

    # ── 状态机 ──────────────────────────────────────

    @staticmethod
    async def update_status(
        db: AsyncSession,
        algorithm_id: int,
        target_status: int,
    ) -> None:
        """
        修改算法状态（状态机流转）

        流转规则:
        - 草稿 → 测试中（提交测试）
        - 测试中 → 待审核（测试通过）
        - 待审核 → 已发布（审核通过） / 测试中（审核驳回）
        - 已发布 → 已停用
        - 已停用 → 已发布 / 已归档
        - 已归档为终态
        """
        algorithm = await algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException("算法不存在", ResultCode.ALGORITHM_NOT_FOUND.code)

        if not can_transition(algorithm.status, target_status):
            raise BusinessException(
                f"状态不允许从 {algorithm.status} 流转到 {target_status}",
                ResultCode.ALGORITHM_STATUS_NOT_ALLOWED.code,
            )

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
            raise BusinessException("算法不存在", ResultCode.ALGORITHM_NOT_FOUND.code)

        if algorithm.status != AlgorithmStatus.PENDING_AUDIT:
            raise BusinessException(
                "仅待审核状态的算法可审核",
                ResultCode.ALGORITHM_STATUS_NOT_ALLOWED.code,
            )

        if not passed and not remark:
            raise BusinessException(
                "驳回时必须填写原因",
                ResultCode.ALGORITHM_AUDIT_REMARK_REQUIRED.code,
            )

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
            raise BusinessException("算法不存在", ResultCode.ALGORITHM_NOT_FOUND.code)

        if await algorithm_repository.check_version_exists(db, algorithm_id, version):
            raise BusinessException(
                f"版本号 {version} 已存在",
                ResultCode.ALGORITHM_VERSION_EXISTS.code,
            )

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
            raise BusinessException("算法不存在", ResultCode.ALGORITHM_NOT_FOUND.code)

        # 仅已停用/已发布状态可回滚
        if algorithm.status not in (AlgorithmStatus.DISABLED, AlgorithmStatus.PUBLISHED):
            raise BusinessException(
                "仅已停用/已发布状态的算法可回滚",
                ResultCode.ALGORITHM_ROLLBACK_NOT_ALLOWED.code,
            )

        result = await algorithm_repository.rollback_to_version(db, algorithm_id, version_id)
        if not result:
            raise BusinessException(
                "版本不存在或不属于该算法",
                ResultCode.ALGORITHM_ROLLBACK_NOT_ALLOWED.code,
            )

    # ── 导入/导出 ──────────────────────────────────────

    @staticmethod
    async def export_algorithm(db: AsyncSession, algorithm_id: int) -> bytes:
        """
        同步导出单个算法为 ZIP

        ZIP 结构:
        - algorithm.json (算法元数据)
        - model/{模型文件} (如果 path 指向有效文件)
        """
        algorithm = await algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException("算法不存在", ResultCode.ALGORITHM_NOT_FOUND.code)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            meta = {
                "id": algorithm.id,
                "name": algorithm.name,
                "type": algorithm.type,
                "importPath": algorithm.import_path,
                "description": algorithm.description,
                "version": algorithm.version,
                "path": algorithm.path,
                "exportedAt": datetime.now(timezone.utc).isoformat(),
            }
            zf.writestr("algorithm.json", json.dumps(meta, ensure_ascii=False, indent=2))

            # 模型文件（如果存在）
            model_path = algorithm.path
            if model_path and os.path.isfile(model_path):
                with open(model_path, "rb") as f:
                    zf.writestr(f"model/{os.path.basename(model_path)}", f.read())

        buf.seek(0)
        return buf.getvalue()

    @staticmethod
    async def validate_import_package(file_bytes: bytes) -> dict[str, Any]:
        """
        校验导入包格式

        Returns:
            {"valid": bool, "name": str, "version": str, "message": str}
        """
        try:
            buf = io.BytesIO(file_bytes)
            with zipfile.ZipFile(buf, "r") as zf:
                names = zf.namelist()
                if "algorithm.json" not in names:
                    return {"valid": False, "message": "缺少 algorithm.json"}

                meta_bytes = zf.read("algorithm.json")
                meta = json.loads(meta_bytes.decode("utf-8"))

                # 必填字段校验
                if not meta.get("name"):
                    return {"valid": False, "message": "算法名称不能为空"}

                return {
                    "valid": True,
                    "name": meta.get("name"),
                    "version": meta.get("version", "v1.0.0"),
                    "message": "校验通过",
                }
        except zipfile.BadZipFile:
            return {"valid": False, "message": "ZIP 文件格式错误"}
        except json.JSONDecodeError:
            return {"valid": False, "message": "algorithm.json 解析失败"}
        except Exception as e:
            return {"valid": False, "message": f"校验失败: {e}"}

    @staticmethod
    async def import_algorithm(db: AsyncSession, file_bytes: bytes) -> int:
        """
        导入算法包

        - 校验包格式
        - 名称唯一性校验
        - 创建算法记录
        - 解压模型文件到本地
        """
        # 校验
        validation = await AlgorithmService.validate_import_package(file_bytes)
        if not validation["valid"]:
            raise BusinessException(
                validation["message"], ResultCode.ALGORITHM_IMPORT_FORMAT_ERROR.code
            )

        buf = io.BytesIO(file_bytes)
        with zipfile.ZipFile(buf, "r") as zf:
            meta = json.loads(zf.read("algorithm.json").decode("utf-8"))

            # 名称唯一性校验
            existing = await algorithm_repository.get_list_with_keywords(db, meta["name"])
            for algo in existing:
                if algo.name == meta["name"]:
                    raise BusinessException(
                        f"算法名称 {meta['name']} 已存在",
                        ResultCode.ALGORITHM_NAME_EXISTS.code,
                    )

            # 解压模型文件
            model_dir = os.path.join("models", "imported", meta["name"])
            os.makedirs(model_dir, exist_ok=True)
            model_path = ""
            for name in zf.namelist():
                if name.startswith("model/") and not name.endswith("/"):
                    file_data = zf.read(name)
                    basename = os.path.basename(name)
                    dest = os.path.join(model_dir, basename)
                    with open(dest, "wb") as f:
                        f.write(file_data)
                    model_path = dest

            # 创建算法记录
            algorithm = SysAlgorithm(
                parent_id=0,
                type=meta.get("type", ""),
                name=meta["name"],
                path=model_path,
                import_path=meta.get("importPath", ""),
                description=meta.get("description", ""),
                status=AlgorithmStatus.DRAFT,
                version=meta.get("version"),
            )
            if model_path and os.path.isfile(model_path):
                algorithm.size = get_file_size(model_path)

            created = await algorithm_repository.create(db, algorithm)
            return created.id

    # ── 监控 ──────────────────────────────────────

    @staticmethod
    async def get_monitor_data(db: AsyncSession, algorithm_id: int) -> dict[str, Any]:
        """获取算法监控数据"""
        algorithm = await algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException("算法不存在", ResultCode.ALGORITHM_NOT_FOUND.code)

        stats = await algorithm_repository.get_monitor_stats(db, algorithm_id)
        recent = await algorithm_repository.get_recent_calls(db, algorithm_id, limit=1)
        last_call_time = format_time(recent[0].create_time) if recent else None

        return {
            "algorithmId": algorithm_id,
            "algorithmName": algorithm.name,
            "totalCalls": stats["totalCalls"],
            "avgTime": stats["avgTime"],
            "successRate": stats["successRate"],
            "lastCallTime": last_call_time,
        }

    @staticmethod
    async def get_monitor_stats_report(db: AsyncSession, algorithm_id: int) -> dict[str, Any]:
        """获取算法监控统计报表"""
        algorithm = await algorithm_repository.get_by_id(db, algorithm_id)
        if not algorithm:
            raise BusinessException("算法不存在", ResultCode.ALGORITHM_NOT_FOUND.code)

        stats = await algorithm_repository.get_monitor_stats(db, algorithm_id)
        recent = await algorithm_repository.get_recent_calls(db, algorithm_id, limit=20)

        # 构建时间序列
        time_series = [
            {
                "time": format_time(log.create_time),
                "duration": log.time,
            }
            for log in recent
        ]
        time_series.reverse()  # 时间正序

        return {
            "algorithmId": algorithm_id,
            "timeSeries": time_series,
            "totalCalls": stats["totalCalls"],
            "avgTime": stats["avgTime"],
            "maxTime": stats["maxTime"],
            "minTime": stats["minTime"],
            "successRate": stats["successRate"],
        }
