"""
算法导出处理器
"""
from __future__ import annotations

import io

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import BATCH_SIZE
from app.repository.algorithm_repository import (AlgorithmStatus,
                                                 algorithm_repository)
from app.service.import_export.file_generator import write_csv, write_excel
from app.service.import_export.models import ExportContext, ExportFieldConfig
from app.service.import_export.registry import ExportHandler

_ALGORITHM_STATUS_LABELS = {
    AlgorithmStatus.DRAFT: "草稿",
    AlgorithmStatus.TESTING: "测试中",
    AlgorithmStatus.PENDING_AUDIT: "待审核",
    AlgorithmStatus.PUBLISHED: "已发布",
    AlgorithmStatus.DISABLED: "已停用",
    AlgorithmStatus.ARCHIVED: "已归档",
}


class AlgorithmExportHandler(ExportHandler):

    def get_module(self) -> str:
        return "algorithm"

    async def estimate_count(self, db: AsyncSession, query_params: dict) -> int:
        keywords = query_params.get("keywords")
        algos = await algorithm_repository.get_list_with_keywords(db, keywords)
        return len(algos)

    async def export(
        self,
        db: AsyncSession,
        ctx: ExportContext,
        output: io.BytesIO,
        progress_cb,
        cancel_cb,
    ) -> None:
        params = ctx.query_params
        keywords = params.get("keywords")
        algos = await algorithm_repository.get_list_with_keywords(db, keywords)
        total = ctx.total_count or len(algos)

        all_rows: list[dict] = []
        for i, a in enumerate(algos, 1):
            all_rows.append(_algorithm_to_row(a))
            if i % BATCH_SIZE == 0:
                await progress_cb(min(i, total), total)
                if await cancel_cb():
                    break
        await progress_cb(len(all_rows), total)

        fields = self.filter_fields(ctx.selected_fields)
        if ctx.format == "csv":
            write_csv(fields, all_rows, output)
        else:
            write_excel(fields, all_rows, output)

    def get_field_configs(self) -> list[ExportFieldConfig]:
        return [
            ExportFieldConfig(field="id", label="ID", order=1),
            ExportFieldConfig(field="name", label="算法名称", order=2),
            ExportFieldConfig(field="parent_id", label="父算法ID", order=3),
            ExportFieldConfig(field="type", label="算法类型", order=4),
            ExportFieldConfig(field="path", label="模型文件路径", order=5),
            ExportFieldConfig(field="import_path", label="导入路径", order=6),
            ExportFieldConfig(field="description", label="描述", order=7),
            ExportFieldConfig(field="version", label="版本", order=8),
            ExportFieldConfig(field="status_label", label="状态", order=9),
            ExportFieldConfig(field="size", label="大小", order=10),
            ExportFieldConfig(field="flops", label="FLOPs", order=11),
            ExportFieldConfig(field="params", label="参数量", order=12),
            ExportFieldConfig(field="create_time", label="创建时间", order=13, date_format="%Y-%m-%d %H:%M:%S"),
        ]


def _algorithm_to_row(a) -> dict:
    status_value = int(a.status) if a.status is not None else AlgorithmStatus.DRAFT
    return {
        "id": a.id,
        "name": a.name or "",
        "parent_id": a.parent_id if a.parent_id is not None else "",
        "type": a.type or "",
        "path": a.path or "",
        "import_path": a.import_path or "",
        "description": a.description or "",
        "version": a.version or "",
        "status_label": _ALGORITHM_STATUS_LABELS.get(status_value, ""),
        "size": a.size or "",
        "flops": a.flops or "",
        "params": a.params or "",
        "create_time": a.create_time,
    }
