"""
导出文件生成器（Excel/CSV）
"""
from __future__ import annotations

import csv
import io
from typing import Iterable

from openpyxl import Workbook

from app.service.import_export.models import ExportFieldConfig


def write_excel(
    fields: list[ExportFieldConfig],
    rows: Iterable[dict],
    output: io.BytesIO,
) -> None:
    visible_fields = [f for f in sorted(fields, key=lambda x: x.order) if not f.hidden]
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append([f.label for f in visible_fields])
    for row in rows:
        ws.append([_to_cell_value(row.get(f.field), f) for f in visible_fields])
    wb.save(output)


def write_csv(
    fields: list[ExportFieldConfig],
    rows: Iterable[dict],
    output: io.BytesIO,
) -> None:
    visible_fields = [f for f in sorted(fields, key=lambda x: x.order) if not f.hidden]
    text_buf = io.TextIOWrapper(output, encoding="utf-8", write_through=True)
    text_buf.write("\ufeff")
    writer = csv.writer(text_buf)
    writer.writerow([f.label for f in visible_fields])
    for row in rows:
        writer.writerow([_to_cell_value(row.get(f.field), f) for f in visible_fields])
    text_buf.detach()


def _to_cell_value(value, field: ExportFieldConfig):
    if value is None:
        return ""
    if field.date_format and hasattr(value, "strftime"):
        return value.strftime(field.date_format)
    return value
