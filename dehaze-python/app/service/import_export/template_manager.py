"""
导入模板生成器
"""

from __future__ import annotations

import csv
import io

from openpyxl import Workbook

from app.service.import_export.registry import ImportHandler


def generate_template_excel(handler: ImportHandler) -> bytes:
    headers, sample_rows = _build_template_data(handler)
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(headers)
    for row in sample_rows:
        ws.append(row)
    output = io.BytesIO()
    wb.save(output)
    return output.getvalue()


def generate_template_csv(handler: ImportHandler) -> bytes:
    headers, sample_rows = _build_template_data(handler)
    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.writer(output)
    writer.writerow(headers)
    for row in sample_rows:
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def _build_template_data(handler: ImportHandler) -> tuple[list[str], list[list]]:
    fields = handler.get_field_configs()
    headers = [("*" if f.required else "") + f.label for f in fields]
    sample_rows = [
        [sample.get(f.field, "") for f in fields] for sample in handler.get_template_sample_data()
    ]
    return headers, sample_rows
