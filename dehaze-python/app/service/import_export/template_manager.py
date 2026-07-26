"""
导入模板生成器
"""
from __future__ import annotations

import csv
import io

from openpyxl import Workbook

from app.service.import_export.registry import ImportHandler


def generate_template_excel(handler: ImportHandler) -> bytes:
    fields = handler.get_field_configs()
    samples = handler.get_template_sample_data()
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    headers = []
    for f in fields:
        prefix = "*" if f.required else ""
        headers.append(f"{prefix}{f.label}")
    ws.append(headers)
    for sample in samples:
        ws.append([sample.get(f.field, "") for f in fields])
    output = io.BytesIO()
    wb.save(output)
    return output.getvalue()


def generate_template_csv(handler: ImportHandler) -> bytes:
    fields = handler.get_field_configs()
    samples = handler.get_template_sample_data()
    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.writer(output)
    headers = []
    for f in fields:
        prefix = "*" if f.required else ""
        headers.append(f"{prefix}{f.label}")
    writer.writerow(headers)
    for sample in samples:
        writer.writerow([sample.get(f.field, "") for f in fields])
    return output.getvalue().encode("utf-8")
