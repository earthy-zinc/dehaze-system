"""
模型层

目录结构：
- entity/   数据库实体模型（ORM，__init__ 聚合全部实体做 metadata 注册）
- schema/   Pydantic 模型（请求/响应）
- enum/     枚举类型
- base.py   基础模型类

⚠ 数据库 DDL 的事实来源是 `config/sql/schema/`（三端共享，非 Alembic）；
entity 聚合将全部实体注册进 Base.metadata，供与 config/sql 做 schema
漂移核对等场景使用，实体增删须同步两处。
"""

import app.models.entity  # noqa: F401  # 触发全部实体注册进 Base.metadata
