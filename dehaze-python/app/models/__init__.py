"""
模型层

目录结构：
- entity/   数据库实体模型（ORM，__init__ 聚合全部实体做 metadata 注册）
- schema/   Pydantic 模型（请求/响应）
- enum/     枚举类型
- base.py   基础模型类

⚠ entity 聚合是 alembic autogenerate 的注册入口（migrations/env.py
经 `import app.models` 触发），不得移除；实体增删须同步清单。
"""

import app.models.entity  # noqa: F401  # 触发全部实体注册进 Base.metadata（alembic 依赖）
