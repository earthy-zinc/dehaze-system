"""可观测性增量迁移：补齐开发库新增列（幂等，缺列才 ALTER）。

覆盖 2026-08-29 可观测性改造新增字段：
- sys_ai_trace: error_detail(JSON) / trace_type(varchar default 'conversation')
- sys_ai_llm_call: attempts(JSON)
- sys_ai_agent_thought: agent_code(varchar) / is_subagent(tinyint default 0)

用法：cd dehaze-system && dehaze-python/.venv/bin/python dehaze-test/scripts/migrate_observability.py [database]
默认数据库取 .env MYSQL_DATABASE（dehaze）。幂等：已存在的列跳过。
"""

import sys
from pathlib import Path

# 让脚本不依赖 PYTHONPATH 也能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.mysql import execute, query

# (表, 列, 列定义 ALTER 片段)
_MIGRATIONS = [
    (
        "sys_ai_trace",
        "error_detail",
        "ADD COLUMN `error_detail` json NULL COMMENT '异常详情(消息+堆栈截断,失败/中断时填充)' AFTER `context_snapshot`",
    ),
    (
        "sys_ai_trace",
        "trace_type",
        "ADD COLUMN `trace_type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'conversation' "
        "COMMENT '过程链类型(conversation主对话; summary会话摘要压缩; memory_extraction记忆提取; suggestion类似问题推荐; step_summary步骤摘要)' AFTER `agent_code`",
    ),
    (
        "sys_ai_llm_call",
        "attempts",
        "ADD COLUMN `attempts` json NULL COMMENT '物理调用尝试明细JSON(逐Key/逐路由: provider_id/key_id/model/status/error_code/latency_ms)' AFTER `output_snapshot`",
    ),
    (
        "sys_ai_agent_thought",
        "agent_code",
        "ADD COLUMN `agent_code` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL "
        "COMMENT '此步骤来源Agent编码(关联sys_ai_agent.agent_code,为空表示主Agent)' AFTER `position`",
    ),
    (
        "sys_ai_agent_thought",
        "is_subagent",
        "ADD COLUMN `is_subagent` tinyint NOT NULL DEFAULT 0 COMMENT '是否为子Agent的推理步骤(0:否,主Agent;1:是,子Agent)' AFTER `agent_code`",
    ),
]


def main() -> int:
    database = sys.argv[1] if len(sys.argv) > 1 else None
    applied: list[str] = []
    skipped: list[str] = []
    for table, column, alter in _MIGRATIONS:
        existing = {row["Field"] for row in query(f"SHOW COLUMNS FROM `{table}`", database=database)}
        if column in existing:
            skipped.append(f"{table}.{column}")
            continue
        execute(f"ALTER TABLE `{table}` {alter}", database=database)
        applied.append(f"{table}.{column}")
    print("已迁移:", applied or "（无）")
    print("已存在跳过:", skipped or "（无）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
