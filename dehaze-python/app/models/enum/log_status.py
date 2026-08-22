"""
预测/评估日志状态枚举

对齐 config/sql/schema/sys_pred_log.sql 与 sys_eval_log.sql 中 status 字段（tinyint）：
1=处理中 / 2=已完成 / 3=失败 / 4=已取消
"""

from enum import IntEnum


class LogStatus(IntEnum):
    PROCESSING = 1
    COMPLETED = 2
    FAILED = 3
    CANCELLED = 4
