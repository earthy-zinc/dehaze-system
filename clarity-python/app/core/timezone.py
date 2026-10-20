"""时区与时段工具（服务端统一 Asia/Shanghai，见 API 规范 §6.2）"""

from datetime import datetime
from zoneinfo import ZoneInfo

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def is_peak_hour(at_time: datetime) -> bool:
    """高峰时段判定：周一至周五 9:00-12:00、14:00-18:00（Asia/Shanghai）。

    售价（sys_ai_model_price）与成本价（sys_ai_model_cost）档位判定共用。
    """
    if at_time.tzinfo is None:
        at_time = at_time.replace(tzinfo=SHANGHAI_TZ)
    local = at_time.astimezone(SHANGHAI_TZ)
    if local.weekday() >= 5:
        return False
    minutes = local.hour * 60 + local.minute
    return 540 <= minutes < 720 or 840 <= minutes < 1080
