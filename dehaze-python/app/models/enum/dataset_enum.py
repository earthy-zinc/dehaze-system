"""
数据集相关枚举

sys_item_file.haze_level 字段为 VARCHAR(32)，可空，支持多种规范：
- light / medium / heavy（人工分级）
- beta=0.5（β 参数单值）
- A=0.8,beta=0.2（A+β 双参数）
- 空值（未标注或无雾/真实雾图）
不做硬性枚举校验，详见需求规格.md 2.6.2 节。
"""
