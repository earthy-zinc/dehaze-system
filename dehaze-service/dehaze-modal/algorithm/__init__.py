"""去雾算法实现包。

每个算法目录的 `run.py` 导出 `dehaze(image, model_path)`；本模块不做任何导入，
使 `import algorithm.model_loader` 这类轻量用途（权重可用性探测、能力发现）
不被包级重依赖（torch/thop）牵连。
"""
