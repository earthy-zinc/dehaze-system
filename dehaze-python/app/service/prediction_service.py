"""
预测服务 —— 编排算法预测流程

调用算法模块的 dehaze() 函数，管理输入/输出/存储
"""

import importlib
import io
import logging
import time
from pathlib import Path
from typing import Optional

import httpx
import PIL.Image
from sqlalchemy import select

from app.database import async_session_factory
from app.models.entity.sys_algorithm import SysAlgorithm
from algorithm.config import algorithm_config

logger = logging.getLogger(__name__)


class PredictionService:
    """模型预测编排服务"""

    # 算法模块名 → 目录名的映射
    ALGORITHM_REGISTRY = {
        "DCP": "DCP",
        "AODNet": "AODNet",
        "DehazeFormer": "DehazeFormer",
        "FFANet": "FFANet",
        "GridDehazeNet": "GridDehazeNet",
        "MSBDN": "MSBDN",
        "RIDCP": "RIDCP",
        "DarkChannelPrior": "DCP",
    }

    async def predict(self, algorithm_id: int, image_url: str, params: Optional[dict] = None) -> dict:
        """
        执行单次预测

        Args:
            algorithm_id: 算法ID
            image_url: 输入图片的完整URL
            params: 可选参数字典

        Returns:
            {
                "resultUrl": str,
                "resultMd5": str,
                "resultThumbnailUrl": Optional[str],
                "format": str,
            }
        """
        # 1. 从数据库获取算法信息
        algorithm = await self._get_algorithm(algorithm_id)

        # 2. 下载输入图片
        image_bytes = await self._download_image(image_url)

        # 3. 调用算法去雾
        result_bytes = await self._run_dehaze(
            algorithm.import_path or algorithm.name,
            image_bytes,
        )

        # 4. 上传结果图片
        result_url = await self._upload_result(result_bytes, algorithm.name)

        return {
            "resultUrl": result_url,
            "resultMd5": "",
            "resultThumbnailUrl": None,
            "format": "png",
        }

    @staticmethod
    async def _get_algorithm(algorithm_id: int) -> SysAlgorithm:
        """从数据库获取算法"""
        async with async_session_factory() as session:
            result = await session.execute(
                select(SysAlgorithm).where(SysAlgorithm.id == algorithm_id)
            )
            algorithm = result.scalar_one_or_none()
            if algorithm is None:
                raise ValueError(f"算法不存在: id={algorithm_id}")
            return algorithm

    @staticmethod
    async def _download_image(url: str) -> io.BytesIO:
        """从URL或本地路径下载图片"""
        import os.path

        # 处理本地文件路径 /api/v1/files/download/... → upload/...
        if url.startswith("/api/v1/files/download/"):
            local_path = url[len("/api/v1/files/download/"):]
            # 尝试相对于 upload 目录
            full_path = Path("upload") / local_path
            if full_path.exists():
                with open(full_path, "rb") as f:
                    return io.BytesIO(f.read())
            raise FileNotFoundError(f"本地文件不存在: {full_path}")

        # 处理绝对本地路径
        if not url.startswith("http://") and not url.startswith("https://"):
            local_path = Path(url)
            if local_path.exists():
                with open(local_path, "rb") as f:
                    return io.BytesIO(f.read())
            raise FileNotFoundError(f"图片文件不存在: {url}")

        # HTTP/HTTPS 下载
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return io.BytesIO(response.content)

    @staticmethod
    def _run_dehaze(import_path: str, image_bytes: io.BytesIO) -> io.BytesIO:
        """
        调用算法去雾

        Args:
            import_path: 算法导入路径，如 "DCP" 或 "AODNet"
            image_bytes: 输入的雾图

        Returns:
            去雾后的图片
        """
        # 映射到实际模块名
        module_name = PredictionService.ALGORITHM_REGISTRY.get(
            import_path, import_path
        )

        try:
            algo_module = importlib.import_module(f"algorithm.{module_name}.run")
        except ImportError:
            raise ValueError(
                f"算法模块未找到: algorithm.{module_name}.run, "
                f"请确认 import_path '{import_path}' 是否正确"
            )

        if not hasattr(algo_module, "dehaze"):
            raise ValueError(f"算法模块 {module_name} 未导出 dehaze() 函数")

        dehaze_fn = algo_module.dehaze

        # 获取模型路径
        model_dir = Path(algorithm_config.MODEL_PATH) / module_name
        model_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt"))
        model_path = str(model_files[0]) if model_files else ""

        logger.info(f"执行去雾: module={module_name}, model={model_path}")

        # 调用 dehaze 函数
        result = dehaze_fn(image_bytes, model_path)

        if isinstance(result, io.BytesIO):
            return result
        elif isinstance(result, PIL.Image.Image):
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            buf.seek(0)
            return buf
        else:
            raise ValueError(f"dehaze() 返回了不支持的类型: {type(result)}")

    async def _upload_result(self, result_bytes: io.BytesIO, algorithm_name: str) -> str:
        """上传结果到本地文件存储，由文件管理接口提供服务"""
        from datetime import datetime

        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{algorithm_name}_{int(time.time() * 1000)}.png"

        upload_dir = Path("upload/predictions") / date_str
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest_path = upload_dir / filename
        with open(dest_path, "wb") as f:
            f.write(result_bytes.getvalue())

        logger.info(f"预测结果已保存: {dest_path}")
        return f"/api/v1/files/download/predictions/{date_str}/{filename}"


# 单例
prediction_service = PredictionService()
