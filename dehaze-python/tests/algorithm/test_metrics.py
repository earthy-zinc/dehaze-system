"""图像质量指标计算（algorithm/metrics.py）边界口径测试。

不依赖 pyiqa：以假指标模型注入 _get_metric_model，验证指标筛选、
非有限值口径（全同图 PSNR=inf 钳制 100）与对抗性图像输入。

遵循 dehaze 测试规范：只断言业务结果，命名 test_功能_场景。
"""

import io
from types import SimpleNamespace

import pytest
from PIL import Image

from algorithm import metrics as metrics_module
from algorithm.metrics import METRICS_CONFIG, calculate

pytestmark = pytest.mark.api


def _png_bytes(color=128, size=(8, 8)) -> io.BytesIO:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def _install_fake_model(monkeypatch, values: dict):
    """values: {metric_name: 调用返回值}；模型调用签名兼容 FR/NR 两种形式"""

    def fake_get_model(name):
        def model(haze, clear=None):
            return _FakeResult(values[name])

        return model

    monkeypatch.setattr(metrics_module, "_get_metric_model", fake_get_model)


# ===== 指标筛选口径 =====


def test_calculate_without_clear_only_no_ref_metrics(monkeypatch):
    """未提供参考图：仅计算无参考指标（NIQE/NIMA/BRISQUE）"""
    _install_fake_model(monkeypatch, {name: 0.5 for name in METRICS_CONFIG})
    result = calculate(_png_bytes())

    labels = [item["label"] for item in result]
    assert labels == ["NIQE", "NIMA", "BRISQUE"]


def test_calculate_with_clear_computes_all_metrics(monkeypatch):
    """提供参考图：六项指标全部计算"""
    _install_fake_model(monkeypatch, {name: 0.5 for name in METRICS_CONFIG})
    result = calculate(_png_bytes(), _png_bytes())

    labels = [item["label"] for item in result]
    assert labels == ["PSNR", "SSIM", "LPIPS", "NIQE", "NIMA", "BRISQUE"]


def test_calculate_same_input_deterministic(monkeypatch):
    """不变量：相同图对重复计算结果完全一致（固定口径可复现）"""
    _install_fake_model(monkeypatch, {name: 0.42 for name in METRICS_CONFIG})
    first = calculate(_png_bytes(), _png_bytes())
    second = calculate(_png_bytes(), _png_bytes())

    assert first == second


# ===== 非有限值口径 =====


def test_calculate_identical_images_psnr_inf_clamped_to_100(monkeypatch):
    """全同图 MSE=0 → PSNR=inf：钳制到 100 dB（JSON 无法承载非有限值）"""
    _install_fake_model(
        monkeypatch,
        {name: (float("inf") if name == "psnr" else 0.5) for name in METRICS_CONFIG},
    )
    result = calculate(_png_bytes(), _png_bytes())

    psnr = next(item for item in result if item["label"] == "PSNR")
    assert psnr["value"] == 100.0


def test_calculate_non_finite_metric_raises(monkeypatch):
    """其他指标出现非有限值（如全黑图 NIQE=NaN）→ 计算失败，不落脏数据"""
    _install_fake_model(
        monkeypatch,
        {name: (float("nan") if name == "niqe" else 0.5) for name in METRICS_CONFIG},
    )

    with pytest.raises(ValueError, match="非有限值"):
        calculate(_png_bytes())


# ===== 尺寸与对抗性输入 =====


def test_calculate_size_mismatch_fails_fast(monkeypatch):
    """pred/gt 尺寸不一致：前置校验报错，而非 pyiqa 内部广播异常"""
    _install_fake_model(monkeypatch, {name: 0.5 for name in METRICS_CONFIG})

    with pytest.raises(ValueError, match="尺寸不一致"):
        calculate(_png_bytes(size=(8, 8)), _png_bytes(size=(16, 16)))


@pytest.mark.parametrize(
    "payload",
    [b"", b"not-an-image", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32],
    ids=["empty", "text-bytes", "truncated-png"],
)
def test_calculate_corrupted_image_bytes_raises(monkeypatch, payload):
    """0 字节/文本冒充/截断 PNG 等损坏输入 → 直接报错（任务置 failed）"""
    _install_fake_model(monkeypatch, {name: 0.5 for name in METRICS_CONFIG})

    with pytest.raises(Exception):
        calculate(io.BytesIO(payload))
