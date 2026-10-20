"""POST /v1/metrics：指标逐项隔离失败与请求校验。

不依赖 pyiqa 下载指标权重：以假指标模型注入 _get_metric_model，验证隔离口径与
非有限值口径（全同图 PSNR=inf 钳制 100）。
"""

import json

import pytest

from algorithm import metrics as metrics_module


class _Scalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakeMetricModel:
    def __init__(self, value):
        self._value = value

    def __call__(self, *args):
        if self._value is None:
            raise RuntimeError("indicator model failed")
        return _Scalar(self._value)


@pytest.fixture
def fake_metric_models(monkeypatch):
    def _install(values: dict[str, float | None]):
        monkeypatch.setattr(
            metrics_module,
            "_get_metric_model",
            lambda name: _FakeMetricModel(values.get(name, 0.5)),
        )

    return _install


def _post(client, headers, image_bytes, names, reference=None, model="指标集A"):
    files = {"image": ("result.png", image_bytes, "image/png")}
    if reference is not None:
        files["reference"] = ("gt.png", reference, "image/png")
    return client.post(
        "/v1/metrics",
        headers=headers,
        data={"model": model, "metrics": json.dumps(names)},
        files=files,
    )


def test_metrics_isolates_single_metric_failure(
    client, auth_headers, png_bytes, fake_metric_models
):
    fake_metric_models({"psnr": 31.2, "ssim": 0.86, "niqe": None})

    response = _post(
        client, auth_headers, png_bytes(), ["psnr", "ssim", "niqe"], reference=png_bytes()
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metrics"] == {"psnr": 31.2, "ssim": 0.86}
    assert body["failed"] == ["niqe"]


def test_metrics_skips_reference_metric_without_reference(
    client, auth_headers, png_bytes, fake_metric_models
):
    """缺参考图时参考类指标进 failed，无参考指标照常返回"""
    fake_metric_models({"psnr": 31.2, "niqe": 3.4})

    response = _post(client, auth_headers, png_bytes(), ["psnr", "niqe"])

    assert response.status_code == 200
    assert response.json()["metrics"] == {"niqe": 3.4}
    assert response.json()["failed"] == ["psnr"]


def test_metrics_reports_image_size_meta(client, auth_headers, png_bytes, fake_metric_models):
    fake_metric_models({"niqe": 3.4})

    response = _post(client, auth_headers, png_bytes(size=(32, 48)), ["niqe"])

    assert response.json()["meta"] == {"image_size": 32 * 48, "width": 32, "height": 48}


def test_metrics_clamps_infinite_psnr(client, auth_headers, png_bytes, fake_metric_models):
    fake_metric_models({"psnr": float("inf")})

    response = _post(client, auth_headers, png_bytes(), ["psnr"], reference=png_bytes())

    assert response.json() == {
        "metrics": {"psnr": 100.0},
        "failed": [],
        "meta": {"image_size": 1024, "width": 32, "height": 32},
    }


def test_metrics_isolates_non_finite_value(client, auth_headers, png_bytes, fake_metric_models):
    fake_metric_models({"niqe": float("nan")})

    response = _post(client, auth_headers, png_bytes(), ["niqe"])

    assert response.json()["metrics"] == {}
    assert response.json()["failed"] == ["niqe"]


def test_metrics_unknown_name_returns_400(client, auth_headers, png_bytes, fake_metric_models):
    fake_metric_models({})

    response = _post(client, auth_headers, png_bytes(), ["psnr", "entropy"])

    assert response.status_code == 400
    assert response.json()["code"] == "PARAM_ERROR"


def test_metrics_size_mismatch_returns_400(client, auth_headers, png_bytes, fake_metric_models):
    fake_metric_models({})

    response = _post(
        client,
        auth_headers,
        png_bytes(size=(32, 32)),
        ["psnr"],
        reference=png_bytes(size=(16, 16)),
    )

    assert response.status_code == 400
    assert "尺寸不一致" in response.json()["message"]


def test_metrics_invalid_metrics_json_returns_400(client, auth_headers, png_bytes):
    response = client.post(
        "/v1/metrics",
        headers=auth_headers,
        data={"model": "指标集A", "metrics": "psnr"},
        files={"image": ("result.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "PARAM_ERROR"


def test_metrics_empty_metrics_returns_400(client, auth_headers, png_bytes):
    response = _post(client, auth_headers, png_bytes(), [])

    assert response.status_code == 400
    assert response.json()["code"] == "PARAM_ERROR"


def test_metrics_invalid_image_returns_422(client, auth_headers):
    response = _post(client, auth_headers, b"not-an-image", ["niqe"])

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_IMAGE"
