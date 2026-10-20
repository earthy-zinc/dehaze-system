"""`POST /v1/images/edits` 与模型目录（app.catalog）：slug 解析、OpenAI 风格响应、错误码。"""

import base64
import json
import time

from app import catalog, executor
from app.config import settings
from app.errors import ModalError


def _post(client, headers, model, image_bytes, params=None):
    data = {"model": model}
    if params is not None:
        data["params"] = params
    return client.post(
        "/v1/images/edits",
        headers=headers,
        data=data,
        files={"image": ("input.png", image_bytes, "image/png")},
    )


def test_catalog_seed_covers_all_slugs():
    """种子含阶段 1 迁入的 85 个规范 slug（抽样断言防种子被误删/误改）"""
    items = catalog.artifacts()
    for slug in ("aecrnet", "aodnet", "c2pnet-its", "c2pnet-ots", "d4", "dcpdn"):
        assert slug in items, f"种子缺少统一模型 slug: {slug}"
        assert items[slug]["import_path"].startswith("algorithm.")
        assert items[slug]["weights_path"]


def test_resolve_edit_rejects_unknown_and_disabled():
    import pytest

    with pytest.raises(ModalError) as exc_info:
        catalog.resolve_edit("no-such-model")
    assert exc_info.value.code == "MODEL_NOT_FOUND"

    disabled = next(s for s, i in catalog.artifacts().items() if not i["enabled"])
    with pytest.raises(ModalError) as exc_info:
        catalog.resolve_edit(disabled)
    assert exc_info.value.code == "MODEL_DISABLED"


def test_images_edits_returns_openai_style_b64(client, auth_headers, png_bytes, monkeypatch):
    """正常路径：slug 解析 → 推理 → {created, data:[{b64_json}]}，b64 可还原为图像"""
    captured = {}

    async def _fake_run(model, import_path, weights_path, file_obj, image_size):
        captured["import_path"] = import_path
        captured["weights"] = weights_path
        return b"fake-png-bytes"

    monkeypatch.setattr("app.routers.images.run_inference", _fake_run)

    response = _post(client, auth_headers, "aecrnet", png_bytes())

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"created", "data"}
    assert len(payload["data"]) == 1
    assert base64.b64decode(payload["data"][0]["b64_json"]) == b"fake-png-bytes"
    # 运行工件由服务端目录解析（编排侧未下发 import_path/weights）
    assert captured["import_path"] == catalog.artifacts()["aecrnet"]["import_path"]
    assert captured["weights"] == catalog.artifacts()["aecrnet"]["weights_path"]


def test_images_edits_unknown_model_404(client, auth_headers, png_bytes):
    response = _post(client, auth_headers, "no-such-model", png_bytes())
    assert response.status_code == 404
    assert response.json()["code"] == "MODEL_NOT_FOUND"


def test_images_edits_rejects_bad_params_json(client, auth_headers, png_bytes):
    response = _post(client, auth_headers, "aecrnet", png_bytes(), params="not-json{")
    assert response.status_code == 400


def test_images_edits_requires_auth(client, png_bytes):
    response = _post(client, {}, "aecrnet", png_bytes())
    assert response.status_code == 401


def test_models_lists_image_edit_slugs(client, auth_headers):
    """/v1/models 的 image_edit 项以规范 slug 为 id（模型清单事实源）"""
    response = client.get("/v1/models", headers=auth_headers)
    assert response.status_code == 200
    edit_items = [i for i in response.json()["data"] if i.get("type") == "image_edit"]
    assert {i["id"] for i in edit_items} >= {"aecrnet", "aodnet", "c2pnet-its"}
    aecrnet = next(i for i in edit_items if i["id"] == "aecrnet")
    assert aecrnet["artifacts"]["import_path"] == "algorithm.AECRNet.run"
    assert aecrnet["owned_by"] == "local-modal"


# ---------- 执行链路错误码（原 /v1/inference 用例迁移至统一 slug 入口） ----------


def test_images_edits_accepts_optional_params(client, auth_headers, png_bytes, monkeypatch):
    async def _fake_run(model, import_path, weights_path, file_obj, image_size):
        return b"fake-png-bytes"

    monkeypatch.setattr("app.routers.images.run_inference", _fake_run)

    response = _post(
        client, auth_headers, "aecrnet", png_bytes(), params=json.dumps({"strength": 0.8})
    )

    assert response.status_code == 200


def test_images_edits_missing_image_returns_400(client, auth_headers):
    response = client.post("/v1/images/edits", headers=auth_headers, data={"model": "aecrnet"})

    assert response.status_code == 400
    assert response.json()["code"] == "PARAM_ERROR"


def test_images_edits_invalid_image_content_returns_422(client, auth_headers):
    response = _post(client, auth_headers, "aecrnet", b"not-an-image")

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_IMAGE"


def test_images_edits_unavailable_weights_returns_503(client, auth_headers, png_bytes, monkeypatch):
    """缓存未命中且下载源不可达 → 权重不可用（503），不降级到空权重"""
    monkeypatch.setattr(
        catalog,
        "resolve_edit",
        lambda _m: {
            "import_path": "algorithm.AECRNet.run",
            "weights_path": "NoSuchAlgorithm/no_such_weights.pth",
            "name": "AECR-NET",
            "enabled": True,
        },
    )

    response = _post(client, auth_headers, "aecrnet", png_bytes())

    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_UNAVAILABLE"


def test_images_edits_timeout_returns_504(client, auth_headers, png_bytes, monkeypatch):
    def _slow_execute(*_args):
        time.sleep(0.3)
        return b"never-returned"

    monkeypatch.setattr(settings, "INFERENCE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(executor, "_execute", _slow_execute)

    response = _post(client, auth_headers, "aecrnet", png_bytes())

    assert response.status_code == 504
    assert response.json()["code"] == "INFERENCE_TIMEOUT"


def test_images_edits_algorithm_exception_returns_inference_failed(
    client, auth_headers, png_bytes, monkeypatch
):
    """算法内部异常被包装为 500 INFERENCE_FAILED，且只回摘要不回堆栈"""
    from algorithm.DCP import run as dcp_run

    monkeypatch.setattr(
        catalog,
        "resolve_edit",
        lambda _m: {
            "import_path": "algorithm.DCP.run",
            "weights_path": "",
            "name": "DCP",
            "enabled": True,
        },
    )

    def _boom(image, model_path):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(dcp_run, "dehaze", _boom)

    response = _post(client, auth_headers, "dcp", png_bytes())

    assert response.status_code == 500
    assert response.json()["code"] == "INFERENCE_FAILED"
    assert "algorithm.DCP.run" in response.json()["message"]
    assert "CUDA out of memory" in response.json()["message"]


def test_images_edits_unexpected_exception_returns_500_without_leaking(
    client, auth_headers, png_bytes, monkeypatch
):
    """线程内未包装的异常由全局处理器兜底：响应固定文案，异常原文只进日志"""

    def _boom(*_args):
        raise RuntimeError("/srv/secrets/path not readable")

    monkeypatch.setattr(executor, "_execute", _boom)

    response = _post(client, auth_headers, "aecrnet", png_bytes())

    assert response.status_code == 500
    assert response.json() == {"code": "INTERNAL_ERROR", "message": "内部错误"}
