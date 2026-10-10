"""GET /v1/models：能力发现（算法清单 / 指标集清单 / 权重可用性）。"""

from pathlib import Path

from app.config import settings


def test_models_lists_algorithms_and_metric_sets(client, auth_headers):
    body = client.get("/v1/models", headers=auth_headers).json()

    algorithms = [item for item in body["data"] if item["type"] == "inference"]
    metrics = [item for item in body["data"] if item["type"] == "metrics"]

    assert len(algorithms) >= 29
    assert {"id", "type", "available", "capabilities", "limits"} <= algorithms[0].keys()
    assert "algorithm.DCP.run" in {item["id"] for item in algorithms}
    assert algorithms[0]["capabilities"]["task_types"] == ["dehazing"]
    assert algorithms[0]["limits"]["concurrency"] == settings.INFERENCE_THREAD_POOL_SIZE

    assert {item["id"] for item in metrics} == {"psnr", "ssim", "lpips", "niqe", "nima", "brisque"}
    psnr = next(item for item in metrics if item["id"] == "psnr")
    assert psnr["available"] is True
    assert psnr["capabilities"] == {
        "label": "PSNR",
        "better": "higher",
        "requires_clear": True,
        "description": psnr["capabilities"]["description"],
    }


def test_model_detail_returns_algorithm_item(client, auth_headers):
    body = client.get("/v1/models/algorithm.AECRNet.run", headers=auth_headers).json()

    assert body["id"] == "algorithm.AECRNet.run"
    assert body["type"] == "inference"
    assert body["available"] is True
    assert body["unavailable_reason"] is None


def test_model_detail_unknown_algorithm_returns_404(client, auth_headers):
    response = client.get("/v1/models/algorithm.NoSuchAlgorithm.run", headers=auth_headers)

    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"


def test_model_detail_weight_available_from_cache(client, auth_headers):
    cache_path = Path(settings.MODEL_CACHE_DIR) / "AECR-Net" / "NH_train.pk"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"x" * 128)

    body = client.get("/v1/models/AECR-Net/NH_train.pk", headers=auth_headers).json()

    assert body == {
        "id": "AECR-Net/NH_train.pk",
        "type": "inference",
        "available": True,
        "unavailable_reason": None,
        "size_bytes": 128,
    }


def test_model_detail_weight_unavailable(client, auth_headers):
    """本地无缓存且下载源不可达：返回可用性判定而非报错（供编排侧算法 CRUD 校验）"""
    body = client.get("/v1/models/Missing/no_such_weight.pth", headers=auth_headers).json()

    assert body["available"] is False
    assert body["size_bytes"] is None
    assert "不可用" in body["unavailable_reason"]
