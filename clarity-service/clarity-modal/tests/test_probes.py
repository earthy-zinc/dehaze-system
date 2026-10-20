"""探针与指标端点。"""

from pathlib import Path

from app.config import settings


def test_ready_reports_device_and_weight_source(client, auth_headers):
    Path(settings.MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    response = client.get("/ready", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["device"] in {"cpu", "cuda:0"}
    assert body["weight_source"]["cache_dir"] == settings.MODEL_CACHE_DIR
    assert body["weight_source"]["base_url"].endswith("/models")


def test_ready_not_ready_when_cache_dir_missing(client, auth_headers):
    response = client.get("/ready", headers=auth_headers)

    assert response.status_code == 503
    assert response.json()["code"] == "SERVICE_UNAVAILABLE"


def test_health_does_not_require_model_load(client, auth_headers):
    assert client.get("/health", headers=auth_headers).json() == {
        "status": "ok",
        "service": settings.APP_NAME,
    }


def test_metrics_exposes_http_gpu_and_inference_metrics(client, auth_headers):
    client.get("/health", headers=auth_headers)

    text = client.get("/metrics", headers=auth_headers).text

    assert 'dehaze_http_requests_total{endpoint="/health",status="200"}' in text
    assert 'dehaze_http_request_duration_seconds_count{endpoint="/health"}' in text
    assert "dehaze_gpu_memory_used_bytes" in text
    assert "dehaze_inference_requests_total" in text
    assert "dehaze_inference_image_size_pixels" in text
