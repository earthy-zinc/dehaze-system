"""pytest 共享 fixtures。"""
from __future__ import annotations

import sys
from pathlib import Path

# 让 tests/ 也能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from utils import auth, api, config, cleanup


@pytest.fixture(params=list(config.BACKENDS.keys()))
def backend(request) -> str:
    """参数化后端：java / go / python。"""
    return request.param


@pytest.fixture
def session(backend):
    """每个后端登录一次，测试结束自动登出。"""
    sid = auth.login(backend=backend)
    yield sid
    try:
        auth.logout(backend=backend)
    except Exception:
        pass


@pytest.fixture(scope="session", autouse=True)
def _cleanup_after_session():
    """所有测试跑完后清理限流缓存（避免影响下次跑）。"""
    yield
    cleanup.clear_login_rate_limit()
    api.close()
