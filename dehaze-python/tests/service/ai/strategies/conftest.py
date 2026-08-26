"""AI 策略域测试 fixtures

共享桩类在 `stubs.py`（测试文件需要直接实例化时 import 它）；
此处仅提供 fixture 形态。仅 `test_agent_config_resolver.py` 使用，勿扩散。
"""

import pytest

from tests.stubs.fakes import StubAsyncSession, StubInterruptHandler


@pytest.fixture
def stub_db():
    """标准 AsyncSession 桩（见 stubs.StubAsyncSession 文档）。"""
    return StubAsyncSession()


@pytest.fixture
def stub_ih():
    """无挂起中断的 interrupt_handler 桩（no-op）。"""
    return StubInterruptHandler()
