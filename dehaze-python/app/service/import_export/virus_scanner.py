"""
病毒扫描接口（预留扩展点）

开发环境使用 NoOpVirusScanner（不扫描），生产环境可接入 ClamAV 等扫描器。
"""
from __future__ import annotations

import abc
from typing import Optional


class VirusScanner(abc.ABC):
    @abc.abstractmethod
    def scan(self, content: bytes) -> bool:
        """扫描文件内容，返回 True 表示安全"""

    @abc.abstractmethod
    def is_enabled(self) -> bool:
        """是否启用病毒扫描"""


class NoOpVirusScanner(VirusScanner):
    def scan(self, content: bytes) -> bool:
        return True

    def is_enabled(self) -> bool:
        return False


_scanner: Optional[VirusScanner] = None


def get_virus_scanner() -> VirusScanner:
    global _scanner
    if _scanner is None:
        _scanner = NoOpVirusScanner()
    return _scanner


def set_virus_scanner(scanner: VirusScanner) -> None:
    global _scanner
    _scanner = scanner
