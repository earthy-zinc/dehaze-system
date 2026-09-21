"""PII（个人敏感信息）检测与脱敏工具。

统一承载身份证/手机号/银行卡/密码/API 密钥等敏感信息的正则与处理逻辑，
供记忆写入过滤（memory_extraction）与护栏脱敏（guardrail_middleware）复用，
避免多端重复实现同一套正则。
"""

import re

# 身份证（18 位）；用环视界定边界而非 \b，避免紧贴中文（\w 含 CJK）时漏匹配
_ID_CARD_RE = re.compile(r"(?<![\dXx])\d{17}[\dXx](?![\dXx])")
# 大陆手机号
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
# API 密钥（sk-/pk-/ak- 前缀 + 较长凭据串）
_SECRET_RE = re.compile(
    r"(?<![A-Za-z0-9_\-])(sk|pk|ak)-[A-Za-z0-9_\-]{8,}(?![A-Za-z0-9_\-])",
    re.IGNORECASE,
)
# 银行卡号（13-19 位连续数字）
_BANK_CARD_RE = re.compile(r"(?<!\d)\d{13,19}(?!\d)")
# 密码类（"密码/口令/password=" 后跟随的敏感值，宽泛匹配常见写法）
_PASSWORD_RE = re.compile(
    r"(?:密码|口令|password)\s*[:：=]\s*[\S]{4,}",
    re.IGNORECASE,
)


def mask_pii(text: str) -> str:
    """对身份证号、手机号、银行卡、API 密钥等 PII 与凭据做正则脱敏。

    脱敏值统一替换为 ``***``，保留脱敏后的非敏感上下文。
    """
    masked = _ID_CARD_RE.sub("***", text)
    masked = _PHONE_RE.sub("***", masked)
    masked = _BANK_CARD_RE.sub("***", masked)
    masked = _PASSWORD_RE.sub("密码：***", masked)
    return _SECRET_RE.sub("***", masked)


# 尾部可能被 chunk 边界切断的敏感串：纯数字串（手机/银行卡/身份证）或密钥前缀串
_CANDIDATE_TAIL_RE = re.compile(r"[0-9A-Za-z_\-]+$")
_SECRET_PREFIX_RE = re.compile(r"(?:sk|pk|ak)-", re.IGNORECASE)
# 尾部是密码类关键词（值尚未到达，整段都不能外发）
_PASSWORD_PREFIX_RE = re.compile(r"(?:密码|口令|password)\s*[:：=]?\s*$", re.IGNORECASE)


class StreamingPiiMasker:
    """流式文本的 PII 脱敏。

    敏感串可能被 chunk 边界切断（手机号分两次到达），逐块独立脱敏会漏判。故尾部
    保留"可能是敏感串前缀"的一段不外发，待后续 chunk 补全后再判定；流结束时
    flush 输出残余。
    """

    def __init__(self) -> None:
        self._pending = ""

    def push(self, text: str) -> str:
        """追加增量文本，返回可安全外发的脱敏文本（可能为 ""）。"""
        self._pending += text
        masked = mask_pii(self._pending)
        hold = self._hold_len(masked)
        if not hold:
            self._pending = ""
            return masked
        if len(masked) <= hold:
            return ""
        emit, self._pending = masked[:-hold], masked[-hold:]
        return emit

    def flush(self) -> str:
        """流结束：输出残余文本。"""
        text, self._pending = mask_pii(self._pending), ""
        return text

    @staticmethod
    def _hold_len(masked: str) -> int:
        if _PASSWORD_PREFIX_RE.search(masked):
            return len(masked)
        tail = _CANDIDATE_TAIL_RE.search(masked)
        if not tail:
            return 0
        run = tail.group()
        # 普通文本（含字母数字混排的单词）不暂存，否则正常输出会被逐词压住
        if run.isdigit() or _SECRET_PREFIX_RE.match(run):
            return len(run)
        return 0


def contains_pii(text: str) -> bool:
    """判断文本是否命中任意 PII 规则（用于写入前过滤判定）。"""
    return bool(
        _ID_CARD_RE.search(text)
        or _PHONE_RE.search(text)
        or _BANK_CARD_RE.search(text)
        or _PASSWORD_RE.search(text)
        or _SECRET_RE.search(text)
    )
