#!/usr/bin/env python3
"""批量将 service 层静态方法类转换为"实例方法 + 模块级单例"模式。

处理：
1. service 文件本身：删除 @staticmethod、方法签名加 self、类内自调用改 self.、
   确保文件末尾有模块级单例实例。
2. 所有调用方（app/ 与 tests/）：XxxService 全词替换为 xxx_service，
   定义文件中跳过自身的类名（保留 class 定义与单例实例化）。
"""
import pathlib
import re

ROOT = pathlib.Path("/home/earthyzinc/ProgramProject/dehaze-system/dehaze-python")

# 目标 service：路径 -> (类名, 单例名)。含已有单例实例但方法仍为 staticmethod 的
# balance/quota/search（脚本会为其去 staticmethod 加 self，单例实例已存在不重复添加）。
TARGETS = {
    "app/service/order_service.py": ("OrderService", "order_service"),
    "app/service/billing/billing_service.py": ("BillingService", "billing_service"),
    "app/service/billing/billing_record_service.py": ("BillingRecordService", "billing_record_service"),
    "app/service/billing/estimate_service.py": ("EstimateService", "estimate_service"),
    "app/service/billing/billing_stat_service.py": ("BillingStatService", "billing_stat_service"),
    "app/service/billing/refund_service.py": ("RefundService", "refund_service"),
    "app/service/billing/recharge_service.py": ("RechargeService", "recharge_service"),
    "app/service/billing/bill_service.py": ("BillService", "bill_service"),
    "app/service/billing/billing_anomaly_service.py": ("BillingAnomalyService", "billing_anomaly_service"),
    "app/service/billing/balance_service.py": ("BalanceService", "balance_service"),
    "app/service/billing/quota_service.py": ("QuotaService", "quota_service"),
    "app/service/voice/voice_billing_service.py": ("VoiceBillingService", "voice_billing_service"),
    "app/service/voice/tts_service.py": ("TtsService", "tts_service"),
    "app/service/voice/asr_service.py": ("AsrService", "asr_service"),
    "app/service/voice/hotword_service.py": ("HotwordService", "hotword_service"),
    "app/service/kb/document_service.py": ("DocumentService", "document_service"),
    "app/service/kb/knowledge_base_service.py": ("KnowledgeBaseService", "knowledge_base_service"),
    "app/service/kb/search_service.py": ("SearchService", "search_service"),
    "app/service/dataset/dataset_service.py": ("DatasetService", "dataset_service"),
    "app/service/dataset/item_file_service.py": ("ItemFileService", "item_file_service"),
    "app/service/dataset/dataset_item_service.py": ("DatasetItemService", "dataset_item_service"),
}


def rewrite_service_file(path: pathlib.Path, cls: str, inst: str) -> None:
    """改造单个 service 文件：去 @staticmethod、方法加 self、类内自调用改 self.、确保末尾单例。"""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    pending_staticmethod = False
    in_class = False
    for line in lines:
        stripped = line.strip()
        if stripped == "@staticmethod":
            pending_staticmethod = True
            continue
        if stripped.startswith("class "):
            in_class = re.match(rf"^class {cls}(\s*:|\(\s*$)", stripped) is not None
            out.append(line)
            continue
        if in_class:
            # 类内自调用 XxxService._xxx(...) -> self._xxx(...)
            if re.search(rf"\b{cls}\.\w+", line):
                line = re.sub(rf"\b{cls}\.", "self.", line)
            # 紧跟 @staticmethod 的方法定义补 self 参数
            m = re.match(r"^(    )(async )?(def \w+\()", line)
            if m and pending_staticmethod:
                params_rest = line[m.end():]
                if not re.match(r"^\s*(self|cls)\s*[,)]", params_rest):
                    line = f"{m.group(1)}{m.group(2)}{m.group(3)}self, {line[m.end():]}"
                pending_staticmethod = False
            elif not re.match(r"^    (async )?def ", line):
                pending_staticmethod = False
        out.append(line)
    result = "".join(out)
    if not re.search(rf"^{inst}\s*=\s*{cls}\(\)", result, re.M):
        if not result.endswith("\n"):
            result += "\n"
        result += f"\n\n{inst} = {cls}()\n"
    path.write_text(result, encoding="utf-8")


def rewrite_callers(cls: str, inst: str, skip_paths: set[str]) -> None:
    """在除 skip_paths 外的所有 .py 中把裸类名 XxxService 替换为单例名。"""
    pat = re.compile(rf"\b{cls}\b")
    for py in ROOT.rglob("*.py"):
        rel = py.relative_to(ROOT).as_posix()
        if rel in skip_paths or py.name == "scripts_refactor_service.py":
            continue
        text = py.read_text(encoding="utf-8")
        new = pat.sub(inst, text)
        if new != text:
            py.write_text(new, encoding="utf-8")
            print(f"    caller: {rel}")


def main() -> None:
    print("=== Step 1: rewrite service files ===")
    for rel, (cls, inst) in TARGETS.items():
        print(f"  service: {rel} ({cls} -> {inst})")
        rewrite_service_file(ROOT / rel, cls, inst)
    print("=== Step 2: rewrite callers ===")
    for rel, (cls, inst) in TARGETS.items():
        print(f"  class: {cls} -> {inst}")
        rewrite_callers(cls, inst, skip_paths={rel})


if __name__ == "__main__":
    main()
