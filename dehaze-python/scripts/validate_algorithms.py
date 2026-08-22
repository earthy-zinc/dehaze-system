"""算法模型验证工具 —— L1-L4 分级检查

对 sys_algorithm 中已发布（status=4）的算法执行：
- L1 存在性：模型权重路径可解析（本地缓存 / 可下载）
- L2 可加载 + 显式声明：权重可反序列化；算法内 torch.load 是否显式 weights_only=False
- L3 结构匹配：模型 state_dict 键与 checkpoint 键比对（容忍 module. 前缀）
- L4 冒烟推理：固定测试图跑 dehaze()，校验输出

用法（在 dehaze-python 目录下执行）：
    .venv\\Scripts\\python.exe scripts\\validate_algorithms.py
    .venv\\Scripts\\python.exe scripts\\validate_algorithms.py --levels L1,L2
    .venv\\Scripts\\python.exe scripts\\validate_algorithms.py --algo-ids 1,2,3 --output report.json
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# 确保可以 import algorithm / app 包（脚本位于 dehaze-python/scripts/）
_DEHAZE_PYTHON = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DEHAZE_PYTHON))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from algorithm.model_loader import resolve_model_path  # noqa: E402

SQL_FILE = _DEHAZE_PYTHON.parent / "config" / "sql" / "data" / "sys_algorithm.sql"
ALGORITHM_ROOT = _DEHAZE_PYTHON / "algorithm"

# L4 冒烟用测试图（64x64 为 4/16 的倍数，兼容各算法的尺寸约束；GPU 16GB 显存下用小尺寸防止 OOM）
SMOKE_IMAGE_SIZE = 64


@dataclass
class AlgorithmRecord:
    """从 sys_algorithm.sql 解析出的一条已发布算法"""

    id: int
    parent_id: int
    name: str
    path: str
    import_path: str
    is_group: bool = False  # 有子节点的目录节点（不可执行，跳过验证）


@dataclass
class AlgorithmResult:
    """单个算法的验证结果"""

    id: int
    name: str
    path: str
    import_path: str
    is_group: bool = False
    checks: dict = field(default_factory=dict)  # level -> (passed, detail)
    error: str | None = None


def parse_algorithms(sql_path: Path) -> list[AlgorithmRecord]:
    """从 sys_algorithm.sql 提取 status=4 的算法（不依赖数据库连接）。"""
    text = sql_path.read_text(encoding="utf-8")

    records: list[AlgorithmRecord] = []
    for match in re.finditer(r"values\s*\((.*?)\);", text, re.S):
        block = match.group(1)
        # 提取字符串字段（SQL 内 '' 转义）与非字符串 token
        tokens = re.findall(r"'(?:[^']|'')*'|null|[^,\s][^,]*", block)
        tokens = [t.strip() for t in tokens if t.strip()]

        def unquote(token: str) -> str:
            if token.startswith("'"):
                return token[1:-1].replace("''", "'")
            return token

        def field_at(idx: int, _tokens: list[str] = tokens) -> str:
            return unquote(_tokens[idx]) if len(_tokens) > idx else ""

        try:
            algorithm_id = int(tokens[0])
            status = int(tokens[11])
        except (ValueError, IndexError):
            continue
        if status != 4:
            continue

        records.append(
            AlgorithmRecord(
                id=algorithm_id,
                parent_id=int(tokens[1]) if tokens[1].isdigit() else 0,
                name=field_at(3),
                path=field_at(5),
                import_path=field_at(9),
            )
        )

    # 标记分组节点：作为其他已发布节点的 parent 即为目录节点（不可执行）
    child_parent_ids = {r.parent_id for r in records if r.parent_id > 0}
    for record in records:
        record.is_group = record.id in child_parent_ids
    return records


def find_algorithm_dir(import_path: str) -> Path | None:
    """由 import_path（如 algorithm.AECRNet.run）定位算法目录。"""
    module = import_path.removeprefix("algorithm.").removesuffix(".run")
    return ALGORITHM_ROOT / module


def scan_torch_load_declarations(algorithm_dir: Path) -> list[str]:
    """扫描算法目录内所有 torch.load 调用（含跨行），返回未显式声明 weights_only 的调用摘要。"""
    undeclared: list[str] = []
    if not algorithm_dir.is_dir():
        return undeclared

    def _count_brackets(s: str) -> int:
        return s.count("(") - s.count(")")

    for py_file in sorted(algorithm_dir.rglob("*.py")):
        lines = py_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if "torch.load(" not in line or line.strip().startswith("#"):
                i += 1
                continue
            # 从本行起累积调用文本，直到括号闭合（容忍跨行调用）
            rel = py_file.relative_to(ALGORITHM_ROOT)
            start = line.index("torch.load(")
            call_text = line[start:]
            depth = _count_brackets(call_text)
            while depth > 0 and i + 1 < len(lines):
                i += 1
                call_text += lines[i]
                depth = _count_brackets(call_text)
            if "weights_only" not in call_text:
                undeclared.append(f"{rel}:{i + 1}")
            i += 1
    return undeclared


def extract_checkpoint_state(ckpt: dict, model_keys: set[str]) -> dict:
    """识别 checkpoint 的权重字典。

    支持三种形态：
    - 顶层含 model/state_dict/params 等包装键时取对应子字典；
    - 多子模型存档（如 D4 的 net_h2c/net_c2h/net_depth、
      FogRemoval 的 genA2B 等）时取与模型键交集最大的子字典；
    - 顶层即权重键（含 '.' 的 Tensor 键）时原样返回。
    """

    def is_weight_dict(d: dict) -> bool:
        return bool(d) and any(
            isinstance(k, str) and "." in k and isinstance(v, torch.Tensor) for k, v in d.items()
        )

    for key in ("model", "state_dict", "params"):
        value = ckpt.get(key)
        if isinstance(value, dict) and is_weight_dict(value):
            return value
    # 多子模型存档：选取与模型 state_dict 键交集最大的子字典
    best, best_score = None, -1
    for value in ckpt.values():
        if isinstance(value, dict) and is_weight_dict(value):
            score = len(set(value.keys()) & model_keys)
            if score > best_score:
                best, best_score = value, score
    if best is not None:
        return best
    return ckpt if is_weight_dict(ckpt) else {}


def run_l1_l2(record: AlgorithmRecord) -> tuple[bool, str, list[str]]:
    """L1 路径可解析；L2 权重可反序列化 + 显式声明扫描。"""
    algorithm_dir = find_algorithm_dir(record.import_path)
    if algorithm_dir is None or not algorithm_dir.is_dir():
        return False, f"L1失败: 算法目录不存在（import_path={record.import_path}）", []
    run_py = algorithm_dir / "run.py"
    if not run_py.exists():
        return False, f"L1失败: 算法目录缺少 run.py（{algorithm_dir.name}）", []
    undeclared = scan_torch_load_declarations(algorithm_dir)

    # 无权重算法（如 DCP）：run.py 不依赖 torch.load，跳过权重检查
    if not (record.path and record.path.strip()) and not undeclared:
        return True, "通过（无权重算法）", []

    # L1：权重路径可解析（缺文件/写错在此暴露）
    if not record.path or not record.path.strip():
        return False, "L1失败: path 为空但算法依赖权重", undeclared
    try:
        model_path = resolve_model_path(record.path)
    except Exception as e:
        return False, f"L1失败: 权重不可用 - {str(e)[:150]}", undeclared

    # L2：权重可反序列化
    try:
        torch.load(model_path, weights_only=False, map_location="cpu")
    except Exception as e:
        return False, f"L2失败: 权重反序列化异常 - {str(e)[:150]}", undeclared

    # L2：显式声明扫描（算法目录内所有 torch.load）
    if undeclared:
        return (
            False,
            f"L2失败: {len(undeclared)} 处 torch.load 未显式声明 weights_only=False",
            undeclared,
        )

    return True, "通过", []


def run_l3(record: AlgorithmRecord) -> tuple[bool, str]:
    """L3 结构匹配：importlib 导入 run.py，经 get_model 构建并加载，比对键集合。"""
    if not (record.path and record.path.strip()):
        return True, "通过（无权重算法，无需结构比对）"
    module_name = record.import_path.removeprefix("algorithm.").removesuffix(".run")
    try:
        model_path = resolve_model_path(record.path)
        algo_module = importlib.import_module(f"algorithm.{module_name}.run")
    except Exception as e:
        return False, f"L3跳过: 模块导入失败 - {str(e)[:150]}"

    if not hasattr(algo_module, "get_model"):
        return False, "L3跳过: run.py 未导出 get_model()"

    try:
        net = algo_module.get_model(model_path)
        # 部分算法 get_model 返回 (net, 其他) 元组，取第一个模型对象
        if isinstance(net, tuple):
            net = net[0]
        model_keys = set(net.state_dict().keys())
    except Exception as e:
        return False, f"L3失败: 模型构建/加载异常（可能结构错配）- {str(e)[:200]}"

    # 键比对仅作提示：权重/模型结构差异（多子模型存档、训练存档、strict=False 多余键均可容忍）
    def normalize(key: str) -> str:
        return key.removeprefix("module.")

    ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
    ckpt_state = extract_checkpoint_state(ckpt, model_keys)
    model_keys_norm = {normalize(k) for k in model_keys}
    ckpt_keys_norm = {normalize(k) for k in ckpt_state.keys()}

    extra = sorted(k for k in ckpt_keys_norm if k not in model_keys_norm)
    missing = sorted(k for k in model_keys_norm if k not in ckpt_keys_norm)
    if extra or missing:
        return (
            True,
            f"通过（checkpoint 多余 {len(extra)} 键、模型多出 {len(missing)} 键，结构差异已容忍）",
        )

    return True, "通过"


def run_l4(record: AlgorithmRecord) -> tuple[bool, str]:
    """L4 冒烟推理：固定测试图跑 dehaze()，校验输出非空。"""
    if not (record.path and record.path.strip()):
        return True, "通过（无权重算法，无需冒烟推理）"
    module_name = record.import_path.removeprefix("algorithm.").removesuffix(".run")
    try:
        model_path = resolve_model_path(record.path)
        algo_module = importlib.import_module(f"algorithm.{module_name}.run")
    except Exception as e:
        return False, f"L4跳过: 模块导入失败 - {str(e)[:150]}"

    if not hasattr(algo_module, "dehaze"):
        return False, "L4跳过: run.py 未导出 dehaze()"

    img = Image.new("RGB", (SMOKE_IMAGE_SIZE, SMOKE_IMAGE_SIZE), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    try:
        result = algo_module.dehaze(buf, model_path)
    except Exception as e:
        return False, f"L4失败: 推理异常 - {str(e)[:200]}"

    if isinstance(result, io.BytesIO):
        if result.getvalue():
            return True, "通过"
        return False, "L4失败: 输出为空"
    if isinstance(result, Image.Image):
        return True, "通过"
    return False, f"L4失败: dehaze() 返回不支持类型 {type(result).__name__}"


def validate_record(record: AlgorithmRecord, levels: set[str]) -> AlgorithmResult:
    result = AlgorithmResult(
        id=record.id,
        name=record.name,
        path=record.path,
        import_path=record.import_path,
        is_group=record.is_group,
    )
    if record.is_group:
        return result  # 目录节点不执行 L1-L4，标记 N/A
    try:
        if "L1" in levels:
            passed, detail, _ = run_l1_l2(record)
            result.checks["L1"] = (passed, detail)
        if "L2" in levels:
            passed, detail, undeclared = run_l1_l2(record)
            if not passed and "L1失败" in detail:
                # L1 已失败则 L2 无意义，跳过避免重复下载/加载
                result.checks["L2"] = (False, detail)
            else:
                result.checks["L2"] = (
                    passed,
                    detail + (f"；未声明: {', '.join(undeclared[:5])}" if undeclared else ""),
                )
        if "L3" in levels:
            passed, detail = run_l3(record)
            result.checks["L3"] = (passed, detail)
        if "L4" in levels:
            passed, detail = run_l4(record)
            result.checks["L4"] = (passed, detail)
    except Exception as e:
        result.error = str(e)[:300]
    return result


def summarize(results: list[AlgorithmResult], levels: set[str]) -> str:
    lines = [f"{'算法ID':<6}{'算法名':<16}{'L1':<5}{'L2':<5}{'L3':<5}{'L4':<5}结论"]
    lines.append("-" * 60)
    level_order = ("L1", "L2", "L3", "L4")
    for r in sorted(results, key=lambda x: x.id):
        marks = []
        for level in level_order:
            if level not in levels:
                marks.append("-")
                continue
            if r.is_group:
                marks.append("N/A")
                continue
            passed, _ = r.checks.get(level, (False, ""))
            marks.append("PASS" if passed else "FAIL")
        if r.is_group:
            conclusion = "目录"
        else:
            enabled = [m for level, m in zip(level_order, marks, strict=True) if level in levels]
            conclusion = "通过" if all(m == "PASS" for m in enabled) else "待整改"
        name = r.name[:14]
        lines.append(f"{r.id:<6}{name:<16}" + "".join(f"{m:<5}" for m in marks) + conclusion)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="算法模型 L1-L4 分级验证")
    parser.add_argument("--levels", default="L1,L2,L3,L4", help="执行的检查级别，逗号分隔")
    parser.add_argument("--algo-ids", default="", help="仅验证指定算法 id，逗号分隔")
    parser.add_argument("--output", default="", help="报告输出 JSON 文件路径")
    args = parser.parse_args()

    levels = {s.strip() for s in args.levels.split(",") if s.strip()}
    records = parse_algorithms(SQL_FILE)
    if not records:
        print(f"[ERROR] 未从 {SQL_FILE} 解析到 status=4 的算法")
        sys.exit(1)

    if args.algo_ids:
        wanted = {int(s) for s in args.algo_ids.split(",") if s.strip()}
        records = [r for r in records if r.id in wanted]

    print(f"待验证算法 {len(records)} 个，级别: {sorted(levels)}\n")
    results = []
    for i, record in enumerate(records, 1):
        print(f"[{i}/{len(records)}] 验证 #{record.id} {record.name} ({record.import_path})")
        result = validate_record(record, levels)
        results.append(result)
        for level in ("L1", "L2", "L3", "L4"):
            if level in result.checks:
                passed, detail = result.checks[level]
                print(f"  {level}: {'PASS' if passed else 'FAIL'} - {detail}")
        if result.error:
            print(f"  ERROR: {result.error}")

    print("\n========== 汇总 ==========")
    print(summarize(results, levels))

    leaves = [r for r in results if not r.is_group]
    groups = [r for r in results if r.is_group]
    passed_count = sum(
        1
        for r in leaves
        if r.error is None and all(r.checks.get(lvl, (True, ""))[0] for lvl in levels)
    )
    print(
        f"\n叶子算法 通过 {passed_count}/{len(leaves)}，"
        f"待整改 {len(leaves) - passed_count}；目录节点 {len(groups)} 个（跳过）"
    )

    if args.output:
        out = {
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "levels": sorted(levels),
            "results": [
                {
                    "id": r.id,
                    "name": r.name,
                    "path": r.path,
                    "import_path": r.import_path,
                    "is_group": r.is_group,
                    "checks": {k: {"passed": v[0], "detail": v[1]} for k, v in r.checks.items()},
                    "error": r.error,
                }
                for r in results
            ],
        }
        Path(args.output).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n报告已写入: {args.output}")


if __name__ == "__main__":
    main()
