"""对话计划（plan）wire 形状契约守卫（免库、免 Redis，源码级扫描）。

两条不变量：

1. **`PlanEvent` 全域不存在**（含 `PlanEventVO` 这类衍生名）。计划下发已收敛为 SSE `plan` 事件 +
   `plan_approve` 中断两种载荷，`PlanEvent` 类事件类型再现即回退。
2. **`to_wire_plan` 是唯一 wire 归一出口**。内部计划恒为 snake_case（Redis 中断点与执行器直接
   消费），只在 SSE 出口归一一次为 camelCase；出现第二个归一位置＝某条下发路径自己拼形状，
   两条路径会漂移，而三端各自单测仍全绿（静默契约破损）。

口径：本文件只做源码级扫描，无运行时断言；识别式各自带对照组自证（见各用例），避免负向断言
因检测式失效而恒绿。
"""

import ast
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"

FORBIDDEN_IDENTIFIER = "PlanEvent"

# 允许出现 `to_wire_plan` 的位置：(相对 app/ 的路径, 限定名)；`<module>`＝模块顶层（定义处）。
# 新增下发路径只能复用该出口；确实需要第二处时在此显式登记（登记动作本身就是评审点）。
ALLOWED_WIRE_EXITS: set[tuple[str, str]] = {
    ("service/ai/paradigms/plan_execute.py", "<module>"),
    ("infrastructure/sse/sse_event_converter.py", "SseEventConverter.record_plan"),
    ("infrastructure/sse/sse_event_converter.py", "SseEventConverter._handle_updates"),
}


def _iter_app_sources():
    for path in sorted(APP_DIR.rglob("*.py")):
        yield path.relative_to(APP_DIR).as_posix(), path.read_text(encoding="utf-8")


def _forbidden_identifier_lines(text: str) -> list[int]:
    """含禁用语串的行号（按子串匹配，故 `PlanEventVO` 等衍生名同样判出）。"""
    return [
        lineno for lineno, line in enumerate(text.splitlines(), 1) if FORBIDDEN_IDENTIFIER in line
    ]


def _wire_exit_sites_in(rel: str, text: str) -> set[tuple[str, str]]:
    """扫出单份源码里引用/定义 `to_wire_plan` 的位置：(相对路径, 限定名)。"""
    sites: set[tuple[str, str]] = set()

    def walk(node: ast.AST, stack: tuple[str, ...]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # 定义处归属其**父**作用域：模块顶层的 `def to_wire_plan` 记为 `<module>`
            if node.name == "to_wire_plan":
                sites.add((rel, ".".join(stack) or "<module>"))
            for child in node.body:
                walk(child, (*stack, node.name))
            return
        if isinstance(node, ast.Name):
            identifier = node.id
        elif isinstance(node, ast.Attribute):
            identifier = node.attr
        elif isinstance(node, ast.alias):
            identifier = node.name.rsplit(".", 1)[-1]
        else:
            identifier = None
        if identifier == "to_wire_plan":
            sites.add((rel, ".".join(stack) or "<module>"))
        for child in ast.iter_child_nodes(node):
            walk(child, stack)

    walk(ast.parse(text), ())
    return sites


def _wire_exit_sites() -> set[tuple[str, str]]:
    return {site for rel, text in _iter_app_sources() for site in _wire_exit_sites_in(rel, text)}


def test_forbidden_identifier_detector_is_effective():
    """检测式自证：含该串的内存样本必须判出，内部 snake_case 命名不得误伤。"""
    assert _forbidden_identifier_lines("class PlanEvent:\n    pass\n") == [1]
    assert _forbidden_identifier_lines('LEGACY = "PlanEventVO"\n') == [1]
    assert _forbidden_identifier_lines("plan_event = None\n") == []


def test_app_has_no_plan_event_identifier():
    """`app/` 全域不得再现 `PlanEvent`（注释/字符串里出现同样算回退）。"""
    hits = [
        f"{rel}:{lineno}"
        for rel, text in _iter_app_sources()
        for lineno in _forbidden_identifier_lines(text)
    ]
    assert not hits, "PlanEvent 已全域下线，再现即回退：\n" + "\n".join(hits)


def test_wire_exit_detector_is_effective():
    """检测式自证：新增出口与定义处都要判出（否则白名单对拍恒绿）。"""
    novel_call = (
        "from app.service.ai.paradigms import plan_execute\n"
        "\n"
        "class SomeConverter:\n"
        "    async def emit(self, plan):\n"
        "        return plan_execute.to_wire_plan(plan)\n"
    )
    assert _wire_exit_sites_in("x.py", novel_call) == {("x.py", "SomeConverter.emit")}
    assert _wire_exit_sites_in("p.py", "def to_wire_plan(plan):\n    return plan\n") == {
        ("p.py", "<module>")
    }


def test_to_wire_plan_is_the_only_wire_normalizer():
    """`to_wire_plan` 的位置必须与白名单逐条相等：新增出口或出口被删都红。"""
    sites = _wire_exit_sites()
    unexpected = sorted(sites - ALLOWED_WIRE_EXITS)
    missing = sorted(ALLOWED_WIRE_EXITS - sites)
    assert not unexpected, "出现白名单外的 wire 归一出口：\n" + "\n".join(
        f"{rel}::{qual}" for rel, qual in unexpected
    )
    assert not missing, "wire 归一出口丢失（下发路径不再经 to_wire_plan）：\n" + "\n".join(
        f"{rel}::{qual}" for rel, qual in missing
    )
