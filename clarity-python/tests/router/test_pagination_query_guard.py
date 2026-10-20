"""分页参数下界约束守卫（三端一致性基线）。

分页下界是统一口径（python 400+A0400 / go `parsePagination` 400 / java `@Min(1)`）：
负值与零值必须拒绝，不得静默放行。约束有两条来源：
1. Query 模型继承 `BasePageQuery`（`pageNum ge=1` / `pageSize ge=1, le=100`）；
2. router 裸分页参数显式 `Query(default=..., ge=1)`。

本守卫用 AST 扫源码而非逐端点写用例：新增分页端点漏加约束即失败，避免
"新端点静默放行 pageNum=0"回流（与 `AgentConfigDefaults` 契约防漂移同法）。
"""

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models.schema.common import BasePageQuery

_ROUTER_DIR = Path(__file__).resolve().parents[2] / "app" / "router"
_PAGE_PARAMS = {"pageNum", "pageSize", "page_num", "page_size", "page", "size", "limit", "offset"}


class TestBasePageQueryBounds:
    @pytest.mark.parametrize(
        "kwargs", [{"pageNum": 0}, {"pageNum": -1}, {"pageSize": 0}, {"pageSize": -1}]
    )
    def test_non_positive_rejected(self, kwargs):
        with pytest.raises(ValidationError):
            BasePageQuery(**kwargs)

    def test_over_limit_rejected(self):
        with pytest.raises(ValidationError):
            BasePageQuery(pageSize=101)


def _is_int_annotation(node) -> bool:
    """int 或 int | None 注解（其余形态不参与本守卫，避免误报）。"""
    if isinstance(node, ast.Name):
        return node.id == "int"
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.BitOr)
        and _is_int_annotation(node.left)
    )


def _param_defaults(func: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, ast.expr | None]:
    args = func.args
    positional = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)
    mapping = dict(zip([a.arg for a in args.args], positional, strict=False))
    mapping.update({a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=False)})
    return mapping


def _has_ge_one(node) -> bool:
    """默认值为 `Query(...)` 且 ge 关键字是 >=1 的整数字面量。"""
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Name) or node.func.id != "Query":
        return False
    for kw in node.keywords:
        if kw.arg != "ge":
            continue
        value = kw.value
        return isinstance(value, ast.Constant) and isinstance(value.value, int) and value.value >= 1
    return False


def test_every_router_pagination_param_is_bounded():
    unguarded = []
    for path in sorted(_ROUTER_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            defaults = _param_defaults(node)
            for arg in [*node.args.args, *node.args.kwonlyargs]:
                if arg.arg not in _PAGE_PARAMS or not _is_int_annotation(arg.annotation):
                    continue
                if not _has_ge_one(defaults.get(arg.arg)):
                    unguarded.append(f"{path.name}:{node.lineno} 的 {arg.arg}")
    assert unguarded == [], f"分页参数缺少 ge=1 下界约束（负值/零值将放行）: {unguarded}"
