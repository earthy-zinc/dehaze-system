"""分页缺省值跨端对拍（免库、免 Redis，源码级）。

背景：跨端口径是「默认值逐端点对 python」——python 各端点缺省 pageSize 为：计费 20、可观测 10、
非 AI 域统一 10（favorite / 数据集数据项等 5 个端点为 20）。缺省值不一致的表现不是报错，而是
"同一页请求换端后条数/分页语义不同"（java 缺省 10、go 缺省 20 时前端换端即跳页），静默且各端
单测各自全绿，故用本表钉死。

口径：

- 表中 `expected` 是跨端锚定值（以 python 实现为事实源逐端点核对），**三端都必须等于它**：
  任一端单独漂移即红；改口径须三端同改并同步此表。
- python 侧的值由 AST 从源码提取（router 形参 `Query(default=N)`、继承 `BasePageQuery` 的
  schema 模型），不在此抄死；java 侧按「查询对象字段 / 实例初始化块 / Controller
  `@RequestParam` 默认值」三种形态提取；go 侧按 handler 内 `parsePagination*` 的缺省实参提取。
- 覆盖范围＝本表条目；go 侧另有 `internal/api/pagination_sweep_test.go` 按端点逐个核对 python
  行号，本表不重复枚举，仅覆盖跨端口径最容易漂移的几类。
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PY_APP = REPO_ROOT / "clarity-python" / "app"
JAVA_SRC = REPO_ROOT / "clarity-java" / "src" / "main" / "java"
GO_SRC = REPO_ROOT / "clarity-go"

_JAVA_FIELD_PATTERN = r"pageSize\s*=\s*(\d+)\s*;"
_JAVA_INIT_PATTERN = r"setPageSize\((\d+)\)"
_JAVA_REQUEST_PARAM_PATTERN = r'defaultValue = "(\d+)"\)\s*[Ii]nt(?:eger)? pageSize'


@dataclass(frozen=True)
class PageDefaultAnchor:
    """一条跨端分页缺省值锚点；`py_router` / `py_schema` 二选一（python 侧声明形态）。"""

    endpoint: str
    expected: int
    py_router: tuple[str, str] | None = None  # (相对 app/ 的 router 路径, 端点函数名)
    py_schema: str | None = None  # schema 模型类名（未覆写则继承 BasePageQuery）
    java: tuple[str, str] | None = None  # (相对 java src/main/java 的路径, 抽取正则)
    go: tuple[str, str] | None = None  # (相对 clarity-go 的路径, handler 函数名)


ANCHORS: tuple[PageDefaultAnchor, ...] = (
    PageDefaultAnchor(
        "计费.billing.listRecords",
        20,
        py_router=("router/ai_billing.py", "list_records"),
        java=("com/pei/clarity/model/query/AiBillingRecordQuery.java", _JAVA_FIELD_PATTERN),
        go=("internal/api/ai_billing.go", "ListRecords"),
    ),
    PageDefaultAnchor(
        "计费.billing.listCreditLogs",
        20,
        py_router=("router/ai_billing.py", "list_credit_logs"),
        java=("com/pei/clarity/model/query/AiCreditLogQuery.java", _JAVA_FIELD_PATTERN),
        go=("internal/api/ai_billing.go", "ListCreditLogs"),
    ),
    PageDefaultAnchor(
        "计费.billing.listRefunds",
        20,
        py_router=("router/ai_billing.py", "list_refunds"),
        java=("com/pei/clarity/model/query/AiRefundQuery.java", _JAVA_FIELD_PATTERN),
        go=("internal/api/ai_billing.go", "ListRefunds"),
    ),
    PageDefaultAnchor(
        "计费.billing.listAnomalies",
        20,
        py_router=("router/ai_billing.py", "list_anomalies"),
        java=("com/pei/clarity/model/query/AiBillingAnomalyQuery.java", _JAVA_FIELD_PATTERN),
        go=("internal/api/ai_billing.go", "ListAnomalies"),
    ),
    PageDefaultAnchor(
        "计费.billing.listCosts",
        20,
        py_router=("router/ai_billing.py", "list_costs"),
        java=("com/pei/clarity/model/query/AiModelCostQuery.java", _JAVA_FIELD_PATTERN),
        go=("internal/api/ai_billing.go", "ListCosts"),
    ),
    PageDefaultAnchor(
        "可观测.observability.listTraces",
        10,
        py_schema="TracePageQuery",
        java=("com/pei/clarity/model/query/AiObservabilityTraceQuery.java", _JAVA_FIELD_PATTERN),
        go=("internal/api/ai_observability.go", "ListTraces"),
    ),
    PageDefaultAnchor(
        "可观测.observability.getCosts",
        10,
        py_schema="CostsQuery",
        java=("com/pei/clarity/model/query/AiObservabilityCostsQuery.java", _JAVA_FIELD_PATTERN),
        go=("internal/api/ai_observability.go", "GetCosts"),
    ),
    PageDefaultAnchor(
        "图像输入历史.inputHistory.listHistory",
        10,
        py_router=("router/image_input.py", "list_history"),
        java=("com/pei/clarity/model/query/HistoryQuery.java", _JAVA_FIELD_PATTERN),
        go=("internal/api/sys_input_history.go", "ListHistory"),
    ),
    PageDefaultAnchor(
        "收藏.favorite.getPage",
        20,
        py_router=("router/favorite.py", "get_page"),
        java=("com/pei/clarity/model/query/FavoritePageQuery.java", _JAVA_INIT_PATTERN),
        go=("internal/api/favorite.go", "GetPage"),
    ),
    PageDefaultAnchor(
        "数据集数据项.datasetItem.listDatasetItems",
        20,
        py_router=("router/dataset_item.py", "list_dataset_items"),
        java=("com/pei/clarity/model/query/DatasetItemQuery.java", _JAVA_INIT_PATTERN),
        go=("internal/api/sys_dataset_item.go", "GetDatasetItems"),
    ),
    PageDefaultAnchor(
        "文件.file.getFilePage",
        10,
        py_router=("router/file.py", "get_file_page"),
        java=("com/pei/clarity/controller/FileController.java", _JAVA_REQUEST_PARAM_PATTERN),
        go=("internal/api/sys_file.go", "GetFilePage"),
    ),
    PageDefaultAnchor(
        "消息.message.searchMessages",
        20,
        py_router=("router/message.py", "search_messages"),
        java=("com/pei/clarity/model/query/MessageSearchQuery.java", _JAVA_INIT_PATTERN),
        go=("internal/api/sys_message.go", "Search"),
    ),
    PageDefaultAnchor(
        "基础默认.BasePageQuery",
        10,
        py_schema="BasePageQuery",
        java=("com/pei/clarity/common/base/BasePageQuery.java", _JAVA_FIELD_PATTERN),
        go=("internal/api/sys_feedback.go", "__plain_pagination__"),
    ),
)


def _match_one(text: str, pattern: str) -> int:
    """恰好命中一次才取值：0 次（抽取式失效/声明被删）与多次（口径分叉）都必须报错。"""
    found = re.findall(pattern, text)
    assert len(found) == 1, f"正则应恰好命中 1 次，实际 {len(found)} 次：{pattern}"
    return int(found[0])


def _java_default(rel: str, pattern: str) -> int:
    return _match_one((JAVA_SRC / rel).read_text(encoding="utf-8"), pattern)


def _query_call_default(node: ast.expr | None) -> int | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    if node.func.id != "Query":
        return None
    for keyword in node.keywords:
        if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            if isinstance(value, int):
                return value
    return None


def _py_router_default_in(text: str, endpoint: str) -> int | None:
    """router 端点函数的 pageSize 形参缺省值。"""
    tree = ast.parse(text)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != endpoint:
            continue
        args = [*node.args.posonlyargs, *node.args.args]
        defaults = [None] * (len(args) - len(node.args.defaults)) + list(node.args.defaults)
        for arg, default in zip(args, defaults, strict=True):
            if arg.arg == "pageSize":
                return _query_call_default(default)
    return None


def _py_router_default(rel: str, endpoint: str) -> int | None:
    return _py_router_default_in((PY_APP / rel).read_text(encoding="utf-8"), endpoint)


def _py_base_page_default_in(text: str) -> int | None:
    """`BasePageQuery.pageSize` 的 Field 缺省值（非 AI 域统一缺省的事实源）。"""
    tree = ast.parse(text)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "BasePageQuery":
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == "pageSize"
                and isinstance(stmt.value, ast.Call)
            ):
                for keyword in stmt.value.keywords:
                    if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                        value = keyword.value.value
                        if isinstance(value, int):
                            return value
    return None


def _py_schema_default(class_name: str) -> int | None:
    """schema 模型的 pageSize 缺省值：显式赋值优先，否则继承 BasePageQuery 的缺省。"""
    schema_dir = PY_APP / "models" / "schema"
    for path in sorted(schema_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name != class_name:
                continue
            for stmt in node.body:
                target = None
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                elif isinstance(stmt, ast.AnnAssign):
                    target = stmt.target
                if isinstance(target, ast.Name) and target.id == "pageSize":
                    value = getattr(stmt, "value", None)
                    if isinstance(value, ast.Constant) and isinstance(value.value, int):
                        return value.value
            return _py_base_page_default_in((schema_dir / "common.py").read_text(encoding="utf-8"))
    return None


def _py_default(anchor: PageDefaultAnchor) -> int | None:
    if anchor.py_router is not None:
        return _py_router_default(*anchor.py_router)
    assert anchor.py_schema is not None, f"{anchor.endpoint} 未给 python 侧声明位置"
    return _py_schema_default(anchor.py_schema)


def _go_plain_pagination_default() -> int:
    """`parsePagination` 的缺省值（定义处取值，避免 go 内侧两套口径）。"""
    return _match_one(
        (GO_SRC / "internal/api/sys_feedback.go").read_text(encoding="utf-8"),
        r"func parsePagination\(c \*gin\.Context\)[^{]*\{"
        r"\s*return parsePaginationWithSize\(c, (\d+)\)",
    )


def _go_handler_default_in(text: str, handler: str, plain_default: int) -> int | None:
    """go handler 内的分页缺省实参（`parsePagination` 取基础缺省）。"""
    match = re.search(rf"func \([^)\n]*\) {handler}\(c \*gin\.Context\) \{{", text)
    if match is None:
        return None
    rest = text[match.end() :]
    body_end = rest.find("\nfunc ")
    body = rest[: body_end if body_end > 0 else len(rest)]
    sized = re.search(r"parsePaginationWithSize\(c, (\d+)\)", body)
    if sized:
        return int(sized.group(1))
    named = re.search(r'parsePaginationNamed\(c, "[^"]+", "[^"]+", (\d+)\)', body)
    if named:
        return int(named.group(1))
    return plain_default if "parsePagination(" in body else None


def _go_default(rel: str, handler: str) -> int | None:
    plain_default = _go_plain_pagination_default()
    if handler == "__plain_pagination__":
        return plain_default
    source = (GO_SRC / rel).read_text(encoding="utf-8")
    return _go_handler_default_in(source, handler, plain_default)


def test_extractors_are_discriminating():
    """检测式自证（对照组）：0 命中/多命中不得静默取默认值，三种声明形态都要能取到值。"""
    assert _match_one("private Integer pageSize = 20;", _JAVA_FIELD_PATTERN) == 20
    assert _match_one("setPageSize(20);", _JAVA_INIT_PATTERN) == 20
    assert (
        _match_one(
            '@RequestParam(defaultValue = "20") Integer pageSize,',
            _JAVA_REQUEST_PARAM_PATTERN,
        )
        == 20
    )
    with pytest.raises(AssertionError):
        _match_one("setPageSize(20);", _JAVA_FIELD_PATTERN)
    with pytest.raises(AssertionError):
        _match_one("pageSize = 1;\npageSize = 2;", _JAVA_FIELD_PATTERN)

    router_src = (
        "async def get_page(\n"
        "    pageNum: int = Query(default=1, ge=1),\n"
        "    pageSize: int = Query(default=20, ge=1, le=100),\n"
        ") -> None:\n"
        "    pass\n"
    )
    assert _py_router_default_in(router_src, "get_page") == 20
    assert _py_router_default_in(router_src, "missing_endpoint") is None

    schema_src = (
        "class BasePageQuery(BaseModel):\n"
        '    pageSize: int = Field(default=10, ge=1, le=100, description="每页记录数")\n'
    )
    assert _py_base_page_default_in(schema_src) == 10

    go_src = (
        "func (a *X) ListRecords(c *gin.Context) {\n"
        "\tpage, size, ok := parsePaginationWithSize(c, 20)\n"
        "}\n"
        "func (a *X) Other(c *gin.Context) {\n"
        "\tpage, size, ok := parsePagination(c)\n"
        "}\n"
    )
    assert _go_handler_default_in(go_src, "ListRecords", 10) == 20
    assert _go_handler_default_in(go_src, "Other", 10) == 10
    assert _go_handler_default_in(go_src, "Absent", 10) is None


def test_anchor_table_is_not_shrunk():
    """表规模锚定：删条目会把守卫缩成空表，故钉住条数与必覆盖的端点。"""
    endpoints = {anchor.endpoint for anchor in ANCHORS}
    assert len(ANCHORS) >= 12, f"对拍表被缩减（{len(ANCHORS)} 条）"
    assert {
        "计费.billing.listRecords",
        "可观测.observability.listTraces",
        "图像输入历史.inputHistory.listHistory",
        "收藏.favorite.getPage",
        "数据集数据项.datasetItem.listDatasetItems",
        "文件.file.getFilePage",
        "基础默认.BasePageQuery",
    } <= endpoints


def test_page_defaults_match_python_across_ends():
    """三端缺省值逐端点对 python：任一端与锚定值不一致即红。"""
    violations: list[str] = []
    for anchor in ANCHORS:
        actual = {
            "python": _py_default(anchor),
            "java": _java_default(*anchor.java) if anchor.java else None,
            "go": _go_default(*anchor.go) if anchor.go else None,
        }
        for end, value in actual.items():
            if value != anchor.expected:
                violations.append(
                    f"{anchor.endpoint}: {end} 缺省 {value}（应为 {anchor.expected}）"
                )
    detail = "\n".join(violations)
    assert not violations, f"分页缺省值跨端不一致（默认值逐端点对 python）：\n{detail}"
