"""防重复提交（A0002）三端契约守卫（免库、免 Redis，源码级对拍）。

背景：三端共享同一 Redis，防重的**端点集合**与**键格式**是跨端契约：

- 端点集合不一致 ⇒ 同一请求在 A 端被拦、B 端放行（跨端重复提交防不住，或误伤合法连续操作）；
- 键格式不一致 ⇒ 三端各写各的键，跨端互不设防。

两者都不会报错，只会静默失效，故用本测试钉死：

1. **白名单集合**：python 中间件白名单 == java `@PreventDuplicateSubmit` == go 路由挂载（8 条）；
2. **键格式**：三端均为 `anti_repeat:{identity}:{METHOD}:{path}:{sortedQuery}:{bodyHash}` 五段、
   段序一致、`identity` 统一用户维度（`sessionId`/`jti` 会随登录态变化而绕过）；
3. **query 归一**：两端实现按 (key, value) 全排序（`?a=1&b=2` 与 `?b=2&a=1` 必须同键）。

口径变更须三端同改并同步本表；本测试为源码级，改任一端即红。
"""

import re
from pathlib import Path

import pytest

from app.middleware.anti_repeat import (
    _IMPORT_MODULES,
    _PROTECTED_PATHS,
    _PROTECTED_PATTERNS,
    is_protected,
    normalize_query,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
JAVA_CONTROLLER_DIR = (
    REPO_ROOT / "dehaze-java" / "src" / "main" / "java" / "com" / "pei" / "dehaze" / "controller"
)
GO_ROUTER_DIR = REPO_ROOT / "dehaze-go" / "internal" / "router"

JAVA_ASPECT = (
    REPO_ROOT
    / "dehaze-java"
    / "src"
    / "main"
    / "java"
    / "com"
    / "pei"
    / "dehaze"
    / "plugin"
    / "dupsubmit"
    / "aspect"
    / "DuplicateSubmitAspect.java"
)
GO_MIDDLEWARE = REPO_ROOT / "dehaze-go" / "pkg" / "server" / "gin" / "middleware" / "anti_repeat.go"
PY_MIDDLEWARE = REPO_ROOT / "dehaze-python" / "app" / "middleware" / "anti_repeat.py"

# 期望白名单（单一信息源）：8 条。入选标准＝创建/导入类且重复提交有副作用；
# 只读查询（`_export`）与"可连续操作的合法场景"（上传、数据项创建、输入历史）不入列。
EXPECTED_ENDPOINTS = frozenset(
    {
        "POST /api/v1/depts",
        "POST /api/v1/dict",
        "POST /api/v1/dict/types",
        "POST /api/v1/menus",
        "POST /api/v1/roles",
        "POST /api/v1/users",
        "POST /api/v1/announcements",
        "POST /api/v1/{module}/_import",
    }
)

# 键的五段在源码中的锚点（顺序敏感）：任一端漏段或换序即红。
KEY_SEGMENT_ANCHORS = {
    # python 用带花括号的占位符（docstring 里有同名裸词，会干扰顺序判定）
    "python": ("{identity}", "{method}", "{path}", "{query}", "{body_hash}"),
    "java": (
        "resolveIdentity(request)",
        "request.getMethod()",
        "request.getRequestURI()",
        "normalizeQuery(request.getQueryString())",
        "resolveBodyHash(request)",
    ),
    # go 侧 `c.Request.Method` / `c.Request.URL.Path` 在日志与 SkipPaths 处也出现，
    # 锚点须带键拼接后缀，才能锁定 defaultKeyGenerator 内的段序
    "go": (
        'resolveIdentity(c) + ":"',
        'c.Request.Method + ":"',
        'c.Request.URL.Path + ":"',
        'normalizeQuery(c.Request.URL.RawQuery) + ":"',
        "resolveBodyHash(c)",
    ),
}


def _join_path(*parts: str) -> str:
    """拼接路径段并规范化斜杠（各端 Group 前缀有的带前导斜杠、有的不带）。"""
    segments = [part.strip("/") for part in parts if part and part.strip("/")]
    return "/" + "/".join(segments) if segments else "/"


def collect_java_endpoints(source: str) -> set[str]:
    """从单个 java controller 源码提取带 `@PreventDuplicateSubmit` 的 POST 端点。

    解析形态：方法级 `@PostMapping`（可无参或带字面量路径）+ 紧随的 `@PreventDuplicateSubmit`，
    类级前缀取其 `@RequestMapping("...")`。
    """
    class_match = re.search(r'@RequestMapping\(\s*"([^"]*)"\s*\)', source)
    class_path = class_match.group(1) if class_match else ""

    endpoints: set[str] = set()
    method_path: str | None = None
    pending = False
    for raw_line in source.splitlines():
        line = raw_line.strip()
        post = re.match(r'@PostMapping\s*(?:\(\s*(?:value\s*=\s*)?"([^"]*)"\s*\))?', line)
        if post:
            method_path = post.group(1) or ""
            continue
        if line.startswith("@PreventDuplicateSubmit"):
            pending = True
            continue
        if pending and re.match(r"(public|protected)\s", line):
            endpoints.add(f"POST {_join_path(class_path, method_path or '')}")
            pending = False
            method_path = None
    return endpoints


def collect_go_endpoints(source: str) -> set[str]:
    """从单个 go router 源码提取挂了 `DefaultAntiRepeat()` 的 POST 端点。

    解析形态：`<groupVar>.POST("<path>", ..., middleware.DefaultAntiRepeat(), ...)`，
    路径前缀取同文件内 `<groupVar> := ...Group("<prefix>")`；循环注册的 `"/"+module+"/_import"`
    归一为参数化表示。
    """
    group_prefix = {
        match.group(1): match.group(2)
        for match in re.finditer(r'(\w+)\s*:?=\s*[\w.]*Group\(\s*"([^"]*)"\s*\)', source)
    }

    endpoints: set[str] = set()
    for anchor in re.finditer(r"DefaultAntiRepeat\(\)", source):
        # 取锚点之前**最近**的 POST 调用（挂载点可能跨行）
        post = None
        # 参数化的拼接式路径要排在字面量之前匹配，否则会先被 `"[^"]*"` 吃掉半截
        for candidate in re.finditer(
            r'(\w+)\.POST\(\s*("/"\s*\+\s*module\s*\+\s*"/_[a-z]+"|"[^"]*")',
            source[: anchor.start()],
        ):
            post = candidate
        if post is None:
            continue
        target = post.group(2)
        if "module" in target:
            action = re.search(r'"/_([a-z]+)"', target)
            assert action is not None, f"无法解析参数化导入路径：{target}"
            endpoints.add(f"POST /api/v1/{{module}}/_{action.group(1)}")
            continue
        literal_path = target.strip('"')
        endpoints.add(
            f"POST {_join_path('/api/v1', group_prefix.get(post.group(1), ''), literal_path)}"
        )
    return endpoints


def _collect_directory(directory: Path, suffixes: tuple[str, ...], collector) -> set[str]:
    endpoints: set[str] = set()
    for path in sorted(directory.rglob("*")):
        if path.suffix not in suffixes or not path.is_file():
            continue
        endpoints |= collector(path.read_text(encoding="utf-8"))
    return endpoints


def python_endpoints() -> set[str]:
    """python 侧白名单的对外表示（参数化模块路径统一为 `{module}`）。"""
    endpoints = {f"POST {path}" for path in _PROTECTED_PATHS}
    if any("_import" in pattern.pattern for pattern in _PROTECTED_PATTERNS):
        endpoints.add("POST /api/v1/{module}/_import")
    return endpoints


# ---------- 1. 白名单集合三端对拍 ----------


def test_python_whitelist_matches_expected():
    assert python_endpoints() == EXPECTED_ENDPOINTS


def test_java_annotations_match_expected():
    assert (
        _collect_directory(JAVA_CONTROLLER_DIR, (".java",), collect_java_endpoints)
        == EXPECTED_ENDPOINTS
    )


def test_go_route_mounts_match_expected():
    assert _collect_directory(GO_ROUTER_DIR, (".go",), collect_go_endpoints) == EXPECTED_ENDPOINTS


def test_import_module_set_matches_between_ends():
    """导入端点白名单的模块闭集必须与 go 的路由注册列表一致。

    两者不一致时，同一模块的导入会在 A 端被防重、B 端不拦（白名单"相等"的另一半）。
    """
    go_source = (GO_ROUTER_DIR / "import_export.go").read_text(encoding="utf-8")
    match = re.search(r"supportedImportExportModules\s*=\s*\[\]string\{([^}]*)\}", go_source)
    assert match is not None, "未找到 go 的 supportedImportExportModules 声明"
    go_modules = tuple(re.findall(r'"([^"]+)"', match.group(1)))
    assert tuple(_IMPORT_MODULES) == go_modules


def test_python_whitelist_actually_takes_effect():
    """白名单条目必须真的被 `is_protected` 命中（防"表里有、判定里没有"）。"""
    sample_import = f"/api/v1/{_IMPORT_MODULES[0]}/_import"
    for path in [*_PROTECTED_PATHS, sample_import]:
        assert is_protected(path), f"{path} 在白名单表内但未被 is_protected 命中"


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/dataset-items",
        "/api/v1/dataset-items/upload",
        "/api/v1/files",
        "/api/v1/image-input/history",
        "/api/v1/item-files",
        "/api/v1/user/_export",
        "/api/v1/ai/conversations",
        "/api/v1/orders",
        "/api/v1/users/1/password",
        f"/api/v1/{_IMPORT_MODULES[0]}/_import/x",
    ],
)
def test_non_whitelist_paths_are_exempt(path: str):
    """反向断言：不在白名单的路径必须放行。

    尤其 `/api/v1/users/1/password` —— "排除整个前缀"的旧实现会连带放过该域其他 POST，
    白名单制必须精确到端点。
    """
    assert not is_protected(path)


# ---------- 2. 键格式三端对拍 ----------


def test_python_key_format_is_five_segments():
    source = PY_MIDDLEWARE.read_text(encoding="utf-8")
    assert re.search(
        r'f"anti_repeat:\{identity\}:\{method\}:\{path\}:\{query\}:\{body_hash\}"',
        source,
    ), "python 键格式不再是五段固定式"


@pytest.mark.parametrize("target", ["java", "go"])
def test_other_ends_declare_same_prefix(target: str):
    source = (JAVA_ASPECT if target == "java" else GO_MIDDLEWARE).read_text(encoding="utf-8")
    assert '"anti_repeat:"' in source, f"{target} 未声明统一键前缀 anti_repeat:"
    assert "LOCK:RESUBMIT:" not in source, f"{target} 仍残留旧前缀 LOCK:RESUBMIT:"


@pytest.mark.parametrize("target", ["python", "java", "go"])
def test_key_segment_order_is_consistent(target: str):
    """五段（identity → method → path → query → body）在源码中出现顺序必须一致。"""
    source = (
        PY_MIDDLEWARE if target == "python" else JAVA_ASPECT if target == "java" else GO_MIDDLEWARE
    ).read_text(encoding="utf-8")
    positions = []
    for anchor in KEY_SEGMENT_ANCHORS[target]:
        index = source.find(anchor)
        assert index >= 0, f"{target} 键缺段锚点：{anchor}"
        positions.append(index)
    assert positions == sorted(positions), f"{target} 键段序与契约不符：{positions}"


def test_java_identity_does_not_read_session_cookie():
    """java 身份不得再取 session cookie：sessionId 随登录态变化，会让"重新登录即绕过防重"。"""
    source = JAVA_ASPECT.read_text(encoding="utf-8")
    assert "SESSION_COOKIE_NAME" not in source
    assert "getCookies()" not in source
    assert '"user:" + userId' in source


def test_go_identity_does_not_use_jwt_jti():
    """go 身份不得再用 JWT jti（刷新即变化），须收敛到 `user:{userId}`。"""
    source = GO_MIDDLEWARE.read_text(encoding="utf-8")
    assert "customClaims.ID" not in source
    assert "customClaims.UserID" in source


# ---------- 3. query 归一（python 实现可直接执行） ----------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("a=1", "a=1"),
        ("b=2&a=1", "a=1&b=2"),
        ("t=b&t=a", "t=a&t=b"),
        ("flag", "flag="),
        ("a=1&b=2", "a=1&b=2"),
    ],
)
def test_normalize_query(raw: str, expected: str):
    assert normalize_query(raw) == expected


def test_normalize_query_makes_order_irrelevant():
    """排序归一的实质目的：参数顺序不同也必须同键（否则留一条绕过路径）。"""
    assert normalize_query("b=2&a=1") == normalize_query("a=1&b=2")


# ---------- 4. 变异自证：解析器真的会发现"少了声明" ----------


def test_detector_detects_missing_java_annotation():
    source = (
        '@RequestMapping("/api/v1/depts")\n@PostMapping\npublic Result<?> save() { return null; }\n'
    )
    assert collect_java_endpoints(source) == set()


def test_detector_detects_removed_go_mount():
    source = (
        "package router\n\n"
        "func Register(rg *gin.RouterGroup) {\n"
        '\tsysMenuRouter := rg.Group("menus")\n'
        '\tsysMenuRouter.POST("", middleware.Permission("sys:menu:add"), sysMenuApi.SaveMenu)\n'
        "}\n"
    )
    assert collect_go_endpoints(source) == set()


def test_detector_reads_go_multiline_mount():
    """go 的挂载常与 `DefaultAntiRepeat()` 分列两行，解析必须跨行命中。"""
    source = (
        "package router\n\n"
        "func Register(rg *gin.RouterGroup) {\n"
        "\tfor _, module := range supportedImportExportModules {\n"
        '\t\trg.POST("/"+module+"/_import", withModule(module), modulePermission("import"),\n'
        "\t\t\tmiddleware.DefaultAntiRepeat(), importExportApi.Import)\n"
        "\t}\n"
        "}\n"
    )
    assert collect_go_endpoints(source) == {"POST /api/v1/{module}/_import"}


def test_detector_normalizes_group_prefix_slash():
    """核心与 API 路径的 group 前缀写法不统一（`menus` vs `/announcements`），须归一。"""
    source = (
        "package router\n\n"
        "func Register(rg *gin.RouterGroup) {\n"
        '\tannRouter := rg.Group("/announcements")\n'
        '\tannRouter.POST("", middleware.Permission("notify:announcement:add"), '
        "middleware.DefaultAntiRepeat(), annApi.Create)\n"
        "}\n"
    )
    assert collect_go_endpoints(source) == {"POST /api/v1/announcements"}
