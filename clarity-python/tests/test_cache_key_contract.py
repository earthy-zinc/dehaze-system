"""缓存键声明契约（免库）：清单输出 + 唯一性 / 命名空间遮蔽守卫。

背景：缓存键散落在 `app/` 各模块的常量里（20+ 文件）且没有单一登记处，跨端（go/java）
对拍时无权威清单。本测试以 AST 扫描 **声明**（模块级 / 类级字符串常量）产出清单，并守住
三条不变量：

1. **同名异值**：同一常量名在不同模块声明了不同键 ⇒ 语义漂移
   （反例：`SESSION_PREFIX` 一处 `session:`、一处 `sess:`）。
2. **异名同值**：同一个键被两个常量名声明 ⇒ 重复定义、改一处漏一处
   （反例：`A = "ai:model:list"` 且 `B = "ai:model:list"`）。
3. **命名空间遮蔽**：完整键同时充当命名空间前缀（`x` 与 `x:*`）⇒ 键实例归属无法判定
   （反例：`x = "kb:list"` 与 `x_ns = "kb:list:{uid}"` 并存）。

口径（有意收窄，避免误报）：

- 只收**声明**：模块级 / 类级赋值的字符串常量，名字含
  `KEY(S)/PREFIX(ES)/PATTERN(S)/CHANNEL/NAMESPACE` 词段，或所在类名形如 `*CacheKeys`。
  函数内局部变量与内联 f-string 键（如 `f"kb:list:{uid}"`）不在范围内——要覆盖须先提成模块常量。
- 失效 pattern 清单（`router/cache.py` 的 `_BUSINESS_CACHE_PATTERNS`、
  `promotion_service.ONSALE_KEYS` 这类**集合**声明）按其用途本就覆盖自身键
  （如 `dept:options` 与 `dept:options*`），只进清单、不参与遮蔽判定。
- 声明名自称 `PREFIX/PATTERN` 的一律按命名空间看待：
  `TASK_REDIS_KEY_PREFIX = "task:running"`（值缺尾冒号，以 `f"{PREFIX}:{id}"` 与
  `f"{PREFIX}:*"` 使用）因此不与 `task:running:cache:` 冲突。
- 只比对声明，不解释调用语义：`MATCH x:*` 之类扫描用法能否扫到不该扫的键，属运行期问题，
  本契约不作判定。

清单供跨端逐键对拍：`pytest tests/test_cache_key_contract.py -s -k inventory`。
"""

import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"

_KEY_NAME_RE = re.compile(r"(^|_)(KEYS?|PREFIX|PREFIXES|PATTERNS?|CHANNEL|NAMESPACE)($|_)")
_NAMESPACE_NAME_RE = re.compile(r"(^|_)(PREFIX|PREFIXES|PATTERNS?)($|_)")
_CACHE_KEYS_CLASS_RE = re.compile(r"(Cache)?Keys$")
# 键值形态：仅字母数字与 `_:.{}%-*`（排除 URL / SQL / 日志格式 / 中文文案）
_KEY_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:\-{}%*]+$")
_PLACEHOLDER_RE = re.compile(r"\{[^}]*\}|%[sdif]")


@dataclass(frozen=True)
class KeyDeclaration:
    """一条缓存键声明：来源 `文件:行`、常量名、字面值"""

    file: str
    line: int
    name: str
    value: str
    collection: bool = False
    """是否来自集合声明（失效 pattern 清单），集合元素只进清单、不参与冲突判定"""

    @property
    def namespace(self) -> bool:
        """命名空间角色：自称 PREFIX/PATTERN（值可缺尾冒号）、或以 `:` / `*` 结尾、或含占位符"""
        return (
            bool(_NAMESPACE_NAME_RE.search(self.name))
            or self.value.endswith((":", "*"))
            or bool(_PLACEHOLDER_RE.search(self.value))
        )

    @property
    def concrete(self) -> bool:
        """完整键角色：字面值本身即一个具体键（无占位符、不以 `:` / `*` 结尾、不自称 PREFIX）"""
        return not self.namespace

    def __str__(self) -> str:
        return f"{self.value} ({self.name} @ {self.file}:{self.line})"


def _looks_like_key(value: str) -> bool:
    return bool(_KEY_VALUE_RE.match(value)) and ":" in value and "://" not in value


def _literal_values(node: ast.expr | None) -> list[str] | None:
    """取字符串常量：单值一项；列表 / 元组 / 集合 / `frozenset(...)` 的全部元素；否则 None"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        elements: list[ast.expr] = list(node.elts)
    elif isinstance(node, ast.Call):
        elements = list(node.args)
    else:
        return None
    values = [e.value for e in elements if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return values if elements and len(values) == len(elements) else None


def _scan_body(
    body: list[ast.stmt],
    file: str,
    out: list[KeyDeclaration],
    *,
    in_keys_class: bool,
) -> None:
    """扫描模块 / 类体的赋值语句（函数内的局部变量跳过）"""
    for node in body:
        if isinstance(node, ast.ClassDef):
            _scan_body(
                node.body, file, out, in_keys_class=bool(_CACHE_KEYS_CLASS_RE.search(node.name))
            )
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
            values = _literal_values(node.value)
            single = isinstance(node.value, ast.Constant)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            values = _literal_values(node.value)
            single = isinstance(node.value, ast.Constant)
        else:
            continue
        if values is None:
            continue
        for target in targets:
            if isinstance(target, ast.Attribute):
                name = target.attr
            elif isinstance(target, ast.Name):
                name = target.id
            else:
                continue
            if not in_keys_class and not _KEY_NAME_RE.search(name):
                continue
            out.extend(
                KeyDeclaration(file, node.lineno, name, value, not single)
                for value in values
                if _looks_like_key(value)
            )


def _collect_declarations() -> list[KeyDeclaration]:
    declarations: list[KeyDeclaration] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _scan_body(tree.body, str(path.relative_to(APP_DIR)), declarations, in_keys_class=False)
    return declarations


def _duplicate_name_conflicts(
    declarations: list[KeyDeclaration],
) -> dict[str, list[str]]:
    """同名异值：常量名 → 该名下的多个不同键值"""
    values_by_name: dict[str, set[str]] = defaultdict(set)
    for d in declarations:
        values_by_name[d.name].add(d.value)
    return {name: sorted(values) for name, values in values_by_name.items() if len(values) > 1}


def _duplicate_value_conflicts(
    declarations: list[KeyDeclaration],
) -> dict[str, list[str]]:
    """异名同值：键值 → 声明该键的多个常量名"""
    names_by_value: dict[str, set[str]] = defaultdict(set)
    for d in declarations:
        names_by_value[d.value].add(d.name)
    return {value: sorted(names) for value, names in names_by_value.items() if len(names) > 1}


def _namespace_prefix(value: str) -> str:
    """命名空间字面前缀：截到首个占位符之前，去掉尾部分隔符（`:` / `*`）"""
    match = _PLACEHOLDER_RE.search(value)
    head = value[: match.start()] if match else value
    return head.rstrip(":*")


def _shadowing_conflicts(
    declarations: list[KeyDeclaration],
) -> list[tuple[KeyDeclaration, KeyDeclaration]]:
    """命名空间遮蔽：完整键 `x` 与始于 `x` 的命名空间声明（`x:*` / `x:{id}` / `x:<段>`）并存"""
    conflicts: list[tuple[KeyDeclaration, KeyDeclaration]] = []
    for concrete in (d for d in declarations if d.concrete):
        for ns in (d for d in declarations if d.namespace):
            prefix = _namespace_prefix(ns.value)
            if prefix == concrete.value or prefix.startswith(f"{concrete.value}:"):
                conflicts.append((concrete, ns))
    return conflicts


DECLARATIONS = _collect_declarations()
DECLARED = [d for d in DECLARATIONS if not d.collection]
PATTERNS = [d for d in DECLARATIONS if d.collection]


def _format(declarations: list[KeyDeclaration]) -> list[str]:
    return [
        f"[cache-keys] {d.value}\t{d.file}:{d.line}\t{d.name}"
        for d in sorted(declarations, key=lambda d: (d.value, d.file, d.line))
    ]


def test_cache_key_inventory():
    """打印缓存键清单（`-s` 可见），并锚定收集器未失效（防假绿）"""
    print(
        f"== clarity-python 缓存键清单：{len(DECLARED)} 条声明 + {len(PATTERNS)} 条失效 pattern =="
    )
    print("\n".join(_format(DECLARATIONS)))

    # 锚点：这几条是跨端契约键（go/java 同库读写），缺任一条说明 AST 扫描或命名口径已失效
    assert {
        "captcha_code:",
        "session:",
        "dict:data:",
        "ai:balance:{user_id}",
        "task:running:cache:",
    } <= {d.value for d in DECLARED}
    # 兜底防线：收集量级明显缩水即视为收集器失效（当前 80+ 条声明）
    assert len(DECLARED) >= 60, f"收集到的键声明骤减（{len(DECLARED)} 条），收集口径或目录结构已变"


def test_no_duplicate_key_declarations():
    """同名异值 / 异名同值都必须为 0：一个键一个名字、一个名字一个键"""
    same_name = _duplicate_name_conflicts(DECLARED)
    same_value = _duplicate_value_conflicts(DECLARED)
    assert not same_name, f"同名常量声明了不同缓存键（语义漂移）：{same_name}"
    assert not same_value, f"同一缓存键被多个常量名声明（重复定义）：{same_value}"


def test_no_namespace_shadows_complete_key():
    """完整键不得同时充当命名空间前缀（`x` 与 `x:*` 不得并存）"""
    conflicts = _shadowing_conflicts(DECLARED)
    assert not conflicts, "完整键被命名空间遮蔽：" + "；".join(
        f"{concrete} 被 {ns} 遮蔽" for concrete, ns in conflicts
    )


def test_conflict_rules_detect_synthetic_violations():
    """规则自检：合成违规必须被判出——否则上两条断言可能因规则失效而恒绿"""
    assert _duplicate_name_conflicts(
        [
            KeyDeclaration("a.py", 1, "SESSION_PREFIX", "session:"),
            KeyDeclaration("b.py", 2, "SESSION_PREFIX", "sess:"),
        ]
    )
    assert _duplicate_value_conflicts(
        [
            KeyDeclaration("a.py", 1, "MODEL_LIST_KEY", "ai:model:list"),
            KeyDeclaration("b.py", 2, "MODEL_CACHE_KEY", "ai:model:list"),
        ]
    )
    assert _shadowing_conflicts(
        [
            KeyDeclaration("a.py", 1, "LIST_KEY", "kb:list"),
            KeyDeclaration("a.py", 2, "LIST_PREFIX", "kb:list:{user_id}"),
        ]
    )
    # 反向对照：同一命名空间下的兄弟键与合法嵌套命名空间不算冲突
    assert not _shadowing_conflicts(
        [
            KeyDeclaration("a.py", 1, "CACHE_PREFIX", "task:running:"),
            KeyDeclaration("a.py", 2, "TASK_CACHE_PREFIX", "task:running:cache:"),
            KeyDeclaration("a.py", 3, "TREE_KEY", "dataset:tree"),
            KeyDeclaration("a.py", 4, "OPTIONS_KEY", "dataset:tree:options"),
        ]
    )


# ==================== 跨端共享键对拍（go / java 同名键） ====================
#
# 上面三条不变量只管 python 内部声明；跨端共享键的风险在另一端：python 改了键名/分隔符、
# go/java 没跟（或反之），三端各自单测全绿而缓存互不命中——互不命中的表现是
# "改了数据不生效 / 读到过期值 / java 改字典后 python 侧滞后"，不是报错，极难察觉。
#
# 本段以**显式契约表**（python 为事实源，2026-10-03 逐键实测三端源码）为基准，对拍：
#   ① 每个共享键必须在它应出现的每一端都有声明（字面量）：删一端即红；
#   ② 格式口径：占位符写法（python `{x}` / go·java `%d`/`%s`）归一后必须与事实源一致，
#      无占位符的具体键要求**逐字相等**，有占位符的按命名空间前缀 + 同元数比对。
#   ③ 契约表规模锚定，防止后人"删条目"把守卫缩成空表。
#
# 提取口径：python 走 AST（字符串常量 + f-string 静态段，**排除 docstring**）；go / java 走行级
# 字面量（跳过整行注释与权限码引用行）。已知限制：两个方向的提取都只证明"该端在源码里声明了
# 这个键"，不证明它真的被读写（字符串常量里的说明文本仍会被计入）。

REPO_ROOT = Path(__file__).resolve().parents[2]

_END_ROOTS: dict[str, tuple[Path, ...]] = {
    "python": (APP_DIR,),
    "go": (REPO_ROOT / "clarity-go" / "internal", REPO_ROOT / "clarity-go" / "pkg"),
    "java": (REPO_ROOT / "clarity-java" / "src" / "main",),
}
_END_SUFFIX = {"python": ".py", "go": ".go", "java": ".java"}

_CROSS_PLACEHOLDER_RE = re.compile(r"\{[^}]*\}|%[sdf]")
_CROSS_LITERAL_RE = re.compile(r'"([^"\n]*)"')
# 跳过权限码引用行（`@PreAuthorize` / `middleware.Permission` 等与缓存键同形 `a:b:c`）与整行注释
_CROSS_SKIP_LINE_RE = re.compile(
    r"^\s*(#|//|\*|/\*)"
    r"|hasPerm|hasAuthority|PreAuthorize|Permission\(|CheckPermission|"
    r"require_permission|check_permission"
)
_CROSS_KEY_SHAPE_RE = re.compile(r"^[a-z][a-z0-9_]*(:[a-z0-9_*]*)+:*$")

_ALL_ENDS = ("python", "go", "java")


@dataclass(frozen=True)
class SharedKey:
    """跨端共享键事实源：template 中占位符统一写 `*`；ends＝必须声明该键的端"""

    template: str
    ends: tuple[str, ...] = _ALL_ENDS
    note: str = ""


SHARED_KEYS: tuple[SharedKey, ...] = (
    SharedKey("ai:model:list"),
    SharedKey("ai:provider:list"),
    SharedKey("ai:balance:*"),
    SharedKey("ai:arrears:*"),
    SharedKey("ai:bill:*:*"),
    SharedKey("ai:quota:daily:*:*"),
    SharedKey("ai:quota:monthly:*:*"),
    SharedKey("dict:data:*"),
    SharedKey("kb:list:*"),
    SharedKey("kb:list:admin"),
    SharedKey("kb:detail:*"),
    SharedKey("role:perms:*"),
    SharedKey("role:options"),
    SharedKey("menu:routes"),
    SharedKey("user:level:*"),
    SharedKey("session:*"),
    SharedKey("session:user:*"),
    SharedKey("captcha_code:*"),
    SharedKey("login:fail:ip:*"),
    SharedKey("dict:value:*", ("python", "go"), "java 未失效该前缀（待整改）"),
    SharedKey("package:detail:*", ("python", "go"), "java 无套餐明细缓存"),
    SharedKey(
        "package:onsale:*", ("python",), "go/java 均在售列表无缓存（2026-10-03 实测 0 命中）"
    ),
    SharedKey("menu:options", ("java",), "仅 java 内部菜单选项缓存"),
)

# 契约表锚点：这几条是跨端互认的关键键，缺任一条说明条目被删（契约表静默缩水）
_ANCHOR_TEMPLATES = {
    "ai:model:list",
    "ai:provider:list",
    "dict:data:*",
    "kb:list:*",
    "role:perms:*",
    "menu:routes",
    "package:onsale:*",
}


def _python_source_literals(text: str) -> list[tuple[str, int]]:
    """python 字符串常量 + f-string 静态段（排除 docstring）：`[(字面量, 行号)]`"""
    tree = ast.parse(text)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(id(first.value))

    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            out.append((node.value, node.lineno))
        elif isinstance(node, ast.JoinedStr):
            out.extend(
                (part.value, node.lineno)
                for part in node.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
    return out


def _scan_end_literals(end: str) -> dict[str, str]:
    """扫描某一端源码的字面量 → 缓存键候选（归一化占位符；值 → 首个来源 `文件:行`）"""
    found: dict[str, str] = {}

    def record(raw: str, where: str) -> None:
        value = _CROSS_PLACEHOLDER_RE.sub("*", raw)
        if _CROSS_KEY_SHAPE_RE.match(value):
            found.setdefault(value, where)

    for root in _END_ROOTS[end]:
        suffix = _END_SUFFIX[end]
        for path in sorted(root.rglob(f"*{suffix}")):
            if not path.is_file() or path.name.endswith(f"_test{suffix}"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            rel = path.relative_to(REPO_ROOT)
            if end == "python":
                for raw, lineno in _python_source_literals(text):
                    record(raw, f"{rel}:{lineno}")
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if _CROSS_SKIP_LINE_RE.search(line):
                    continue
                for match in _CROSS_LITERAL_RE.finditer(line):
                    record(match.group(1), f"{rel}:{lineno}")
    return found


END_LITERALS: dict[str, dict[str, str]] = {end: _scan_end_literals(end) for end in _END_ROOTS}


def _placeholder_count(value: str) -> int:
    """占位符元数：字面量已在扫描期归一为 `*`，故按 `*` 计数"""
    return value.count("*")


def _static_prefix(template: str) -> str:
    """命名空间前缀：首个占位符之前的静态文本（含尾分隔符）"""
    return template.split("*", 1)[0]


def _declaring_literal(template: str, lits: dict[str, str]) -> str | None:
    """该端是否声明了该键。

    无占位符（具键）：必须逐字相等；
    有占位符：须有字面量是模板本身的**前缀**（`kb:list:*` ⊇ `kb:list:` / `kb:list:*`），
    仅"同命名空间下的兄弟具键"（如 `kb:list:admin`）不算——否则把动态键改名成
    `kb:lists:{id}` 会被兄弟键蒙过，而它其实已与 python 断链。
    """
    if "*" not in template:
        return template if template in lits else None
    return next(
        (
            lit
            for lit in sorted(lits)
            # 必须是模板的前缀且落在段边界上（`kb:list:` / `kb:list:*`），`kb:list` 这类
            # 半截串拼出的键与对端不互认，不算声明
            if template.startswith(lit) and (lit == template or "*" in lit or lit.endswith(":"))
        ),
        None,
    )


def test_cross_end_shared_key_inventory(capsys):
    """三端提取器有效性锚点 + 契约表规模锚定（提取器或契约表失效即红，防假绿）"""
    for end, lits in END_LITERALS.items():
        assert lits, f"{end} 端未提取到任何缓存键字面量（提取式失效）"
        assert len(lits) >= 15, f"{end} 端提取量骤减（{len(lits)} 条），提取口径或目录结构已变"

    print(f"== 跨端共享键契约表：{len(SHARED_KEYS)} 条 ==")
    for end, lits in END_LITERALS.items():
        print(f"== {end} 端字面量 {len(lits)} 条 ==")
    with capsys.disabled():
        for key in SHARED_KEYS:
            for end in key.ends:
                hit = _declaring_literal(key.template, END_LITERALS[end])
                print(f"[shared-key] {key.template}\t{end}\t{END_LITERALS[end].get(hit or '', '')}")

    templates = {key.template for key in SHARED_KEYS}
    missing_anchors = sorted(_ANCHOR_TEMPLATES - templates)
    assert not missing_anchors, f"共享键契约表关键条目被删：{missing_anchors}"
    assert len(SHARED_KEYS) >= 22, f"共享键契约表被缩减（{len(SHARED_KEYS)} 条），跨端对拍会漏检"


def test_cross_end_shared_keys_declared_in_every_end():
    """每个共享键都必须在其应出现的每一端有声明；缺任一端＝缓存互不命中（静默）"""
    missing = [
        f"{key.template} 缺少 {end} 端声明（{key.note or '无豁免说明'}）"
        for key in SHARED_KEYS
        for end in key.ends
        if _declaring_literal(key.template, END_LITERALS[end]) is None
    ]
    assert not missing, "跨端共享键在部分端未声明（另一端改了键名/漏跟）：\n" + "\n".join(missing)


def test_cross_end_shared_key_format_matches_python():
    """格式口径对拍：具键逐字相等；同占位符元数的模板必须与事实源完全一致（分隔符/段名不得漂移）"""
    violations: list[str] = []
    for key in SHARED_KEYS:
        for end in key.ends:
            lits = END_LITERALS[end]
            if "*" not in key.template:
                continue  # 具键已由上面的逐字相等覆盖
            static = _static_prefix(key.template)
            for lit in sorted(lits):
                # 只比"同一命名空间（静态前缀逐字相同）+ 同占位符元数"的变体：
                # 前缀式声明（go 拼接式 `"kb:list:"`）与嵌套命名空间（`session:user:*`）另有其表项
                if _static_prefix(lit) != static:
                    continue
                if _placeholder_count(lit) != _placeholder_count(key.template):
                    continue
                if lit != key.template:
                    violations.append(f"{key.template}（{end} 端实际 {lit} @ {lits[lit]}）")
    assert not violations, "跨端共享键格式口径与事实源不一致：\n" + "\n".join(violations)


def test_cross_end_matcher_detects_synthetic_violations():
    """规则自检：合成违规必须被判出——否则上面两条断言可能因规则失效而恒绿"""
    assert _declaring_literal("kb:list:*", {"kb:list:*": "java:X:1"}) == "kb:list:*"
    # 前缀式声明（go 用 `"kb:list:"` + strconv 拼接）算已声明
    assert _declaring_literal("kb:list:*", {"kb:list:": "go:X:1"}) == "kb:list:"
    # 动态键改名 / 分隔符漂移 / 兄弟具键顶包，都必须判"未声明"
    assert _declaring_literal("kb:list:*", {"kb:lists:*": "java:X:1"}) is None
    assert _declaring_literal("kb:list:*", {"kb:list": "java:X:1"}) is None
    assert _declaring_literal("kb:list:*", {"kb:list:admin": "java:X:1"}) is None
    assert _declaring_literal("ai:model:list", {"ai:model:lists": "java:X:1"}) is None
    assert _declaring_literal("ai:model:list", {"ai:model:list": "java:X:1"}) == "ai:model:list"
    # 提取器对 python 源码形态的敏感度：f-string 静态段计入、docstring 不计入
    literals = dict.fromkeys(v for v, _ in _python_source_literals('x = f"kb:list:{uid}"'))
    assert "kb:list:" in literals
    assert _python_source_literals('"""kb:list: 说明文字"""') == []
    # 格式对拍的元数口径：占位符按归一化后的 `*` 计数
    assert _placeholder_count("ai:bill:*:*") == 2
    assert _placeholder_count("kb:list:") == 0
    assert _static_prefix("ai:bill:*:*") == "ai:bill:"
