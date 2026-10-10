#!/usr/bin/env python3
"""Dockerfile 铁律门禁（只读静态检查，不构建镜像、不执行 docker）。

测试镜像的三条铁律此前只写在文档与 Dockerfile 注释里，靠人记 —— 漏一条的代价是
「构建成功但单测成片挂」或「镜像莫名多出数十 GB」，且要到 Pod 里才暴露。
本脚本把三条铁律变成机器拦截。

铁律与判据（口径来源：dehaze-doc/docs/02-系统架构/05-测试架构.md、06-部署架构.md）：

1. `uv sync` 必须带 `--no-cache`；测试阶段（`FROM ... AS test`）的 `uv sync`，若该项目
   `pyproject.toml` 把测试依赖声明在 `[project.optional-dependencies].test`，还须带 `--extra test`。
   - 缺 `--no-cache`：已下载的轮子留在 `/root/.cache/uv` 永久占住镜像层（实测 10.2G → 19.7G），
     后一层再 `rm` 也不缩小。
   - 缺 `--extra test`：dehaze-python 的 respx/mongomock/freezegun/pytest-mock 在该 extra 里，
     缺它 6 个测试文件收集失败。测试依赖若落在 `[dependency-groups].dev`
     （uv 默认安装，推理服务即此形态）则**不需要**该参数 —— 故本项按项目实际声明判定，
     不做一刀切。
2. 测试镜像必须 COPY 仓库根 `config/`。三端测试都用 `config/sql` 全量重建测试库；
   go 镜像漏了它会 104 个建库用例全挂。判定方式：测试阶段 COPY 进镜像的源码里若有引用
   `config/sql`，该阶段必须有一条 COPY 的源指向仓库根 `config`（`COPY . .` 这类整体拷贝也算）。
3. Dockerfile 禁用 heredoc（`<<EOF` / `<<-'EOF'`）。不带 `\\` 续行的 heredoc 体会被 Dockerfile
   解析吃掉（`settings.xml` 被写成空文件），改用 `printf '%s\\n'` 续行
   （见 dehaze-java/Dockerfile）。

用法：
    python scripts/check_dockerfiles.py                 # 扫描仓库内全部 Dockerfile / Dockerfile.*
    python scripts/check_dockerfiles.py dehaze-go/Dockerfile   # 只查指定文件

退出码：0 = 全部通过；1 = 存在违规（逐条打印 `文件:行` + 修法）；2 = 用法/环境错误。
只读：仅读取仓库内文本文件。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 扫描被 COPY 源码时剪枝：依赖 / 构建产物 / 缓存目录一律不看
SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "target",
        "build",
        "dist",
        ".next",
        "out",
        ".pytest_cache",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        "logs",
    }
)
# 单个源码文件超过该大小即跳过（测试源码不会这么大，避免误扫二进制/产物）
MAX_TEXT_BYTES = 1 << 20
# 每条铁律最多回报几处「源码引用 config/sql」的证据
MAX_EVIDENCE = 3

TEST_STAGE_NAME = "test"
UV_SYNC_RE = re.compile(r"\buv\s+sync\b")
NO_CACHE_RE = re.compile(r"--no-cache\b")
EXTRA_TEST_RE = re.compile(r"--extra[= ]+test\b")
HEREDOC_RE = re.compile(r"<<-?\s*[\"']?[A-Za-z_][A-Za-z0-9_]*")
CONFIG_SQL_RE = re.compile(r"config[/\\]sql")
FROM_RE = re.compile(r"^FROM\s+(?P<args>.+)$", re.IGNORECASE)
AS_RE = re.compile(r"\bAS\s+(?P<name>[\w.-]+)\s*$", re.IGNORECASE)
COPY_RE = re.compile(r"^COPY\s+(?P<args>.+)$", re.IGNORECASE)
TEST_EXTRA_KEY_RE = re.compile(r"^test[\w.-]*\s*=")
OPTIONAL_DEPS_PREFIX = "[project.optional-dependencies"


def display_path(path: Path) -> str:
    """展示用路径：仓库内用相对路径，仓库外（自证/临时夹具）用绝对路径。"""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class Violation:
    """一条违规：铁律 + 定位 + 现况 + 修法。"""

    def __init__(self, rule: str, path: Path, line: int, message: str, fix: str) -> None:
        self.rule = rule
        self.path = path
        self.line = line
        self.message = message
        self.fix = fix

    def render(self) -> str:
        return (
            f"\u2717 {self.path}:{self.line}  [{self.rule}]\n  {self.message}\n  修法: {self.fix}"
        )


def discover_dockerfiles(root: Path) -> list[Path]:
    """仓库内全部 Dockerfile / Dockerfile.*（跳过依赖与构建产物目录）。"""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
        found.extend(
            Path(dirpath) / filename
            for filename in filenames
            if filename == "Dockerfile" or filename.startswith("Dockerfile.")
        )
    return sorted(found)


def iter_instructions(lines: list[str]):
    """合并反斜杠续行，产出 (起始行号, 单行指令文本)。"""
    buffer: list[str] = []
    start = 0
    for lineno, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        if not buffer:
            start = lineno
        if line.endswith("\\"):
            buffer.append(line[:-1])
            continue
        buffer.append(line)
        yield start, " ".join(part.strip() for part in buffer)
        buffer = []
    if buffer:
        yield start, " ".join(part.strip() for part in buffer)


class Stage:
    """一个构建阶段：名字（`FROM ... AS <name>`）+ 其内指令。"""

    def __init__(self, name: str | None, from_line: int) -> None:
        self.name = name
        self.from_line = from_line
        self.instructions: list[tuple[int, str]] = []

    @property
    def is_test(self) -> bool:
        return self.name == TEST_STAGE_NAME


def parse_stages(text: str) -> list[Stage]:
    stages: list[Stage] = []
    current: Stage | None = None
    for lineno, instruction in iter_instructions(text.splitlines()):
        if instruction.startswith("#"):
            continue
        from_match = FROM_RE.match(instruction)
        if from_match:
            name_match = AS_RE.search(from_match.group("args"))
            name = name_match.group("name").lower() if name_match else None
            current = Stage(name, lineno)
            stages.append(current)
            continue
        if current is not None:
            current.instructions.append((lineno, instruction))
    return stages


def copy_sources(stage: Stage) -> list[tuple[int, str]]:
    """阶段的 COPY 来源（相对仓库根；`--from=` 外部镜像来源不计）。"""
    sources: list[tuple[int, str]] = []
    for lineno, instruction in stage.instructions:
        match = COPY_RE.match(instruction)
        if not match:
            continue
        args = match.group("args")
        if "--from=" in args:
            continue
        tokens = [token for token in args.split() if not token.startswith("--")]
        if len(tokens) < 2:
            continue
        sources.extend((lineno, source) for source in tokens[:-1])
    return sources


def iter_source_files(sources: list[str]):
    """把 COPY 来源展开成待扫描文件（去重、剪枝目录、跳过二进制）。"""
    seen: set[Path] = set()
    for source in sources:
        base = REPO_ROOT / source
        if not base.exists():
            continue
        candidates = [base] if base.is_file() else list(base.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file() or candidate in seen:
                continue
            try:
                parts = candidate.relative_to(REPO_ROOT).parts
                if SKIP_DIR_NAMES.intersection(parts):
                    continue
                if candidate.stat().st_size > MAX_TEXT_BYTES:
                    continue
                text = candidate.read_text(encoding="utf-8")
            except (OSError, ValueError, UnicodeDecodeError):
                continue
            seen.add(candidate)
            yield candidate, text


def find_config_sql_evidence(sources: list[str]) -> list[str]:
    """在 COPY 进镜像的源码里找 `config/sql` 引用。

    返回 `相对路径:行号` 证据（最多 MAX_EVIDENCE 条）。
    """
    evidence: list[str] = []
    for path, text in iter_source_files(sources):
        for lineno, line in enumerate(text.splitlines(), start=1):
            if CONFIG_SQL_RE.search(line):
                evidence.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
                break
        if len(evidence) >= MAX_EVIDENCE:
            break
    return evidence


def copies_repo_root_config(sources: list[str]) -> bool:
    """是否有 COPY 来源指向仓库根 `config/`（`COPY . .` 这类整体拷贝也视为覆盖）。"""
    for source in sources:
        resolved = (REPO_ROOT / source).resolve()
        if resolved in {REPO_ROOT.resolve(), (REPO_ROOT / "config").resolve()}:
            return True
    return False


def project_dir_of(stage: Stage) -> Path | None:
    """阶段内 COPY 进来的 pyproject.toml 所属项目目录（用于判定 test extra 声明方式）。"""
    for _lineno, source in copy_sources(stage):
        if Path(source).name == "pyproject.toml":
            return (REPO_ROOT / source).parent
    return None


def declares_test_extra(project_dir: Path | None) -> bool | None:
    """项目是否把测试依赖声明在 `[project.optional-dependencies].test`。

    None = 定位不到 pyproject。
    """
    if project_dir is None:
        return None
    pyproject = project_dir / "pyproject.toml"
    if not pyproject.is_file():
        return None
    in_section = False
    for raw in pyproject.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("["):
            in_section = line.startswith(OPTIONAL_DEPS_PREFIX)
            if line.startswith(f"{OPTIONAL_DEPS_PREFIX}."):
                extra_name = line[len(OPTIONAL_DEPS_PREFIX) + 1 :].rstrip("]").strip()
                if extra_name.startswith("test"):
                    return True
            continue
        if in_section and TEST_EXTRA_KEY_RE.match(line):
            return True
    return False


def check_dockerfile(path: Path) -> tuple[list[Violation], list[str]]:
    violations: list[Violation] = []
    notes: list[str] = []
    rel = display_path(path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # 铁律 3：禁 heredoc
    for lineno, line in enumerate(lines, start=1):
        heredoc = HEREDOC_RE.search(line)
        if heredoc:
            violations.append(
                Violation(
                    "铁律3-禁 heredoc",
                    rel,
                    lineno,
                    f"检测到 heredoc `{heredoc.group(0)}`：不带 `\\` 续行的 heredoc 体会被 "
                    "Dockerfile 解析吃掉，写入的文件会变成空文件",
                    "改用 `printf '%s\\n'` 逐行写入"
                    "（见 dehaze-java/Dockerfile 的 settings.xml 写法）",
                )
            )

    for stage in parse_stages(text):
        stage_label = f"{rel}#{stage.name or '<未命名阶段>'}"
        project_dir = project_dir_of(stage)

        # 铁律 1：uv sync
        for lineno, instruction in stage.instructions:
            if not UV_SYNC_RE.search(instruction):
                continue
            if not NO_CACHE_RE.search(instruction):
                violations.append(
                    Violation(
                        "铁律1-uv sync 缺 --no-cache",
                        rel,
                        lineno,
                        "`uv sync` 未带 `--no-cache`：已下载的轮子会留在 `/root/.cache/uv` 永久占住"
                        "镜像层（实测 10.2G → 19.7G）",
                        f"改为 `{_uv_sync_fix(instruction, extra=False)}`",
                    )
                )
            if not stage.is_test:
                continue
            if project_dir is None:
                violations.append(
                    Violation(
                        "铁律1-uv sync 项目不可判定",
                        rel,
                        lineno,
                        f"测试阶段 `{stage_label}` 有 `uv sync`，但阶段内没有 COPY 该项目的 "
                        "`pyproject.toml`，无法判定测试依赖是否声明在 test extra",
                        "在 `uv sync` 之前补 "
                        "`COPY <项目目录>/pyproject.toml <项目目录>/uv.lock ./`",
                    )
                )
                continue
            test_extra = declares_test_extra(project_dir)
            if test_extra is True and not EXTRA_TEST_RE.search(instruction):
                violations.append(
                    Violation(
                        "铁律1-uv sync 缺 --extra test",
                        rel,
                        lineno,
                        f"`{project_dir.relative_to(REPO_ROOT)}/pyproject.toml` 把测试依赖声明在 "
                        "`[project.optional-dependencies].test`，但测试阶段的 `uv sync` 未带 "
                        "`--extra test`（缺 respx/mongomock/freezegun/pytest-mock ⇒ "
                        "6 个测试文件收集失败）",
                        f"改为 `{_uv_sync_fix(instruction, extra=True)}`",
                    )
                )
            elif test_extra is False:
                notes.append(
                    f"[ok] {stage_label}: `uv sync` 项目 {project_dir.relative_to(REPO_ROOT)} "
                    "测试依赖在 dependency-groups（uv 默认安装）⇒ 无需 `--extra test`"
                )

        # 铁律 2：测试镜像必须 COPY 仓库根 config/
        if not stage.is_test:
            continue
        sources = [source for _lineno, source in copy_sources(stage)]
        evidence = find_config_sql_evidence(sources)
        if not evidence:
            notes.append(
                f"[ok] {stage_label}: 测试源码未引用 仓库根 config/sql ⇒ 无需 COPY config/"
            )
            continue
        if copies_repo_root_config(sources):
            notes.append(
                f"[ok] {stage_label}: 测试源码引用 config/sql（{evidence[0]}）"
                "⇒ 需 COPY config/，已具备"
            )
            continue
        violations.append(
            Violation(
                "铁律2-测试镜像漏 COPY config/",
                rel,
                stage.from_line,
                f"测试阶段 `{stage_label}` 的源码引用了仓库根 `config/sql`"
                f"（{', '.join(evidence)}），但该阶段没有任何 COPY 指向仓库根 `config`："
                "测试库无法由 schema/种子全量重建（go 镜像漏了它曾 104 个建库用例全挂）",
                "在该阶段补 `COPY config /app/config`（构建上下文须为仓库根）",
            )
        )

    return violations, notes


def _uv_sync_fix(instruction: str, *, extra: bool) -> str:
    """按现况给出补全后的 uv sync 命令（只用于把修法写进提示）。"""
    fixed = instruction
    if not NO_CACHE_RE.search(fixed):
        fixed = UV_SYNC_RE.sub("uv sync --no-cache", fixed, count=1)
    if extra and not EXTRA_TEST_RE.search(fixed):
        fixed = re.sub(r"(uv sync(?: [^ ]+)*)", r"\1 --extra test", fixed, count=1)
    return fixed


def main(argv: list[str]) -> int:
    args = [arg for arg in argv[1:] if arg not in {"-h", "--help"}]
    if "-h" in argv[1:] or "--help" in argv[1:]:
        print(__doc__)
        return 0

    if args:
        targets = []
        for arg in args:
            path = Path(arg)
            if not path.is_absolute():
                path = REPO_ROOT / arg
            if not path.is_file():
                print(f"用法错误：找不到 Dockerfile：{arg}", file=sys.stderr)
                return 2
            targets.append(path)
    else:
        targets = discover_dockerfiles(REPO_ROOT)
    if not targets:
        print("用法错误：仓库内未发现任何 Dockerfile", file=sys.stderr)
        return 2

    violations: list[Violation] = []
    notes: list[str] = []
    for path in targets:
        file_violations, file_notes = check_dockerfile(path)
        violations.extend(file_violations)
        notes.extend(file_notes)

    for note in notes:
        print(note)
    if violations:
        violations.sort(key=lambda item: (str(item.path), item.line, item.rule))
        print(f"\n检测到 {len(violations)} 处违规（扫描 {len(targets)} 个 Dockerfile）：")
        for violation in violations:
            print(violation.render())
        return 1

    print(f"\n\u2705 Dockerfile 铁律检查通过（扫描 {len(targets)} 个 Dockerfile）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
