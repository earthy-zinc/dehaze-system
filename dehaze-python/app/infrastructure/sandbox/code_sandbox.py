"""受限代码执行沙箱（CodeSandbox）

安全边界（产品红线，对齐后端实现 §5.2）：
- 白名单：仅支持 python 与 shell 两种语言；shell 仅放行白名单命令（黑名单可被
  rm 变体/编码绕过），且一律经用户确认。
- 环境隔离：子进程只继承运行必需的环境变量（PATH/LANG 等），宿主进程的 DB/Redis/
  MinIO/MCP 等凭据一律不传入——python 模式可 exec 任意代码，继承环境等于交出凭据。
- 超时：默认 60s 上限（参数可传更小），超时 kill 进程组并返回"执行超时"。
- 资源：POSIX 下用 resource.setrlimit 限制内存(RLIMIT_AS)/进程数(RLIMIT_NPROC)/CPU
  (RLIMIT_CPU)；Windows 无 resource 模块时 try/except 跳过资源限制。
- 弱隔离：子进程 cwd 置于临时目录（tempfile.TemporaryDirectory），不访问真实磁盘工作区。
- 输出：stdout/stderr 各截断 10KB（附总长度提示）；stderr 中临时目录路径替换为
  /workspace，不泄露宿主路径（错误信息结构化，不暴露内部堆栈）。

容器级隔离（每会话独立容器）为部署演进项，不在本轮实现（见后端实现 §5.2 与需求 §2.6.9）。
"""

import asyncio
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 配置（待 Lead 收编进 Settings）──────────────────────
CODE_SANDBOX_DEFAULT_TIMEOUT = int(os.getenv("CODE_SANDBOX_TIMEOUT", "60"))
CODE_SANDBOX_MAX_TIMEOUT = int(os.getenv("CODE_SANDBOX_MAX_TIMEOUT", "60"))
CODE_SANDBOX_MEM_MB = int(os.getenv("CODE_SANDBOX_MEM_MB", "512"))
CODE_SANDBOX_NPROC = int(os.getenv("CODE_SANDBOX_NPROC", "64"))
CODE_SANDBOX_OUTPUT_LIMIT = int(os.getenv("CODE_SANDBOX_OUTPUT_LIMIT", str(10 * 1024)))

# POSIX 资源限制（Windows 无 resource 模块则跳过）
try:
    import resource

    _HAS_RESOURCE = True
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]
    _HAS_RESOURCE = False

# Shell 命令白名单（保守起步：只读/文本处理/沙箱内文件操作）。白名单外一律拒绝，
# 避免 rm 变体、base64 管道、curl -o && sh 等绕过黑名单的组合。
#
# 不含解释器类命令（python/python3/awk/sed）：`python -c "..."` 与 awk 的 system()
# 都可起子进程，放行即把"只读/文本处理"的承诺变成任意代码执行。执行代码走
# python 通道（execute_python_code，带高危能力确认），shell 里不重复提供。
_SHELL_ALLOWED_COMMANDS = frozenset(
    {
        "basename",
        "cat",
        "cp",
        "cut",
        "date",
        "diff",
        "dirname",
        "echo",
        "expr",
        "grep",
        "head",
        "ls",
        "mkdir",
        "mv",
        "printf",
        "pwd",
        "seq",
        "sort",
        "tail",
        "touch",
        "tr",
        "uname",
        "uniq",
        "wc",
    }
)

# 命令段分隔符（管道/顺序/短路执行）
_SHELL_SEGMENT_SPLIT_RE = re.compile(r";|&&|\|\||\||\n")
# 段首环境变量赋值（LD_PRELOAD 之类的注入）
_SHELL_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# 沙箱外的路径（绝对路径/家目录/上级目录）：路径起始符也含引号，否则
# `cat "/etc/passwd"` 这类引号包裹的绝对路径会被漏判
_SHELL_OUTSIDE_PATH_RE = re.compile(r"""(?:^|[\s=("'])(?:~|/)""")
_SHELL_PARENT_PATH_RE = re.compile(r"\.\.")

# Python 高危能力：与 shell 同级，须经用户确认后才执行
_PYTHON_RISK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "网络访问",
        re.compile(
            r"\b(socket|urllib|urllib2|urllib3|requests|httpx|aiohttp|http\.client"
            r"|ftplib|smtplib|telnetlib|webbrowser|paramiko)\b"
        ),
    ),
    (
        "子进程/系统命令执行",
        re.compile(r"\b(subprocess|os\.system|os\.popen|os\.exec|pty\.spawn)\b"),
    ),
    ("沙箱外文件读写", re.compile(r"""open\(\s*["']/""")),
)


def _sandbox_env() -> dict[str, str]:
    """子进程环境：仅保留运行必需项，杜绝宿主凭据经 os.environ 被模型代码读取。"""
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
    }


def _rejected(reason: str) -> str:
    return f"命令未被沙箱放行（{reason}），已拒绝执行。如需执行请走人工确认流程。"


class CodeSandbox:
    """受限子进程执行（asyncio.create_subprocess_exec）。"""

    def __init__(
        self,
        timeout: int = CODE_SANDBOX_DEFAULT_TIMEOUT,
        mem_mb: int = CODE_SANDBOX_MEM_MB,
        nproc: int = CODE_SANDBOX_NPROC,
        output_limit: int = CODE_SANDBOX_OUTPUT_LIMIT,
    ):
        self._timeout = timeout
        self._mem_mb = mem_mb
        self._nproc = nproc
        self._output_limit = output_limit

    def check_command_policy(self, command: str) -> str | None:
        """shell 命令白名单校验：命中拒绝规则返回原因，可放行返回 None。"""
        for raw_segment in _SHELL_SEGMENT_SPLIT_RE.split(command):
            segment = raw_segment.strip()
            if not segment:
                continue
            if _SHELL_ASSIGNMENT_RE.match(segment):
                return _rejected("不支持环境变量赋值")
            if "$(" in segment or "`" in segment:
                return _rejected("不支持命令替换")
            if ">" in segment:
                return _rejected("不支持输出重定向")
            if _SHELL_OUTSIDE_PATH_RE.search(segment) or _SHELL_PARENT_PATH_RE.search(segment):
                return _rejected("不支持访问沙箱工作目录之外的路径")
            program = Path(segment.split()[0]).name
            if program not in _SHELL_ALLOWED_COMMANDS:
                return _rejected(f"命令 {program} 不在白名单内")
        return None

    def check_python_risk(self, code: str) -> str | None:
        """python 代码高危能力检测（网络/子进程/沙箱外文件），命中返回能力描述。

        python 模式免确认可 exec 任意代码，这些能力与 shell 同级，检测命中后由
        工具层走与 shell 相同的用户确认流程。
        """
        for reason, pat in _PYTHON_RISK_PATTERNS:
            if pat.search(code):
                return reason
        return None

    async def execute_code(
        self, code: str, language: str = "python", timeout: int | None = None
    ) -> dict:
        """在受限沙箱中执行代码，返回 {stdout, stderr, exitCode, timedOut}。"""
        language = (language or "python").lower()
        if timeout is None:
            timeout = self._timeout
        timeout = max(1, min(int(timeout), CODE_SANDBOX_MAX_TIMEOUT))

        if language == "python":
            cmd: list[str] = self._build_launcher(code, language)
        elif language == "shell":
            rejection = self.check_command_policy(code)
            if rejection:
                return {"stdout": "", "stderr": rejection, "exitCode": 1, "timedOut": False}
            cmd = self._build_launcher(code, language)
        else:
            return {
                "stdout": "",
                "stderr": f"不支持的语言: {language}",
                "exitCode": 1,
                "timedOut": False,
            }

        with tempfile.TemporaryDirectory(prefix="dehaze_sandbox_") as workdir:
            kwargs: dict = {
                "cwd": workdir,
                "env": _sandbox_env(),
                "stdin": subprocess.DEVNULL,
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
                # POSIX 下独立进程组，便于超时 killpg；Windows 不支持
                "start_new_session": _HAS_RESOURCE,
            }
            try:
                process = await asyncio.create_subprocess_exec(*cmd, **kwargs)
            except Exception as e:
                logger.warning("沙箱子进程启动异常: %s", e)
                return {
                    "stdout": "",
                    "stderr": "沙箱执行失败，请稍后重试",
                    "exitCode": 1,
                    "timedOut": False,
                }

            try:
                stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except TimeoutError:
                self._kill_process_group(process)
                await self._reap(process)
                return {
                    "stdout": "",
                    "stderr": f"执行超时({timeout}s)已终止",
                    "exitCode": -1,
                    "timedOut": True,
                }
            except Exception as e:
                logger.warning("沙箱子进程执行异常: %s", e)
                self._kill_process_group(process)
                await self._reap(process)
                return {
                    "stdout": "",
                    "stderr": "沙箱执行失败，请稍后重试",
                    "exitCode": 1,
                    "timedOut": False,
                }

            exit_code = process.returncode

        stdout = self._sanitize(stdout_b, workdir)
        stderr = self._sanitize(stderr_b, workdir)
        stdout, s_trunc = self._truncate(stdout)
        stderr, e_trunc = self._truncate(stderr)
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exitCode": exit_code,
            "timedOut": False,
            "truncated": {"stdout": s_trunc, "stderr": e_trunc},
        }

    def _build_launcher(self, code: str, language: str) -> list[str]:
        """构造沙箱启动命令。

        资源限制在子进程 Python 主线程内通过 resource.setrlimit 施加（而非 asyncio 的
        preexec_fn——后者在非主线程 fork 时可能死锁/段错误），随后再执行用户代码：
        - python：exec 用户脚本
        - shell：以 /bin/sh -c 执行用户命令
        Windows 无 resource 模块时跳过资源限制（仅保留临时目录弱隔离与超时）。
        """
        code_arg = code
        if _HAS_RESOURCE:
            mem_bytes = self._mem_mb * 1024 * 1024
            cpu = max(1, self._timeout)
            # 各平台对个别 limit 支持不一（macOS 的 RLIMIT_AS 会抛 ValueError），
            # 逐个 try 防御，不支持的平台跳过该限制，保留超时终止与临时目录弱隔离兜底
            rl = (
                "def _rl(n,a,b):\n"
                "  try: resource.setrlimit(n,(a,b))\n"
                "  except (ValueError,OSError): pass\n"
                f"_rl(resource.RLIMIT_AS,{mem_bytes},{mem_bytes});"
                f"_rl(resource.RLIMIT_NPROC,{self._nproc},{self._nproc});"
                f"_rl(resource.RLIMIT_CPU,{cpu},{cpu});"
            )
            if language == "shell":
                inner = (
                    "import resource,subprocess,sys\n"
                    + rl
                    + "sys.exit(subprocess.call(['/bin/sh','-c',sys.argv[1]]))"
                )
            else:  # python
                inner = "import resource,sys\n" + rl + "exec(sys.argv[1])"
        elif language == "shell":
            inner = "import subprocess,sys;sys.exit(subprocess.call(['/bin/sh','-c',sys.argv[1]]))"
        else:  # python, no resource
            inner = "import sys;exec(sys.argv[1])"
        return [sys.executable, "-c", inner, code_arg]

    @staticmethod
    def _kill_process_group(process) -> None:
        """终止整个进程组（POSIX），兜底单进程 kill。"""
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # killpg 失败（进程组已不存在/无权限）退化为单进程 kill
            try:
                process.kill()
            except ProcessLookupError:
                logger.debug("沙箱子进程已退出，无需终止: pid=%s", process.pid)

    @staticmethod
    async def _reap(process) -> None:
        """等待被终止的进程回收，确保 subprocess 传输在当前事件循环内完成清理。

        超时/异常路径下 communicate 被取消后，进程 transport 可能残留到循环关闭
        （触发 "Event loop is closed" ResourceWarning），此处显式回收避免告警。
        回收失败不覆盖上层已返回的超时/错误结果，但必须可见以便定位残留进程。
        """
        try:
            await process.wait()
        except Exception as e:
            logger.warning("回收沙箱子进程失败: %s", e, exc_info=True)

    @staticmethod
    def _sanitize(data: bytes, workdir: str) -> str:
        """解码 stderr/stdout，并将临时目录路径替换为 /workspace（不泄露宿主路径）。"""
        text = data.decode("utf-8", errors="replace")
        if workdir:
            text = text.replace(workdir, "/workspace")
        return text

    def _truncate(self, text: str) -> tuple[str, bool]:
        """截断输出到输出上限（10KB），附总长度提示。"""
        if len(text) <= self._output_limit:
            return text, False
        return (
            text[: self._output_limit]
            + f"\n...[输出已截断，共 {len(text)} 字符，仅显示前 {self._output_limit}]",
            True,
        )


# 模块级单例（工具层引用）
code_sandbox = CodeSandbox()
