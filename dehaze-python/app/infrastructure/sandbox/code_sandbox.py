"""受限代码执行沙箱（CodeSandbox）

安全边界（产品红线，对齐后端实现 §5.2）：
- 白名单：仅支持 python 与 shell 两种语言；shell 命令黑名单拦截破坏性操作。
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

# 破坏性命令黑名单（命中即拒绝，需走人工确认/安全流程）
_BLACKLIST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+-[rfR].*"),  # rm -r / rm -rf 递归删除
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\b"),  # 块设备级拷贝，破坏性
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bpoweroff\b"),
    re.compile(r":\s*\(\s*\)\s*\{"),  # fork 炸弹
    re.compile(r">\s*/dev/sd"),
    re.compile(r"\bchmod\s+-R\b"),
    re.compile(r"\bchown\s+-R\b"),
    re.compile(r"\bcurl\b.*\|\s*(sh|bash)\b"),  # 管道到 shell
    re.compile(r"\bwget\b.*\|\s*(sh|bash)\b"),
    re.compile(r"\bsudo\b"),
]


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

    def check_blacklist(self, command: str) -> str | None:
        """shell 命令黑名单校验，命中返回拒绝原因，未命中返回 None。"""
        for pat in _BLACKLIST_PATTERNS:
            if pat.search(command):
                return (
                    f"命令包含高危破坏性操作（匹配黑名单规则 {pat.pattern}），"
                    "已拒绝执行。如需执行请走人工确认流程。"
                )
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
            rejection = self.check_blacklist(code)
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
                "stdin": subprocess.DEVNULL,
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
                # POSIX 下独立进程组，便于超时 killpg；Windows 不支持
                "start_new_session": _HAS_RESOURCE,
            }
            try:
                process = await asyncio.create_subprocess_exec(*cmd, **kwargs)
            except Exception as e:  # noqa: BLE001
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
            except Exception as e:  # noqa: BLE001
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
            try:
                process.kill()
            except ProcessLookupError:
                pass

    @staticmethod
    async def _reap(process) -> None:
        """等待被终止的进程回收，确保 subprocess 传输在当前事件循环内完成清理。

        超时/异常路径下 communicate 被取消后，进程 transport 可能残留到循环关闭
        （触发 "Event loop is closed" ResourceWarning），此处显式回收避免告警。
        """
        try:
            await process.wait()
        except Exception:  # noqa: BLE001 - 回收失败不影响结果返回
            pass

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
