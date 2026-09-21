"""Skills 管理服务（F-M08-006 Skills 管理部分）。

承担 Skill 的管理职责：列表（管理员全量/普通用户仅启用）、创建/更新（管理员权限由
路由层校验）、启停、软删（删除前校验被 Agent 关联）、内置播种。管理与执行分离：
本服务只维护 sys_ai_skill 主表，执行侧的渐进式加载由 SkillManager（skill_manager.py）承担。

每次变更（创建/更新/启停/删除）后调用 skill_manager.refresh_index 刷新内存索引，
使变更即时对 discover_skills/load_skill 生效（同进程即时；多实例跨进程失效为已知限制）。
"""

import asyncio
import io
import json
import logging
import mimetypes
import re
import uuid
import zipfile
from pathlib import Path
from typing import Any

import yaml
from redis.asyncio import Redis
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.sandbox.code_sandbox import code_sandbox
from app.models.entity.sys_ai_skill import SysAiSkill
from app.models.entity.sys_ai_skill_file import SysAiSkillFile
from app.models.schema.ai_skill import (
    DANGEROUS_PATTERN,
    SkillCreate,
    SkillFileVO,
    SkillListItem,
    SkillMarketVO,
    SkillResult,
    SkillSkippedFile,
    SkillTestForm,
    SkillUpdate,
)
from app.models.schema.common import PageResult
from app.repository.ai_skill_repository import ai_skill_repository
from app.service.storage.factory import get_storage_service

logger = logging.getLogger(__name__)

# Skill 指令内容上限（100KB）
CONTENT_MAX_BYTES = 100 * 1024

# SKILL 压缩包/单文件容量上限（防 zip 炸弹与超大资源撑爆存储/上下文）
_ZIP_MAX_BYTES = 2 * 1024 * 1024  # 压缩包上限 2MB
_SKILL_MD_MAX_BYTES = 100 * 1024  # SKILL.md 正文上限 100KB
_SKILL_FILE_MAX_BYTES = 500 * 1024  # 单资源文件上限 500KB
_ZIP_MAX_FILE_COUNT = 200  # 压缩包内文件数上限（小压缩包塞海量成员的 zip 炸弹）
_ZIP_MAX_TOTAL_BYTES = 10 * 1024 * 1024  # 解压后总量上限 10MB

# 需做内容校验的脚本资源扩展名
_SCRIPT_EXTS = (".sh", ".bash", ".py", ".js")

# SKILL 目录文件对象存储：bucket 复用平台默认桶，对象 key 前缀 skills/{skill_id}/{path}。
# key 用 skill_id 而非 name：name 可改，改名后按 name 定位的对象会与 Skill 脱钩
_SKILL_OBJECT_PREFIX = "skills"

# SKILL.name 命名规范：≤64 字符、小写字母数字、连字符分隔、不能首尾/连续连字符
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_SKILL_NAME_MAX_LEN = 64
_DESCRIPTION_MAX_LEN = 1024

# 内置播种 Skill 文件目录（仅作 builtin 播种源，文件系统不再作为运行时数据源）
_BUILTIN_SKILLS_DIR = Path(__file__).parent / "ai" / "skills"

# 编译危险操作正则（命中即抛参数异常，防止注入破坏性 shell 命令）
_DANGEROUS_RE = re.compile(DANGEROUS_PATTERN, re.IGNORECASE)


def _object_key(skill_id: int, rel_path: str) -> str:
    """资源对象 key：skills/{skill_id}/{rel_path}（与 name 解耦，改名不影响定位）。"""
    return f"{_SKILL_OBJECT_PREFIX}/{skill_id}/{rel_path}"


def _scan_script(rel_path: str, raw: bytes) -> str | None:
    """扫描资源脚本内容，命中禁用命令返回拒绝原因，安全返回 None。

    与沙箱执行策略同源：沙箱不放行的能力，打包进 Skill 只会成为死代码或绕过口子。
    shell 逐行过白名单（含解释器禁令），python 过高危能力检测（网络/子进程/沙箱外文件），
    其余脚本扩展名过与 SKILL.md 正文同源的破坏性命令正则。
    """
    if not rel_path.endswith(_SCRIPT_EXTS):
        return None
    text = raw.decode("utf-8", errors="replace")
    matched = _DANGEROUS_RE.search(text)
    if matched:
        return f"含破坏性命令（{matched.group(0).strip()}）"
    if rel_path.endswith((".sh", ".bash")):
        for line in text.splitlines():
            command = line.strip()
            if not command or command.startswith("#"):
                continue
            rejection = code_sandbox.check_command_policy(command)
            if rejection:
                return rejection
    elif rel_path.endswith(".py"):
        risk = code_sandbox.check_python_risk(text)
        if risk:
            return f"含高危能力（{risk}）"
    return None


def _normalize_zip_path(filename: str) -> str | None:
    """规范化 zip 成员路径：反斜杠转正斜杠、去重段、拒绝绝对路径与 .. 穿越。

    Returns:
        规范化相对路径；非法路径返回 None。
    """
    name = filename.replace("\\", "/")
    parts = [p for p in name.split("/") if p and p not in (".", "")]
    if not parts:
        return None
    if name.startswith("/") or ".." in parts:
        return None
    return "/".join(parts)


def _parse_skill_frontmatter(content: str) -> tuple[dict, str]:
    """解析 SKILL.md：返回 (frontmatter dict, Markdown 正文)。

    要求以 --- 开头的 YAML frontmatter，非法/未闭合抛业务异常。
    """
    if not content.lstrip().startswith("---"):
        raise BusinessException(
            ResultCode.PARAM_ERROR, "SKILL.md 缺少 YAML frontmatter（须以 --- 开头）"
        )
    lines = content.split("\n")
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        raise BusinessException(
            ResultCode.PARAM_ERROR, "SKILL.md frontmatter 未闭合（缺少结束 ---）"
        )
    fm_text = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1 :]).strip()
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as exc:
        raise BusinessException(
            ResultCode.PARAM_ERROR, f"SKILL.md frontmatter YAML 解析失败: {exc}"
        ) from exc
    if not isinstance(fm, dict):
        raise BusinessException(ResultCode.PARAM_ERROR, "SKILL.md frontmatter 格式错误")
    return fm, body


def _validate_skill_frontmatter(fm: dict, dir_name: str) -> tuple[str, str]:
    """校验 SKILL.md frontmatter（Agent Skills 规范）：name 命名/目录一致、description 必填。

    Returns:
        (name, description)
    """
    name = _str_or_none(fm.get("name"))
    if not name:
        raise BusinessException(ResultCode.PARAM_ERROR, "SKILL.md frontmatter 缺少 name")
    name = name.strip()
    if len(name) > _SKILL_NAME_MAX_LEN or not _SKILL_NAME_RE.match(name):
        raise BusinessException(
            ResultCode.PARAM_ERROR,
            "Skill name 需≤64字符且仅含小写字母/数字/连字符（不能首尾或连续连字符）",
        )
    if name != dir_name:
        raise BusinessException(
            ResultCode.PARAM_ERROR, f"SKILL.md name 必须与目录名一致（{name} != {dir_name}）"
        )
    description = _str_or_none(fm.get("description"))
    if not description or not description.strip():
        raise BusinessException(ResultCode.PARAM_ERROR, "SKILL.md frontmatter 缺少 description")
    description = description.strip()
    if len(description) > _DESCRIPTION_MAX_LEN:
        raise BusinessException(ResultCode.PARAM_ERROR, "description 不能超过 1024 字符")
    return name, description


def _str_or_none(value: Any) -> str | None:
    """标量转字符串（None/空返回 None）。"""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


_STATUS_ENABLED = 1
_STATUS_DISABLED = 0


class SkillManageService:
    def __init__(self, ai_skill_repository=ai_skill_repository):
        self.ai_skill_repository = ai_skill_repository

    async def list_skills(
        self,
        db: AsyncSession,
        *,
        enabled_only: bool,
        page: int = 1,
        size: int = 10,
        keyword: str | None = None,
        status: int | None = None,
    ) -> PageResult[SkillListItem]:
        """列表：管理员（enabled_only=False）全量含禁用（支持状态筛选）；普通用户仅启用。

        列表项不含 content 全文（渐进式加载，避免无关 Skill 挤占上下文）。
        """
        if enabled_only:
            # 普通用户仅返回启用项（不分页，直接全量）
            items = await self.ai_skill_repository.list_all(db, status=_STATUS_ENABLED)
            rows = await self._to_list_items(db, items)
            return PageResult[SkillListItem](list=rows, total=len(rows))
        # 管理员全量分页 + 名称模糊 + 状态筛选
        items, total = await self.ai_skill_repository.page(db, page, size, keyword, status)
        rows = await self._to_list_items(db, items)
        return PageResult[SkillListItem](list=rows, total=total)

    async def create_skill(self, db: AsyncSession, form: SkillCreate) -> SkillResult:
        """创建 Skill：唯一性校验 + 指令内容校验（长度上限/危险操作拦截）。

        创建后为禁用态，需管理员显式启用才可被加载（§2.6.11"启用才可被使用"）。
        """
        self._validate_content(form.instruction)
        existing = await self.ai_skill_repository.get_by_name(db, form.name)
        if existing:
            raise BusinessException(ResultCode.DATA_EXISTS, "Skill 名称已存在")

        skill = SysAiSkill(
            name=form.name,
            description=form.description,
            scene=form.scene,
            instruction=form.instruction,
            status=_STATUS_DISABLED,
            source="admin",
            market_shared=0,
        )
        await self.ai_skill_repository.create(db, skill)
        await self._refresh_index(db)
        return await self._to_detail(db, skill)

    async def create_skill_from_zip(self, db: AsyncSession, zip_bytes: bytes) -> SkillResult:
        """zip 压缩包上传创建 SKILL（遵循 Agent Skills 规范）。

        流程：安全解压 → 定位 SKILL.md → 解析 YAML frontmatter → 校验
        （name 命名规范、name 与目录名一致、description 必填、危险操作、容量）
        → 元数据与 SKILL.md 正文入库 sys_ai_skill，其余文件（reference/script/assets）
        入库 sys_ai_skill_file → 刷新 SkillManager 索引。
        """
        if not zip_bytes:
            raise BusinessException(ResultCode.PARAM_ERROR, "请上传 zip 压缩包")
        if len(zip_bytes) > _ZIP_MAX_BYTES:
            raise BusinessException(ResultCode.PARAM_ERROR, "SKILL 压缩包超过 2MB 上限")
        try:
            zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        except zipfile.BadZipFile:
            raise BusinessException(
                ResultCode.PARAM_ERROR, "上传文件不是有效的 zip 压缩包"
            ) from None

        # 收集成员：防路径穿越（规范化路径，禁止 .. 与绝对路径）+ 防 zip 炸弹
        # （文件数、解压后总量双上限，在读取任何成员内容前判定）
        entries: dict[str, zipfile.ZipInfo] = {}
        total_bytes = 0
        for info in zf.infolist():
            if info.is_dir():
                continue
            norm = _normalize_zip_path(info.filename)
            if norm is None:
                raise BusinessException(ResultCode.PARAM_ERROR, "压缩包包含非法路径")
            if norm in entries:
                continue
            if len(entries) >= _ZIP_MAX_FILE_COUNT:
                raise BusinessException(
                    ResultCode.PARAM_ERROR, f"压缩包文件数超过 {_ZIP_MAX_FILE_COUNT} 上限"
                )
            total_bytes += info.file_size
            if total_bytes > _ZIP_MAX_TOTAL_BYTES:
                raise BusinessException(
                    ResultCode.PARAM_ERROR,
                    f"压缩包解压后总量超过 {_ZIP_MAX_TOTAL_BYTES // 1024 // 1024}MB 上限",
                )
            entries[norm] = info

        # 定位 SKILL.md：优先最浅（顶层目录下），要求 SKILL.md 在顶层目录内
        skill_md_candidates = [p for p in entries if p.endswith("/SKILL.md")]
        if not skill_md_candidates:
            raise BusinessException(
                ResultCode.PARAM_ERROR, "压缩包中未找到 SKILL.md（需位于顶层目录下）"
            )
        skill_md_path = min(skill_md_candidates, key=lambda p: p.count("/"))
        base_dir, _ = skill_md_path.rsplit("/", 1)
        dir_name = base_dir.split("/")[-1] if base_dir else ""
        if not dir_name:
            raise BusinessException(
                ResultCode.PARAM_ERROR, "SKILL.md 必须位于顶层目录下（目录名 = skill name）"
            )

        try:
            skill_md_raw = zf.read(skill_md_path)
        except KeyError:
            raise BusinessException(ResultCode.PARAM_ERROR, "SKILL.md 读取失败") from None
        if len(skill_md_raw) > _SKILL_MD_MAX_BYTES:
            raise BusinessException(ResultCode.PARAM_ERROR, "SKILL.md 超过 100KB 上限")
        try:
            skill_md_text = skill_md_raw.decode("utf-8")
        except UnicodeDecodeError:
            raise BusinessException(ResultCode.PARAM_ERROR, "SKILL.md 必须为 UTF-8 编码") from None

        frontmatter, body = _parse_skill_frontmatter(skill_md_text)
        name, description = _validate_skill_frontmatter(frontmatter, dir_name)
        self._validate_content(body)

        # 资源文件：先读出全部内容并完成脚本校验，再落库——校验失败时无任何写入
        payloads: dict[str, bytes] = {}
        skipped: list[SkillSkippedFile] = []
        for path, info in entries.items():
            if path == skill_md_path:
                continue
            if not path.startswith(base_dir + "/"):
                skipped.append(SkillSkippedFile(path=path, reason="不在 SKILL 目录内"))
                continue
            try:
                raw = zf.read(info)
            except (KeyError, zipfile.BadZipFile, RuntimeError, EOFError) as exc:
                raise BusinessException(
                    ResultCode.PARAM_ERROR, f"资源文件读取失败: {path}"
                ) from exc
            rel = path[len(base_dir) + 1 :]
            if len(raw) > _SKILL_FILE_MAX_BYTES:
                skipped.append(
                    SkillSkippedFile(
                        path=rel,
                        reason=f"超过单文件 {_SKILL_FILE_MAX_BYTES // 1024}KB 上限",
                    )
                )
                continue
            payloads[rel] = raw

        # 脚本内容校验：命中即整包拒绝，不允许"部分带毒"入库
        rejected: list[str] = []
        for rel, raw in payloads.items():
            reason = _scan_script(rel, raw)
            if reason:
                rejected.append(f"{rel}: {reason}")
        if rejected:
            raise BusinessException(
                ResultCode.PARAM_ERROR,
                "SKILL 资源脚本含被禁命令，整包拒绝：" + "; ".join(rejected),
            )

        existing = await self.ai_skill_repository.get_by_name(db, name)
        if existing:
            raise BusinessException(ResultCode.DATA_EXISTS, "Skill 名称已存在")

        skill = SysAiSkill(
            name=name,
            description=description,
            scene="",
            instruction=body,
            license=_str_or_none(frontmatter.get("license")),
            compatibility=_str_or_none(frontmatter.get("compatibility")),
            skill_metadata=frontmatter.get("metadata")
            if isinstance(frontmatter.get("metadata"), dict)
            else None,
            allowed_tools=_str_or_none(frontmatter.get("allowed-tools")),
            status=_STATUS_DISABLED,
            source="admin",
            market_shared=0,
        )
        await self.ai_skill_repository.create(db, skill)
        await db.flush()

        # 资源文件内容传对象存储（对象 key=skills/{skill_id}/{path}），DB 只存清单
        # （path/size/type）：支持二进制资源与按需加载（业界 Agent Skills 目录语义）
        storage = get_storage_service()
        bucket = settings.MINIO_BUCKET
        await asyncio.to_thread(storage.ensure_bucket, bucket)

        files: list[SysAiSkillFile] = []
        for rel, raw in payloads.items():
            content_type = mimetypes.guess_type(rel)[0] or "application/octet-stream"
            try:
                await asyncio.to_thread(
                    storage.upload, bucket, _object_key(skill.id, rel), raw, content_type
                )
            except Exception as exc:
                raise BusinessException(
                    ResultCode.BUSINESS_ERROR, f"SKILL 资源写入对象存储失败: {rel}"
                ) from exc
            files.append(
                SysAiSkillFile(
                    skill_id=skill.id,
                    path=rel,
                    file_size=len(raw),
                    file_type=content_type,
                )
            )
        if files:
            db.add_all(files)
            await db.flush()

        await self._refresh_index(db)
        return await self._to_detail(db, skill, skipped)

    async def update_skill(self, db: AsyncSession, skill_id: int, form: SkillUpdate) -> SkillResult:
        """更新 Skill：同样做内容校验；name 变更时校验唯一性。更新后新会话生效。"""
        skill = await self._get_or_404(db, skill_id)

        if form.name is not None and form.name != skill.name:
            duplicate = await self.ai_skill_repository.get_by_name(db, form.name)
            if duplicate and duplicate.id != skill_id:
                raise BusinessException(ResultCode.DATA_EXISTS, "Skill 名称已存在")
        if form.instruction is not None:
            self._validate_content(form.instruction)

        data: dict[str, Any] = {}
        if form.name is not None:
            data["name"] = form.name
        if form.description is not None:
            data["description"] = form.description
        if form.instruction is not None:
            data["instruction"] = form.instruction
        if form.scene is not None:
            data["scene"] = form.scene
        await self.ai_skill_repository.update(db, skill, data)
        await self._refresh_index(db)
        return await self._to_detail(db, skill)

    async def set_status(self, db: AsyncSession, skill_id: int, enabled: bool) -> SkillResult:
        """启停 Skill：禁用后不出现在 SkillManager 索引（discover/load 均不可见），
        返回更新后详情。"""
        skill = await self._get_or_404(db, skill_id)
        target = _STATUS_ENABLED if enabled else _STATUS_DISABLED
        if skill.status != target:
            await self.ai_skill_repository.update(db, skill, {"status": target})
            await self._refresh_index(db)
        return await self._to_detail(db, skill)

    async def delete_skill(self, db: AsyncSession, skill_id: int) -> None:
        """软删 Skill；删除前校验是否被 Agent 关联，有则提示先解绑。"""
        skill = await self._get_or_404(db, skill_id)
        refs = await self.ai_skill_repository.count_agent_references(db, skill.name)
        if refs > 0:
            raise BusinessException(
                ResultCode.DATA_BIND_EXISTS,
                f"Skill [{skill.name}] 已被 {refs} 个 Agent 关联，请先解绑再删除",
            )
        await self.ai_skill_repository.soft_delete_by_ids(db, [skill_id])
        await self._delete_skill_objects(db, skill)
        await self._refresh_index(db)

    async def _delete_skill_objects(self, db: AsyncSession, skill: SysAiSkill) -> None:
        """删除 Skill 的对象存储资源与文件清单（软删主表后清理）。"""

        storage = get_storage_service()
        bucket = settings.MINIO_BUCKET
        prefix = _object_key(skill.id, "")
        try:
            objects = await asyncio.to_thread(storage.list_objects, bucket, prefix)
            for object_name, _ in objects:
                await asyncio.to_thread(storage.delete, bucket, object_name)
        except Exception:
            logger.warning("SKILL 对象存储清理失败 prefix=%s", prefix, exc_info=True)
        try:
            stmt = delete(SysAiSkillFile).where(SysAiSkillFile.skill_id == skill.id)
            await db.execute(stmt)
        except Exception:
            logger.warning("SKILL 文件清单清理失败 skill_id=%s", skill.id, exc_info=True)

    async def get_skill(
        self, db: AsyncSession, skill_id: int, *, enabled_only: bool = False
    ) -> SkillResult:
        """Skill 详情（含指令全文）。

        enabled_only=True（普通用户）时禁用项按不存在处理，与列表"仅启用"口径一致，
        防止经 ID 直读禁用 Skill 的指令全文。
        """
        skill = await self._get_or_404(db, skill_id, enabled_only=enabled_only)
        return await self._to_detail(db, skill)

    async def get_skill_file(
        self, db: AsyncSession, skill_id: int, path: str, *, enabled_only: bool = False
    ) -> bytes:
        """读取 SKILL 资源文件内容（从对象存储下载）。

        path 必须先命中该 Skill 的文件清单，防止任意对象读取。
        """
        skill = await self._get_or_404(db, skill_id, enabled_only=enabled_only)
        files = await self._list_skill_files(db, skill.id)
        if not any(f.path == path for f in files):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "SKILL 文件不存在")

        storage = get_storage_service()
        object_name = _object_key(skill.id, path)
        try:
            return await asyncio.to_thread(storage.download, settings.MINIO_BUCKET, object_name)
        except Exception as exc:
            logger.warning("SKILL 文件读取失败 object=%s: %s", object_name, exc)
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "SKILL 文件读取失败") from exc

    async def test_skill(
        self, db: AsyncSession, redis: Redis, skill_id: int, form: SkillTestForm
    ) -> dict:
        """试运行 Skill：以指令为系统提示词跑一次最小推理，返回模型输出与用量。

        复用 Agent 测试预览通路（DeepAgentBuilder + 独立 thread_id），不落库、
        不推送、不污染生产会话；模型取平台默认模型，配置取系统默认（无 Agent 级覆盖）。
        """
        skill = await self._get_or_404(db, skill_id)
        if skill.status != _STATUS_ENABLED:
            raise BusinessException(ResultCode.PARAM_ERROR, "Skill 已禁用，无法试运行")
        if form.inputData is None:
            raise BusinessException(ResultCode.PARAM_ERROR, "试运行需提供测试输入")

        from app.service.ai.builders.deep_agent_builder import DeepAgentBuilder
        from app.service.ai.strategies.agent_config_resolver import resolve

        model_id = settings.AI_DEFAULT_MODEL
        config = await resolve(db, redis, None, None)
        graph = await DeepAgentBuilder().build_from_snapshot(
            db,
            redis,
            {
                "name": skill.name,
                "description": skill.description,
                "system_prompt": skill.instruction or "",
                "model_id": model_id,
                "config": config,
                "mcp_namespaces": [],
                "subagents": [],
                "skills": [],
                "permissions": None,
            },
        )
        message = (
            form.inputData
            if isinstance(form.inputData, str)
            else json.dumps(form.inputData, ensure_ascii=False, default=str)
        )
        result = await graph.ainvoke(
            {
                "messages": [{"role": "user", "content": message}],
                "user_id": None,
                "conversation_id": 0,
                "message_id": 0,
                "model_id": model_id,
                "system_prompt": skill.instruction,
                "stream_session_id": f"skill-test:{uuid.uuid4()}",
                "step_count": 0,
                "token_used": 0,
                "token_budget": config.get("token_budget", 0),
                "thoughts": [],
                "isolated_token_pool": True,
            },
            config={"configurable": {"thread_id": f"skill-test:{skill_id}:{uuid.uuid4()}"}},
        )
        return {
            "skillId": skill.id,
            "skillName": skill.name,
            "instruction": skill.instruction or "",
            "input": form.inputData,
            "output": result.get("final_response", ""),
            "usage": result.get("usage") or {},
        }

    async def list_market(self, db: AsyncSession) -> list[SkillMarketVO]:
        """Skill 市场目录：返回已共享（market_shared=1）的启用项及被 Agent 关联数。"""
        items = await self.ai_skill_repository.list_market_shared(db)
        rows = []
        for s in items:
            refs = await self.ai_skill_repository.count_agent_references(db, s.name)
            rows.append(
                SkillMarketVO(
                    skillId=s.id,
                    name=s.name,
                    description=s.description,
                    scene=s.scene or "",
                    enabled=s.status == _STATUS_ENABLED,
                    agentCount=refs,
                )
            )
        return rows

    async def share_to_market(self, db: AsyncSession, skill_id: int) -> SkillResult:
        """共享 Skill 至市场（需已启用，幂等：重复共享仍返回当前状态）。"""
        skill = await self._get_or_404(db, skill_id)
        if skill.status != _STATUS_ENABLED:
            raise BusinessException(ResultCode.PARAM_ERROR, "Skill 需先启用才能共享至市场")
        if skill.market_shared != 1:
            await self.ai_skill_repository.update(db, skill, {"market_shared": 1})
        return await self._to_detail(db, skill)

    async def ensure_builtin_skills(self, db: AsyncSession) -> None:
        """内置播种：将 skills/*.md 文件内容迁入 DB（source=builtin，name 不存在才插入）。"""
        if not _BUILTIN_SKILLS_DIR.exists():
            return
        for md_file in sorted(_BUILTIN_SKILLS_DIR.glob("*.md")):
            name = md_file.stem
            existing = await self.ai_skill_repository.get_by_name(db, name)
            if existing:
                continue
            content = md_file.read_text(encoding="utf-8")
            description = self._extract_builtin_description(content)
            skill = SysAiSkill(
                name=name,
                description=description,
                instruction=content,
                status=_STATUS_ENABLED,
                source="builtin",
                market_shared=0,
            )
            await self.ai_skill_repository.create(db, skill)
            logger.info("内置 Skill 播种完成: name=%s", name)

    async def migrate_legacy_object_keys(self, db: AsyncSession) -> int:
        """把 name-key 的历史资源对象迁到 skill_id-key（一次性，幂等）。

        旧对象迁完即删，空前缀直接跳过，重复执行无副作用；改名不再让资源脱钩。
        返回迁移对象数。
        """
        storage = get_storage_service()
        bucket = settings.MINIO_BUCKET
        await asyncio.to_thread(storage.ensure_bucket, bucket)
        moved = 0
        for skill in await self.ai_skill_repository.list_all(db):
            legacy_prefix = f"{_SKILL_OBJECT_PREFIX}/{skill.name}/"
            objects = await asyncio.to_thread(storage.list_objects, bucket, legacy_prefix)
            for object_name, _ in objects:
                rel = object_name[len(legacy_prefix) :]
                target = _object_key(skill.id, rel)
                # name 恰为纯数字且等于 id 时新旧 key 重合，迁移会自己覆盖自己再删除
                if target == object_name:
                    continue
                raw = await asyncio.to_thread(storage.download, bucket, object_name)
                content_type = mimetypes.guess_type(rel)[0] or "application/octet-stream"
                await asyncio.to_thread(storage.upload, bucket, target, raw, content_type)
                await asyncio.to_thread(storage.delete, bucket, object_name)
                moved += 1
        if moved:
            logger.info("SKILL 资源对象 key 迁移完成: %s 个", moved)
        return moved

    # ── 内部工具 ──────────────────────────────────────────

    def _validate_content(self, content: str) -> None:
        """指令内容校验：长度上限（100KB）+ 危险操作拦截。"""
        if len(content.encode("utf-8")) > CONTENT_MAX_BYTES:
            raise BusinessException(ResultCode.PARAM_ERROR, "Skill 指令内容超过 100KB 上限")
        if _DANGEROUS_RE.search(content):
            raise BusinessException(ResultCode.PARAM_ERROR, "指令含危险操作")

    def _extract_builtin_description(self, content: str) -> str:
        """从 Markdown 提取描述（第一段非标题文本），截取前 500 字符。"""
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("---"):
                return line[:500]
        return ""

    async def _get_or_404(
        self, db: AsyncSession, skill_id: int, *, enabled_only: bool = False
    ) -> SysAiSkill:
        skill = await self.ai_skill_repository.get_by_id(db, skill_id)
        if not skill or (enabled_only and skill.status != _STATUS_ENABLED):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "Skill 不存在")
        return skill

    async def _refresh_index(self, db: AsyncSession) -> None:
        """变更后刷新 SkillManager 内存索引，使执行侧即时可见。"""
        from app.service.ai.service.skill_manager import skill_manager

        await skill_manager.refresh_index(db)

    async def _to_detail(
        self,
        db: AsyncSession,
        skill: SysAiSkill,
        skipped: list[SkillSkippedFile] | None = None,
    ) -> SkillResult:
        refs = await self.ai_skill_repository.count_agent_references(db, skill.name)
        files = await self._list_skill_files(db, skill.id)
        return SkillResult(
            **self._base_dict(skill),
            instruction=skill.instruction,
            license=skill.license,
            compatibility=skill.compatibility,
            metadata=skill.skill_metadata,
            allowedTools=skill.allowed_tools,
            files=[
                SkillFileVO(path=f.path, fileSize=f.file_size, fileType=f.file_type) for f in files
            ],
            skippedFiles=skipped or [],
            agentCount=refs,
        )

    async def _list_skill_files(self, db: AsyncSession, skill_id: int) -> list[SysAiSkillFile]:
        """查询 SKILL 目录内资源文件（按 path 排序）。

        文件列表是详情增强信息，查询失败（如文件表未就绪/测试桩无 execute）
        降级返回空列表，不阻断详情返回。
        """
        try:
            stmt = (
                select(SysAiSkillFile)
                .where(SysAiSkillFile.skill_id == skill_id)
                .order_by(SysAiSkillFile.path.asc())
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())
        except Exception:
            logger.warning("查询 Skill 文件列表失败 skill_id=%s", skill_id, exc_info=True)
            return []

    async def _to_list_items(
        self, db: AsyncSession, skills: list[SysAiSkill]
    ) -> list[SkillListItem]:
        """列表项装配：批量聚合被 Agent 关联数（避免逐条子查询）。"""
        refs = await self.ai_skill_repository.count_agent_references_by_names(
            db, [s.name for s in skills]
        )
        return [
            SkillListItem(
                **self._base_dict(s),
                agentCount=refs.get(s.name, 0),
            )
            for s in skills
        ]

    def _base_dict(self, skill: SysAiSkill) -> dict[str, Any]:
        """公共字段装配（键名对齐 schema 字段名 camelCase）。"""
        return {
            "id": skill.id,
            "name": skill.name,
            "description": skill.description,
            "scene": skill.scene,
            "status": skill.status,
            "source": skill.source,
            "marketShared": skill.market_shared,
            "createTime": skill.create_time,
            "updateTime": skill.update_time,
        }


skill_manage_service = SkillManageService()
