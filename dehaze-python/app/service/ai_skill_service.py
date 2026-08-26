"""Skills 管理服务（F-M08-006 Skills 管理部分）。

承担 Skill 的管理职责：列表（管理员全量/普通用户仅启用）、创建/更新（管理员权限由
路由层校验）、启停、软删（删除前校验被 Agent 关联）、内置播种。管理与执行分离：
本服务只维护 sys_ai_skill 主表，执行侧的渐进式加载由 SkillManager（skill_manager.py）承担。

每次变更（创建/更新/启停/删除）后调用 skill_manager.refresh_index 刷新内存索引，
使变更即时对 discover_skills/load_skill 生效（同进程即时；多实例跨进程失效为已知限制）。
"""

import logging
import re
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_skill import SysAiSkill
from app.models.schema.ai_skill import (
    DANGEROUS_PATTERN,
    SkillCreate,
    SkillListItem,
    SkillMarketVO,
    SkillResult,
    SkillTestForm,
    SkillUpdate,
)
from app.models.schema.common import PageResult
from app.repository.ai_skill_repository import ai_skill_repository

logger = logging.getLogger(__name__)

# Skill 指令内容上限（100KB）
CONTENT_MAX_BYTES = 100 * 1024

# 内置播种 Skill 文件目录（仅作 builtin 播种源，文件系统不再作为运行时数据源）
_BUILTIN_SKILLS_DIR = Path(__file__).parent / "ai" / "skills"

# 编译危险操作正则（命中即抛参数异常，防止注入破坏性 shell 命令）
_DANGEROUS_RE = re.compile(DANGEROUS_PATTERN, re.IGNORECASE)

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
    ) -> PageResult[SkillListItem]:
        """列表：管理员（enabled_only=False）全量含禁用；普通用户（enabled_only=True）仅启用。

        列表项不含 content 全文（渐进式加载，避免无关 Skill 挤占上下文）。
        """
        if enabled_only:
            # 普通用户仅返回启用项（不分页，直接全量）
            items = await self.ai_skill_repository.list_all(db, status=_STATUS_ENABLED)
            rows = [self._to_list_item(s) for s in items]
            return PageResult[SkillListItem](list=rows, total=len(rows))
        # 管理员全量分页 + 名称模糊
        items, total = await self.ai_skill_repository.page(db, page, size, keyword)
        rows = [self._to_list_item(s) for s in items]
        return PageResult[SkillListItem](list=rows, total=total)

    async def create_skill(self, db: AsyncSession, form: SkillCreate) -> SkillResult:
        """创建 Skill：唯一性校验 + 指令内容校验（长度上限/危险操作拦截）。"""
        self._validate_content(form.instruction)
        existing = await self.ai_skill_repository.get_by_name_with_deleted(db, form.name)
        if existing:
            raise BusinessException(ResultCode.DATA_EXISTS, "Skill 名称已存在")

        skill = SysAiSkill(
            name=form.name,
            description=form.description,
            instruction=form.instruction,
            status=_STATUS_ENABLED,
            source="admin",
            market_shared=0,
        )
        await self.ai_skill_repository.create(db, skill)
        await self._refresh_index(db)
        return await self._to_detail(db, skill)

    async def update_skill(self, db: AsyncSession, skill_id: int, form: SkillUpdate) -> SkillResult:
        """更新 Skill：同样做内容校验；name 变更时校验唯一性。更新后新会话生效。"""
        skill = await self._get_or_404(db, skill_id)

        if form.name is not None and form.name != skill.name:
            duplicate = await self.ai_skill_repository.get_by_name_with_deleted(db, form.name)
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
        await self.ai_skill_repository.update(db, skill, data)
        await self._refresh_index(db)
        return await self._to_detail(db, skill)

    async def set_status(self, db: AsyncSession, skill_id: int, enabled: bool) -> SkillResult:
        """启停 Skill：禁用后不出现在 SkillManager 索引（discover/load 均不可见），返回更新后详情。"""
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
        await self._refresh_index(db)

    async def get_skill(self, db: AsyncSession, skill_id: int) -> SkillResult:
        """Skill 详情（含指令全文）。"""
        skill = await self._get_or_404(db, skill_id)
        return await self._to_detail(db, skill)

    async def test_skill(self, db: AsyncSession, skill_id: int, form: SkillTestForm) -> dict:
        """试运行 Skill：构造测试会话预览指令执行效果，不入库不推送。

        试运行不进入完整推理链路（避免真实 LLM 推理的成本与不确定性），仅将
        Skill 指令作为系统上下文与测试输入组装为一次性测试会话返回，供前端预览。
        """
        skill = await self._get_or_404(db, skill_id)
        if skill.status != _STATUS_ENABLED:
            raise BusinessException(ResultCode.PARAM_ERROR, "Skill 已禁用，无法试运行")
        return {
            "skillId": skill.id,
            "skillName": skill.name,
            "instruction": skill.instruction or "",
            "input": form.inputData,
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

    async def _get_or_404(self, db: AsyncSession, skill_id: int) -> SysAiSkill:
        skill = await self.ai_skill_repository.get_by_id(db, skill_id)
        if not skill:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "Skill 不存在")
        return skill

    async def _refresh_index(self, db: AsyncSession) -> None:
        """变更后刷新 SkillManager 内存索引，使执行侧即时可见。"""
        from app.service.ai.service.skill_manager import skill_manager

        await skill_manager.refresh_index(db)

    async def _to_detail(self, db: AsyncSession, skill: SysAiSkill) -> SkillResult:
        refs = await self.ai_skill_repository.count_agent_references(db, skill.name)
        return SkillResult(
            **self._base_dict(skill),
            instruction=skill.instruction,
            agentCount=refs,
        )

    def _to_list_item(self, skill: SysAiSkill) -> SkillListItem:
        return SkillListItem(**self._base_dict(skill))

    def _base_dict(self, skill: SysAiSkill) -> dict[str, Any]:
        """公共字段装配（键名对齐 schema 字段名 camelCase）。"""
        return {
            "id": skill.id,
            "name": skill.name,
            "description": skill.description,
            "status": skill.status,
            "source": skill.source,
            "marketShared": skill.market_shared,
            "createTime": skill.create_time,
            "updateTime": skill.update_time,
        }


skill_manage_service = SkillManageService()
