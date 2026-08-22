"""Skills 管理器：渐进式加载指令、按步骤执行工作流（F-M08-006）。

Skill 数据源为 DB（sys_ai_skill 主表）；skills/*.md 仅作
SkillManageService.ensure_builtin_skills 的 builtin 播种源。

为保持执行链同步签名（dehaze_tools_builder._skill_load 依赖 load_skill(name)），
采用「内存缓存 + 异步预热」模式：
- 启动时由 main.py 调用 refresh_index(db)，将启用项（status=1）的名称/描述/全文
  载入内存缓存；
- SkillManageService 每次变更（创建/更新/启停/删除）后调用 refresh_index 刷新缓存，
  使变更即时对 discover_skills/load_skill 生效；
- discover_skills()/load_skill() 仅从内存缓存同步读取（进程启动时已预热），
  不触发 DB 访问。

启动时只加载启用项的名称和描述（几十 tokens），LLM 判断需要时通过 skill_load
加载完整指令（load_skill 从内存缓存读取）。
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.ai_skill_repository import ai_skill_repository

logger = logging.getLogger(__name__)

_STATUS_ENABLED = 1


class SkillManager:
    """Skills 管理器

    - 启动时只加载启用 Skills 的名称和描述（几十 tokens）
    - LLM 判断需要某个 Skill 时，通过 skill_load 加载完整指令（从内存缓存读取）
    - Skill 指令中可引用其他工具，形成 Skills + 工具的协同
    """

    def __init__(self) -> None:
        self._skills_index: list[dict] = []
        self._loaded_skills: dict[str, str] = {}  # name -> full instruction

    async def refresh_index(self, db: AsyncSession) -> None:
        """从 DB 刷新内存索引与指令缓存（仅启用项）。

        调用时机：main.py 启动预热；SkillManageService 每次变更（创建/更新/启停/删除）后。
        多实例部署下，其他实例的缓存不会自动失效（跨进程不一致为已知限制），
        需重启或在该实例再次触发变更/预热后生效。
        """
        skills = await ai_skill_repository.list_all(db, status=_STATUS_ENABLED)
        self._skills_index = [{"name": s.name, "description": s.description} for s in skills]
        self._loaded_skills = {s.name: s.content or "" for s in skills}

    def discover_skills(self) -> list[dict]:
        """发现可用 Skills（只返回启用项的名称和描述，从内存缓存读取）"""
        return list(self._skills_index)

    def load_skill(self, name: str) -> str | None:
        """加载完整 Skill 指令（从内存缓存读取）。

        Args:
            name: Skill 名称

        Returns:
            完整指令内容，或 None 如果不存在（含禁用/未预热）
        """
        return self._loaded_skills.get(name)


skill_manager = SkillManager()
