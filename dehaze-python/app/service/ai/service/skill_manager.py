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

import asyncio
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.entity.sys_ai_skill_file import SysAiSkillFile
from app.repository.ai_skill_repository import ai_skill_repository

logger = logging.getLogger(__name__)

_STATUS_ENABLED = 1

# SKILL 资源对象存储 key 前缀（与 SkillManageService._SKILL_OBJECT_PREFIX 对齐）
_SKILL_OBJECT_PREFIX = "skills"


class SkillManager:
    """Skills 管理器（Agent Skills 规范三级渐进披露）

    - 第一级 元数据：启动时只加载启用 Skills 的名称和描述（几十 tokens）
    - 第二级 指令：LLM 判断需要某个 Skill 时，通过 skill_load 加载 SKILL.md 指令全文
    - 第三级 资源：reference/script/assets 文件按需加载（load_skill_file，从对象存储读）
    """

    def __init__(self) -> None:
        self._skills_index: list[dict] = []
        self._loaded_skills: dict[str, str] = {}  # name -> SKILL.md instruction
        # 文件清单缓存：name -> {path: {"size": int, "type": str}}（内容不入内存，按需读对象存储）
        self._skill_files: dict[str, dict[str, dict]] = {}

    async def refresh_index(self, db: AsyncSession) -> None:
        """从 DB 刷新内存索引与指令缓存（仅启用项）。

        调用时机：main.py 启动预热；SkillManageService 每次变更（创建/更新/启停/删除）后。
        多实例部署下，其他实例的缓存不会自动失效（跨进程不一致为已知限制），
        需重启或在该实例再次触发变更/预热后生效。
        """
        skills = await ai_skill_repository.list_all(db, status=_STATUS_ENABLED)
        self._skills_index = [{"name": s.name, "description": s.description} for s in skills]
        self._loaded_skills = {s.name: s.instruction or "" for s in skills}
        # 资源文件清单索引（渐进披露第三级）；加载失败降级为空，不影响元数据/指令加载
        try:
            if skills:
                skill_ids = [s.id for s in skills]
                stmt = select(SysAiSkillFile).where(SysAiSkillFile.skill_id.in_(skill_ids))
                rows = (await db.execute(stmt)).scalars().all()
                by_skill: dict[int, dict[str, dict]] = {}
                for f in rows:
                    by_skill.setdefault(f.skill_id, {})[f.path] = {
                        "size": f.file_size,
                        "type": f.file_type,
                    }
                id_to_name = {s.id: s.name for s in skills}
                self._skill_files = {
                    id_to_name[sid]: files
                    for sid, files in by_skill.items()
                    if sid in id_to_name
                }
            else:
                self._skill_files = {}
        except Exception:  # noqa: BLE001 - 文件索引降级，不影响核心 Skill 加载
            logger.warning("Skill 资源文件索引刷新失败", exc_info=True)
            self._skill_files = {}

    def discover_skills(self) -> list[dict]:
        """发现可用 Skills（只返回启用项的名称和描述，从内存缓存读取）"""
        return list(self._skills_index)

    def load_skill(self, name: str) -> str | None:
        """加载完整 Skill 指令（SKILL.md 正文，从内存缓存读取）。

        Args:
            name: Skill 名称

        Returns:
            SKILL.md 指令内容，或 None 如果不存在（含禁用/未预热）
        """
        return self._loaded_skills.get(name)

    def list_skill_files(self, name: str) -> list[dict]:
        """返回 SKILL 资源文件清单（path/size/type），供展示与按需加载决策。"""
        files = self._skill_files.get(name) or {}
        return [
            {"path": p, "size": meta.get("size", 0), "type": meta.get("type")}
            for p, meta in sorted(files.items())
        ]

    async def load_skill_file(self, name: str, path: str) -> str | None:
        """按需加载 SKILL 资源文件（reference/script/assets，渐进披露第三级）。

        内容存对象存储（MinIO，key=skills/{name}/{path}），按需下载返回文本。

        Args:
            name: Skill 名称
            path: 相对 SKILL 根目录的文件路径（如 "reference/REFERENCE.md"）

        Returns:
            文件文本内容，或 None 如果不存在/读取失败
        """
        if path not in (self._skill_files.get(name) or {}):
            return None
        from app.service.storage.factory import get_storage_service

        storage = get_storage_service()
        object_name = f"{_SKILL_OBJECT_PREFIX}/{name}/{path}"
        try:
            data = await asyncio.to_thread(
                storage.download, settings.MINIO_BUCKET, object_name
            )
        except Exception:  # noqa: BLE001 - 资源读取失败返回 None，不阻断推理
            logger.warning("SKILL 资源读取失败 object=%s", object_name, exc_info=True)
            return None
        return data.decode("utf-8", errors="replace")


skill_manager = SkillManager()
