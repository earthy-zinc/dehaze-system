"""热词管理服务（F-VS-004）

负责用户级与全局级热词的增删查，以及供 ASR 会话注册用的生效热词合并。
"""

import html

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_voice_hotword import SysVoiceHotword
from app.models.schema.voice import HotwordForm, HotwordResult
from app.repository.voice_hotword_repository import voice_hotword_repository

# 单用户热词数量上限
_MAX_USER_HOTWORDS = 100


def _sanitize_word(raw: str) -> str:
    """热词内容清洗：去除首尾空白后 XSS 转义为纯文本存储"""
    word = raw.strip()
    if not word:
        raise BusinessException(ResultCode.PARAM_ERROR, "热词内容不能为空")
    return html.escape(word, quote=True)


def _to_results(items: list[SysVoiceHotword]) -> list[HotwordResult]:
    """实体列表转为热词响应列表"""
    return [
        HotwordResult(id=item.id, word=item.word, create_time=item.create_time)
        for item in items
    ]


class HotwordService:
    """热词管理服务"""

    @staticmethod
    async def get_effective_words(db: AsyncSession, user_id: int) -> list[str]:
        """合并全局 + 当前用户热词并去重，供 ASR 会话注册到 FunASR。

        顺序：全局热词在前，用户热词追加在后，重复内容以先出现者为准。
        """
        global_words = await voice_hotword_repository.list_by_scope(db, "global", None)
        user_words = await voice_hotword_repository.list_by_scope(db, "user", user_id)
        seen: set[str] = set()
        effective: list[str] = []
        for word in [*global_words, *user_words]:
            if word.word not in seen:
                seen.add(word.word)
                effective.append(word.word)
        return effective

    @staticmethod
    async def list_user_hotwords(db: AsyncSession, user_id: int) -> list[HotwordResult]:
        """查询当前用户的用户级热词"""
        items = await voice_hotword_repository.list_by_scope(db, "user", user_id)
        return _to_results(items)

    @staticmethod
    async def add_user_hotword(
        db: AsyncSession, user_id: int, form: HotwordForm
    ) -> HotwordResult:
        """新增用户热词（含数量上限校验、XSS 转义存储）"""
        count = await voice_hotword_repository.count_user_hotwords(db, user_id)
        if count >= _MAX_USER_HOTWORDS:
            msg = f"个人热词已达上限 {_MAX_USER_HOTWORDS} 个"
            raise BusinessException(ResultCode.BUSINESS_ERROR, msg)
        word = _sanitize_word(form.word)
        entity = SysVoiceHotword(scope="user", user_id=user_id, word=word, create_by=user_id)
        created = await voice_hotword_repository.create(db, entity)
        return HotwordResult(id=created.id, word=created.word, create_time=created.create_time)

    @staticmethod
    async def delete_user_hotword(db: AsyncSession, hotword_id: int, user_id: int) -> None:
        """删除用户热词；不存在或非本人 → A0401"""
        entity = await voice_hotword_repository.get_by_id(db, hotword_id)
        if entity is None or entity.user_id != user_id:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "热词不存在")
        await voice_hotword_repository.soft_delete(db, hotword_id)

    @staticmethod
    async def list_global_hotwords(db: AsyncSession) -> list[HotwordResult]:
        """查询全局热词（登录即可查）"""
        items = await voice_hotword_repository.list_by_scope(db, "global", None)
        return _to_results(items)

    @staticmethod
    async def add_global_hotword(
        db: AsyncSession, form: HotwordForm
    ) -> HotwordResult:
        """新增全局热词（XSS 转义存储）"""
        word = _sanitize_word(form.word)
        entity = SysVoiceHotword(scope="global", word=word)
        created = await voice_hotword_repository.create(db, entity)
        return HotwordResult(id=created.id, word=created.word, create_time=created.create_time)

    @staticmethod
    async def delete_global_hotword(db: AsyncSession, hotword_id: int) -> None:
        """删除全局热词；不存在或非全局热词 → A0401"""
        entity = await voice_hotword_repository.get_by_id(db, hotword_id)
        if entity is None or entity.scope != "global":
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "热词不存在")
        await voice_hotword_repository.soft_delete(db, hotword_id)
