"""
WPX 文件映射数据访问层
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_wpx_file import SysWpxFile
from app.repository.base import BaseRepository


class WpxFileRepository(BaseRepository[SysWpxFile]):
    """WPX 文件映射数据访问层"""

    model = SysWpxFile

    async def get_by_origin_md5(
        self,
        db: AsyncSession,
        origin_md5: str,
    ) -> SysWpxFile | None:
        """根据原始文件 MD5 查询映射记录"""
        stmt = select(SysWpxFile).where(SysWpxFile.origin_md5 == origin_md5)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()


wpx_file_repository = WpxFileRepository()
