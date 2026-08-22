"""
字典数据访问层

设计参照: dehaze-doc/docs/03-模块设计/基础模块/字典管理/后端实现.md
"""

from typing import Any

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_dict import SysDict, SysDictType
from app.repository.base import BaseRepository, escape_like


class DictRepository(BaseRepository[SysDict]):
    """字典数据访问层"""

    model = SysDict

    async def get_page(
        self,
        db: AsyncSession,
        page: int,
        page_size: int,
        keywords: str | None = None,
        type_code: str | None = None,
    ) -> tuple[list[SysDict], int]:
        """获取字典分页列表"""
        stmt = select(SysDict)

        if keywords:
            stmt = stmt.where(SysDict.name.like(f"%{escape_like(keywords)}%", escape="\\"))

        if type_code:
            stmt = stmt.where(SysDict.type_code == type_code)

        # 查询总数
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await db.execute(count_stmt)
        total = total_result.scalar()

        # 排序: sort ASC, create_time DESC, id ASC (id 作为 tiebreaker 保证分页确定性)
        stmt = stmt.order_by(SysDict.sort.asc(), SysDict.create_time.desc(), SysDict.id.asc())

        # 分页查询
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        items = result.scalars().all()

        return list(items), total or 0

    async def get_form_by_id(self, db: AsyncSession, dict_id: int) -> dict[str, Any] | None:
        """获取字典表单数据"""
        stmt = select(SysDict).where(SysDict.id == dict_id)
        result = await db.execute(stmt)
        item = result.scalar_one_or_none()

        if not item:
            return None

        return {
            "id": item.id,
            "typeCode": item.type_code,
            "name": item.name,
            "value": item.value,
            "status": item.status,
            "defaulted": item.defaulted,
            "sort": item.sort,
            "remark": item.remark,
        }

    async def get_by_type_code_and_value(
        self, db: AsyncSession, type_code: str, value: str
    ) -> SysDict | None:
        """根据类型编码和值查询字典项"""
        stmt = select(SysDict).where(
            and_(
                SysDict.type_code == type_code,
                SysDict.value == value,
            )
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_type_code_and_name(
        self, db: AsyncSession, type_code: str, name: str
    ) -> SysDict | None:
        """根据类型编码和名称查询字典项（幂等种子按 name 判重用）"""
        stmt = select(SysDict).where(
            and_(
                SysDict.type_code == type_code,
                SysDict.name == name,
            )
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_enabled_by_type_code(self, db: AsyncSession, type_code: str) -> list[SysDict]:
        """列出某类型下的全部启用字典项（AI 配置默认值/健康阈值等批量读取用）"""
        stmt = select(SysDict).where(SysDict.type_code == type_code, SysDict.status == 1)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def count_by_type_codes(self, db: AsyncSession, type_codes: list[str]) -> dict[str, int]:
        """批量统计多个类型下的字典数据数量（避免 N+1）"""
        if not type_codes:
            return {}
        stmt = (
            select(SysDict.type_code, func.count().label("cnt"))
            .where(SysDict.type_code.in_(type_codes))
            .group_by(SysDict.type_code)
        )
        result = await db.execute(stmt)
        return {str(row.type_code): int(row.cnt) for row in result}

    async def count_by_ids(self, db: AsyncSession, dict_ids: list[int]) -> int:
        """根据ID列表统计存在的字典数量"""
        if not dict_ids:
            return 0
        stmt = select(func.count()).where(SysDict.id.in_(dict_ids))
        result = await db.execute(stmt)
        return result.scalar() or 0

    async def delete_by_type_codes(self, db: AsyncSession, type_codes: list[str]) -> int:
        """根据类型编码列表批量删除字典数据"""
        if not type_codes:
            return 0
        stmt = delete(SysDict).where(SysDict.type_code.in_(type_codes))
        result = await db.execute(stmt)
        return result.rowcount

    async def get_type_codes_by_ids(self, db: AsyncSession, dict_ids: list[int]) -> list[str]:
        """根据ID列表获取对应的类型编码列表"""
        stmt = select(SysDict.type_code).where(SysDict.id.in_(dict_ids))
        result = await db.execute(stmt)
        return [row[0] for row in result.fetchall() if row[0]]

    async def create_dict(self, db: AsyncSession, data: dict[str, Any]) -> SysDict:
        """创建字典项"""
        dict_item = SysDict(
            type_code=data.get("typeCode"),
            name=data.get("name"),
            value=data.get("value"),
            status=data.get("status", 1),
            defaulted=data.get("defaulted", 0),
            sort=data.get("sort", 1),
            remark=data.get("remark", ""),
        )

        db.add(dict_item)
        await db.flush()
        await db.refresh(dict_item)
        return dict_item

    async def update_by_id(self, db: AsyncSession, dict_id: int, data: dict[str, Any]) -> bool:
        """更新字典项"""
        stmt = select(SysDict).where(SysDict.id == dict_id)
        result = await db.execute(stmt)
        dict_item = result.scalar_one_or_none()

        if not dict_item:
            return False

        if "typeCode" in data:
            dict_item.type_code = data["typeCode"]
        if "name" in data:
            dict_item.name = data["name"]
        if "value" in data:
            dict_item.value = data["value"]
        if "status" in data:
            dict_item.status = data["status"]
        if "defaulted" in data:
            dict_item.defaulted = data["defaulted"]
        if "sort" in data:
            dict_item.sort = data["sort"]
        if "remark" in data:
            dict_item.remark = data["remark"]

        await db.flush()
        return True

    async def list_options_by_type(
        self,
        db: AsyncSession,
        type_code: str,
    ) -> list[dict]:
        """
        根据类型编码获取字典下拉选项

        业务规则（T-DM-060/062）：
        - 仅返回启用状态（status=1）的字典项
        - 字典类型被禁用（status=0）时，其下拉选项整体不返回
        排序规则: sort ASC, create_time DESC
        """
        stmt = (
            select(SysDict)
            .join(SysDictType, SysDict.type_code == SysDictType.code)
            .where(
                SysDict.type_code == type_code,
                SysDict.status == 1,
                SysDictType.status == 1,
            )
            .order_by(SysDict.sort.asc(), SysDict.create_time.desc())
        )
        result = await db.execute(stmt)
        items = result.scalars().all()
        return [{"value": item.value, "label": item.name} for item in items]


class DictTypeRepository(BaseRepository[SysDictType]):
    """字典类型数据访问层"""

    model = SysDictType

    async def get_page(
        self,
        db: AsyncSession,
        page: int,
        page_size: int,
        keywords: str | None = None,
    ) -> tuple[list[SysDictType], int]:
        """获取字典类型分页列表"""
        stmt = select(SysDictType)

        if keywords:
            stmt = stmt.where(
                or_(
                    SysDictType.name.like(f"%{escape_like(keywords)}%", escape="\\"),
                    SysDictType.code.like(f"%{escape_like(keywords)}%", escape="\\"),
                )
            )

        # 查询总数
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await db.execute(count_stmt)
        total = total_result.scalar()

        # 排序: create_time DESC, id ASC (id 作为 tiebreaker 保证分页确定性)
        stmt = stmt.order_by(SysDictType.create_time.desc(), SysDictType.id.asc())

        # 分页查询
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        items = result.scalars().all()

        return list(items), total or 0

    async def get_form_by_id(self, db: AsyncSession, type_id: int) -> dict[str, Any] | None:
        """获取字典类型表单数据"""
        stmt = select(SysDictType).where(SysDictType.id == type_id)
        result = await db.execute(stmt)
        item = result.scalar_one_or_none()

        if not item:
            return None

        return {
            "id": item.id,
            "name": item.name,
            "code": item.code,
            "status": item.status,
            "remark": item.remark,
        }

    async def get_by_code(self, db: AsyncSession, code: str) -> SysDictType | None:
        """根据编码查询字典类型"""
        stmt = select(SysDictType).where(SysDictType.code == code)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def create_type(self, db: AsyncSession, data: dict[str, Any]) -> SysDictType:
        """创建字典类型"""
        dict_type = SysDictType(
            name=data.get("name"),
            code=data.get("code"),
            status=data.get("status", 1),
            remark=data.get("remark", ""),
        )

        db.add(dict_type)
        await db.flush()
        await db.refresh(dict_type)
        return dict_type

    async def update_by_id(self, db: AsyncSession, type_id: int, data: dict[str, Any]) -> bool:
        """更新字典类型"""
        stmt = select(SysDictType).where(SysDictType.id == type_id)
        result = await db.execute(stmt)
        dict_type = result.scalar_one_or_none()

        if not dict_type:
            return False

        if "name" in data:
            dict_type.name = data["name"]
        if "code" in data:
            dict_type.code = data["code"]
        if "status" in data:
            dict_type.status = data["status"]
        if "remark" in data:
            dict_type.remark = data["remark"]

        await db.flush()
        return True

    async def count_by_ids(self, db: AsyncSession, type_ids: list[int]) -> int:
        """根据ID列表统计存在的字典类型数量"""
        if not type_ids:
            return 0
        stmt = select(func.count()).where(SysDictType.id.in_(type_ids))
        result = await db.execute(stmt)
        return result.scalar() or 0


# 单例
dict_repository = DictRepository()
dict_type_repository = DictTypeRepository()
