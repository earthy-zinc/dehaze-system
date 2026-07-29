import hashlib
import secrets
from datetime import datetime

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.dependencies.auth import UserContext
from app.models.entity.api_key import SysApiKey
from app.repository.user_repository import user_repository


class ApiKeyService:
    @staticmethod
    async def create_api_key(
        db: AsyncSession,
        user_id: int,
        name: str,
        expires_at: datetime | None = None,
    ) -> dict:
        raw_key = "dhak_" + secrets.token_urlsafe(36)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        entity = SysApiKey(
            user_id=user_id,
            name=name,
            key_prefix=raw_key[:12],
            key_hash=key_hash,
            status=1,
            expires_at=expires_at,
        )
        db.add(entity)
        await db.flush()
        await db.refresh(entity)
        return {
            "id": entity.id,
            "name": entity.name,
            "apiKey": raw_key,
            "keyPrefix": entity.key_prefix,
            "status": entity.status,
            "expiresAt": entity.expires_at,
            "lastUsedAt": entity.last_used_at,
            "createTime": entity.create_time,
        }

    @staticmethod
    async def list_api_keys(db: AsyncSession, user_id: int) -> list[dict]:
        stmt = (
            select(SysApiKey)
            .where(SysApiKey.user_id == user_id, SysApiKey.status == 1)
            .order_by(SysApiKey.id.desc())
        )
        result = await db.execute(stmt)
        items = result.scalars().all()
        return [
            {
                "id": item.id,
                "name": item.name,
                "keyPrefix": item.key_prefix,
                "status": item.status,
                "expiresAt": item.expires_at,
                "lastUsedAt": item.last_used_at,
                "createTime": item.create_time,
            }
            for item in items
        ]

    @staticmethod
    async def delete_api_key(db: AsyncSession, user_id: int, key_id: int) -> bool:
        stmt = select(SysApiKey).where(
            SysApiKey.id == key_id, SysApiKey.user_id == user_id)
        result = await db.execute(stmt)
        entity = result.scalar_one_or_none()
        if not entity:
            return False
        await db.delete(entity)
        await db.flush()
        return True

    @staticmethod
    async def authenticate_by_key(db: AsyncSession, raw_key: str) -> UserContext:
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        stmt = select(SysApiKey).where(SysApiKey.key_hash == key_hash)
        result = await db.execute(stmt)
        api_key = result.scalar_one_or_none()
        if not api_key or api_key.status != 1:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ResultCode.TOKEN_INVALID.msg,
                headers={"WWW-Authenticate": "Bearer"},
            )
        if api_key.expires_at is not None and api_key.expires_at <= datetime.now():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ResultCode.TOKEN_INVALID.msg,
                headers={"WWW-Authenticate": "Bearer"},
            )

        user = await user_repository.get_by_id(db, api_key.user_id)
        if not user or user.status != 1:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ResultCode.ACCESS_UNAUTHORIZED.msg,
                headers={"WWW-Authenticate": "Bearer"},
            )

        roles = await user_repository.get_user_role_codes(db, user.id)
        from app.repository.role_repository import role_repository
        from app.service.menu_service import MenuService
        from app.dependencies.redis import get_redis_client
        redis = await get_redis_client()
        data_scope = await role_repository.get_maximum_data_scope(db, roles)
        perms = await MenuService.list_role_perms(db, redis, set(roles))

        api_key.last_used_at = datetime.now()
        await db.flush()

        return UserContext(
            id=user.id,
            username=user.username or "",
            dept_id=user.dept_id,
            data_scope=data_scope,
            roles=roles,
            permissions=list(perms),
        )
