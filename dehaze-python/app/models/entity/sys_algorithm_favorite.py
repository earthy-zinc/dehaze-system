"""
算法收藏实体
"""
from sqlalchemy import BigInteger, Index, DateTime, UniqueConstraint
from datetime import datetime, timezone
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class SysAlgorithmFavorite(Base):
    """算法收藏表"""
    __tablename__ = 'sys_algorithm_favorite'
    __table_args__ = (
        UniqueConstraint('user_id', 'algorithm_id', name='uk_user_algorithm'),
        Index('idx_user_id', 'user_id'),
        Index('idx_algorithm_id', 'algorithm_id'),
        {'comment': '算法收藏表'},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='id')
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='用户ID')
    algorithm_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='算法ID')
    create_time: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), comment='收藏时间'
    )
