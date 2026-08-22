from sqlalchemy import BigInteger, Integer, String, Text
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysKnowledgeDocument(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_knowledge_document"
    __table_args__ = {"comment": "AI知识库文档表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    knowledge_base_id: Mapped[int] = mapped_column(
        BigInteger, index=True, nullable=False, comment="知识库ID(关联sys_knowledge_base.id)"
    )
    file_id: Mapped[int | None] = mapped_column(
        BigInteger,
        index=True,
        nullable=True,
        comment="文件ID(关联sys_file.id，url导入与自定义文本无关联文件可为空)",
    )
    title: Mapped[str] = mapped_column(
        String(512), nullable=False, comment="文档标题(文件名或手动指定)"
    )
    source: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="upload",
        comment="文档来源(manual:手动;upload:上传;url:URL导入;algorithm_sync:算法同步;experience:经验沉淀)",
    )
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, comment="文档版本号(更新时+1，支撑版本回溯)"
    )
    parsing_strategy: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="auto",
        comment="解析策略(auto:自动;ocr:OCR;text:纯文本;table:表格)",
    )
    content: Mapped[str | None] = mapped_column(
        LONGTEXT, nullable=True, comment="解析后的纯文本内容"
    )
    raw_content: Mapped[str | None] = mapped_column(
        LONGTEXT, nullable=True, comment="原始富文本(含Markdown/表格)"
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="分块数(冗余统计)"
    )
    total_tokens: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment="编码Token总数(冗余统计)"
    )
    processing_status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="pending",
        comment="处理状态(pending:待处理;processing:处理中;completed:已完成;failed:失败)",
    )
    error: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="失败原因(processing_status=failed时填充)"
    )
