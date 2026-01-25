"""
数据集相关表单对象 - 服务层内部使用
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class DatasetQuery:
    """数据集查询参数"""
    keyword: Optional[str] = None
    type: Optional[str] = None
    status: Optional[int] = None


@dataclass
class DatasetAddForm:
    """数据集新增表单"""
    parent_id: int = 0
    name: str = ''
    type: str = ''
    description: str = ''
    path: str = ''
    status: int = 1


@dataclass
class DatasetUpdateForm:
    """数据集更新表单"""
    parent_id: Optional[int] = None
    name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    path: Optional[str] = None
    status: Optional[int] = None


@dataclass
class DatasetItemCreateForm:
    """数据项创建表单"""
    dataset_id: int = 0
    name: Optional[str] = None
    scene_type: Optional[str] = None
    description: Optional[str] = None


@dataclass
class DatasetItemUpdateForm:
    """数据项更新表单"""
    id: int = 0
    name: Optional[str] = None
    scene_type: Optional[str] = None
    description: Optional[str] = None


@dataclass
class DatasetItemUploadForm:
    """数据项上传表单（单个）"""
    dataset_id: int = 0
    item_name: Optional[str] = None
    scene_type: Optional[str] = None
    haze_level: Optional[str] = None
    description: Optional[str] = None


@dataclass
class BatchDatasetItemUploadForm:
    """批量数据项上传表单"""
    dataset_id: int = 0
    items: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ItemFileUpdateForm:
    """数据项文件更新表单"""
    id: int = 0
    type: Optional[str] = None
    scene_type: Optional[str] = None
    haze_level: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ExportTaskCreateForm:
    """导出任务创建表单"""
    type: str = ''
    target_id: Optional[int] = None
    target_ids: List[int] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
