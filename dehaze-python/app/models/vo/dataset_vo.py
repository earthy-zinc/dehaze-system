"""
数据集相关视图对象
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.entity.sys_dataset import SysDataset, SysDatasetItem, SysItemFile
    from app.models.entity.sys_file import SysFile


class DatasetStatistics:
    """数据集统计数据"""

    def __init__(self):
        self.item_count = 0
        self.file_count = 0
        self.total_size = 0
        self.clear_count = 0
        self.hazy_count = 0
        self.scene_distribution = {}
        self.haze_distribution = {}
        self.format_distribution = {}

    def to_dict(self):
        return {
            'itemCount': self.item_count,
            'fileCount': self.file_count,
            'totalSize': self.total_size,
            'clearCount': self.clear_count,
            'hazyCount': self.hazy_count,
            'sceneDistribution': self.scene_distribution,
            'hazeDistribution': self.haze_distribution,
            'formatDistribution': self.format_distribution
        }


class DatasetVO:
    """数据集视图对象"""

    def __init__(self, dataset: 'SysDataset', statistics: DatasetStatistics = None):
        self.id = dataset.id
        self.parent_id = dataset.parent_id
        self.tree_path = getattr(dataset, 'tree_path', '')
        self.type = dataset.type
        self.name = dataset.name
        self.img = dataset.img
        self.description = dataset.description
        self.path = dataset.path
        self.size = dataset.size
        self.status = dataset.status
        self.deleted = dataset.deleted
        self.usage_count = getattr(dataset, 'usage_count', 0)
        self.create_time = dataset.create_time.isoformat() if dataset.create_time else None
        self.update_time = dataset.update_time.isoformat() if dataset.update_time else None
        self.statistics = statistics.to_dict() if statistics else None
        self.children = []

    def to_dict(self):
        return {
            'id': self.id,
            'parentId': self.parent_id,
            'treePath': self.tree_path,
            'type': self.type,
            'name': self.name,
            'img': self.img,
            'description': self.description,
            'path': self.path,
            'size': self.size,
            'status': self.status,
            'deleted': self.deleted,
            'usageCount': self.usage_count,
            'createTime': self.create_time,
            'updateTime': self.update_time,
            'statistics': self.statistics,
            'children': [child.to_dict() if isinstance(child, DatasetVO) else child for child in self.children]
        }


class DatasetItemVO:
    """数据集项视图对象"""

    def __init__(self, dataset_item: 'SysDatasetItem', files: list = None):
        self.id = dataset_item.id
        self.dataset_id = dataset_item.dataset_id
        self.name = dataset_item.name
        self.create_time = dataset_item.create_time.isoformat() if hasattr(dataset_item,
                                                                           'create_time') and dataset_item.create_time else None
        self.update_time = dataset_item.update_time.isoformat() if hasattr(dataset_item,
                                                                           'update_time') and dataset_item.update_time else None
        self.files = files if files else []
        self.image_urls = []

    def to_dict(self):
        return {
            'id': self.id,
            'datasetId': self.dataset_id,
            'name': self.name,
            'createTime': self.create_time,
            'updateTime': self.update_time,
            'files': self.files,
            'imgUrl': self.image_urls
        }


class ItemFileVO:
    """数据项文件视图对象"""

    def __init__(self, item_file: 'SysItemFile', file_obj: 'SysFile' = None):
        self.id = item_file.id
        self.item_id = item_file.item_id
        self.file_id = item_file.file_id
        self.thumbnail_file_id = item_file.thumbnail_file_id
        self.type = item_file.type
        self.description = item_file.description
        if file_obj:
            self.url = file_obj.url
            self.name = file_obj.name
            self.size = file_obj.size
            self.md5 = file_obj.md5


class ImageUrlVO:
    """图片URL视图对象"""

    def __init__(self, file_id: int, url: str, thumbnail_url: str = None):
        self.file_id = file_id
        self.url = url
        self.thumbnail_url = thumbnail_url
