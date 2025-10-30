from typing import Optional, List, Dict, Any

from app.extensions import mysql
from app.models import SysDataset, SysDatasetItem, SysItemFile


class DatasetService:
    """数据集服务类，处理数据集相关的业务逻辑"""

    @staticmethod
    def get_dataset_list(keywords: str = None) -> List[Dict[str, Any]]:
        """
        获取数据集列表（树形结构）

        Args:
            keywords (str, optional): 搜索关键字（数据集名称）

        Returns:
            List[Dict[str, Any]]: 数据集列表
        """
        # 查询所有数据集
        query = SysDataset.query

        if keywords:
            query = query.filter(SysDataset.name.like(f'%{keywords}%'))

        datasets = query.all()

        # 构建数据集树
        return DatasetService._build_dataset_tree(0, datasets)

    @staticmethod
    def _build_dataset_tree(parent_id: int, datasets: List[SysDataset]) -> List[Dict[str, Any]]:
        """
        递归构建数据集树

        Args:
            parent_id (int): 父级数据集ID
            datasets (List[SysDataset]): 数据集列表

        Returns:
            List[Dict[str, Any]]: 树形数据集列表
        """
        tree = []
        for dataset in datasets:
            if dataset.parent_id == parent_id:
                dataset_dict = {
                    'id': dataset.id,
                    'parentId': dataset.parent_id,
                    'type': dataset.type,
                    'name': dataset.name,
                    'img': dataset.img,
                    'description': dataset.description,
                    'path': dataset.path,
                    'size': dataset.size,
                    'status': dataset.status,
                    'deleted': dataset.deleted,
                    'createTime': dataset.create_time.isoformat() if dataset.create_time else None,
                    'updateTime': dataset.update_time.isoformat() if dataset.update_time else None
                }

                # 递归查找子数据集
                children = DatasetService._build_dataset_tree(dataset.id, datasets)
                if children:
                    dataset_dict['children'] = children

                tree.append(dataset_dict)

        return tree

    @staticmethod
    def get_dataset_options() -> List[Dict[str, Any]]:
        """
        获取数据集下拉选项列表

        Returns:
            List[Dict[str, Any]]: 数据集下拉选项列表
        """
        datasets = SysDataset.query.all()
        return DatasetService._build_dataset_options(0, datasets)

    @staticmethod
    def _build_dataset_options(parent_id: int, datasets: List[SysDataset]) -> List[Dict[str, Any]]:
        """
        递归构建数据集下拉选项

        Args:
            parent_id (int): 父级数据集ID
            datasets (List[SysDataset]): 数据集列表

        Returns:
            List[Dict[str, Any]]: 数据集下拉选项列表
        """
        options = []
        for dataset in datasets:
            if dataset.parent_id == parent_id:
                option = {
                    'value': dataset.id,
                    'label': dataset.name
                }

                # 递归查找子数据集选项
                children = DatasetService._build_dataset_options(dataset.id, datasets)
                if children:
                    option['children'] = children

                options.append(option)

        return options

    @staticmethod
    def _generate_dataset_tree_path(parent_id: int) -> str:
        """
        生成数据集树路径

        Args:
            parent_id (int): 父级数据集ID

        Returns:
            str: 树路径，格式如 "0,1,2"
        """
        if parent_id == 0:
            return '0'
        else:
            # 由于数据库中可能没有tree_path字段，我们使用parent_id作为替代
            return str(parent_id)

    @staticmethod
    def get_dataset_by_id(dataset_id: int) -> Optional[Dict[str, Any]]:
        """
        根据ID获取数据集信息

        Args:
            dataset_id (int): 数据集ID

        Returns:
            Optional[Dict[str, Any]]: 数据集信息
        """
        dataset = SysDataset.query.get(dataset_id)
        if not dataset:
            return None

        return {
            'id': dataset.id,
            'parentId': dataset.parent_id,
            'type': dataset.type,
            'name': dataset.name,
            'img': dataset.img,
            'description': dataset.description,
            'path': dataset.path,
            'size': dataset.size,
            'status': dataset.status,
            'deleted': dataset.deleted,
            'createTime': dataset.create_time.isoformat() if dataset.create_time else None,
            'updateTime': dataset.update_time.isoformat() if dataset.update_time else None
        }

    @staticmethod
    def create_dataset(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建数据集

        Args:
            data (Dict[str, Any]): 数据集数据

        Returns:
            Dict[str, Any]: 创建结果
        """
        dataset = SysDataset()
        dataset.parent_id = data.get('parentId', 0)
        # 暂时不设置tree_path，因为数据库中可能没有该字段
        # dataset.tree_path = DatasetService._generate_dataset_tree_path(dataset.parent_id)
        dataset.type = data.get('type', '')
        dataset.name = data.get('name', '')
        dataset.description = data.get('description', '')
        dataset.path = data.get('path', '')
        dataset.status = data.get('status', 1)
        dataset.deleted = 0

        try:
            mysql.session.add(dataset)
            mysql.session.commit()
            return {'data': {'id': dataset.id}}
        except Exception as e:
            mysql.session.rollback()
            return {'error': f'创建数据集失败: {str(e)}'}

    @staticmethod
    def update_dataset(dataset_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新数据集

        Args:
            dataset_id (int): 数据集ID
            data (Dict[str, Any]): 数据集数据

        Returns:
            Dict[str, Any]: 更新结果
        """
        dataset = SysDataset.query.get(dataset_id)
        if not dataset:
            return {'error': '数据集不存在'}

        # 如果父ID发生变化
        parent_id = data.get('parentId', dataset.parent_id)
        # if parent_id != dataset.parent_id:
        #     dataset.tree_path = DatasetService._generate_dataset_tree_path(parent_id)

        dataset.parent_id = parent_id
        dataset.type = data.get('type', dataset.type)
        dataset.name = data.get('name', dataset.name)
        dataset.description = data.get('description', dataset.description)
        dataset.path = data.get('path', dataset.path)
        dataset.status = data.get('status', dataset.status)

        try:
            mysql.session.commit()
            return {'data': '更新成功'}
        except Exception as e:
            mysql.session.rollback()
            return {'error': f'更新数据集失败: {str(e)}'}

    @staticmethod
    def delete_datasets(dataset_ids: List[int]) -> Dict[str, Any]:
        """
        删除数据集（包括子数据集）

        Args:
            dataset_ids (List[int]): 数据集ID列表

        Returns:
            Dict[str, Any]: 删除结果
        """
        try:
            # 删除指定的数据集及其子数据集
            for dataset_id in dataset_ids:
                # 获取所有子数据集ID
                # 由于可能没有tree_path字段，我们使用递归方式查找子数据集
                def get_all_children(parent_id):
                    children = SysDataset.query.filter(SysDataset.parent_id == parent_id).all()
                    all_children = []
                    for child in children:
                        all_children.append(child)
                        all_children.extend(get_all_children(child.id))
                    return all_children

                child_datasets = get_all_children(dataset_id)

                child_ids = [child.id for child in child_datasets]

                # 删除所有相关数据集
                all_ids = [dataset_id] + child_ids
                SysDataset.query.filter(SysDataset.id.in_(all_ids)).delete(synchronize_session=False)

            mysql.session.commit()
            return {'data': '删除成功'}
        except Exception as e:
            mysql.session.rollback()
            return {'error': f'删除数据集失败: {str(e)}'}

    @staticmethod
    def get_image_items(dataset_id: int, page_num: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """
        获取数据集中的图片项（分页）

        Args:
            dataset_id (int): 数据集ID
            page_num (int): 页码
            page_size (int): 每页数量

        Returns:
            Dict[str, Any]: 分页图片项列表
        """
        try:
            # 获取叶子节点数据集ID
            leaf_ids = DatasetService._get_leaf_dataset_ids(dataset_id)

            # 分页查询数据项
            offset = (page_num - 1) * page_size
            dataset_items = SysDatasetItem.query.filter(
                SysDatasetItem.dataset_id.in_(leaf_ids)
            ).offset(offset).limit(page_size).all()

            total = SysDatasetItem.query.filter(
                SysDatasetItem.dataset_id.in_(leaf_ids)
            ).count()

            # 构建返回数据
            image_items = []
            for item in dataset_items:
                # 获取图片URL
                image_urls = DatasetService._get_image_urls(item.id)

                image_items.append({
                    'id': item.id,
                    'datasetId': item.dataset_id,
                    'imgUrl': image_urls
                })

            return {
                'data': {
                    'records': image_items,
                    'total': total,
                    'current': page_num,
                    'size': page_size
                }
            }
        except Exception as e:
            return {'error': f'获取图片项失败: {str(e)}'}

    @staticmethod
    def _get_leaf_dataset_ids(dataset_id: int) -> List[int]:
        """
        获取指定节点下的所有叶子节点ID

        Args:
            dataset_id (int): 数据集ID

        Returns:
            List[int]: 叶子节点ID列表
        """

        # 递归获取所有子节点
        def get_all_children(parent_id):
            children = SysDataset.query.filter(SysDataset.parent_id == parent_id).all()
            all_children = [child.id for child in children]
            for child in children:
                all_children.extend(get_all_children(child.id))
            return all_children

        # 获取所有子节点
        all_children_ids = get_all_children(dataset_id)

        # 获取叶子节点（没有子节点的节点）
        leaf_ids = []
        all_ids = [dataset_id] + all_children_ids
        for id in all_ids:
            children_count = SysDataset.query.filter(SysDataset.parent_id == id).count()
            if children_count == 0:
                leaf_ids.append(id)

        # 如果本身也是叶子节点，也要包含
        if not leaf_ids:
            leaf_ids = [dataset_id]

        return leaf_ids

    @staticmethod
    def _get_image_urls(dataset_item_id: int) -> List[Dict[str, Any]]:
        """
        获取数据项的图片URL

        Args:
            dataset_item_id (int): 数据项ID

        Returns:
            List[Dict[str, Any]]: 图片URL列表
        """
        item_files = SysItemFile.query.filter(SysItemFile.item_id == dataset_item_id).all()

        image_urls = []
        for item_file in item_files:
            # 这里应该从文件服务获取实际的URL
            # 目前简化处理
            image_urls.append({
                'id': item_file.id,
                'type': item_file.type,
                'url': f'/api/v1/files/{item_file.file_id}',
                'description': item_file.description
            })

        return image_urls


class DatasetItemService:
    """数据集项服务类，处理数据集项相关的业务逻辑"""

    @staticmethod
    def create_dataset_item(dataset_id: int, name: str = None) -> Dict[str, Any]:
        """
        创建数据项

        Args:
            dataset_id (int): 数据集ID
            name (str, optional): 数据项名称

        Returns:
            Dict[str, Any]: 创建结果
        """
        try:
            dataset_item = SysDatasetItem()
            dataset_item.dataset_id = dataset_id
            if name:
                dataset_item.name = name

            mysql.session.add(dataset_item)
            mysql.session.commit()
            return {'data': {'id': dataset_item.id}}
        except Exception as e:
            mysql.session.rollback()
            return {'error': f'创建数据项失败: {str(e)}'}

    @staticmethod
    def update_dataset_item(dataset_item_id: int, name: str) -> Dict[str, Any]:
        """
        更新数据项

        Args:
            dataset_item_id (int): 数据项ID
            name (str): 数据项名称

        Returns:
            Dict[str, Any]: 更新结果
        """
        try:
            dataset_item = SysDatasetItem.query.get(dataset_item_id)
            if not dataset_item:
                return {'error': '数据项不存在'}

            dataset_item.name = name
            mysql.session.commit()
            return {'data': '更新成功'}
        except Exception as e:
            mysql.session.rollback()
            return {'error': f'更新数据项失败: {str(e)}'}

    @staticmethod
    def delete_dataset_item(dataset_item_id: int) -> Dict[str, Any]:
        """
        删除数据项

        Args:
            dataset_item_id (int): 数据项ID

        Returns:
            Dict[str, Any]: 删除结果
        """
        try:
            # 删除关联的文件记录
            SysItemFile.query.filter(SysItemFile.item_id == dataset_item_id).delete(synchronize_session=False)

            # 删除数据项
            result = SysDatasetItem.query.filter(SysDatasetItem.id == dataset_item_id).delete(synchronize_session=False)

            if result == 0:
                return {'error': '数据项不存在'}

            mysql.session.commit()
            return {'data': '删除成功'}
        except Exception as e:
            mysql.session.rollback()
            return {'error': f'删除数据项失败: {str(e)}'}
