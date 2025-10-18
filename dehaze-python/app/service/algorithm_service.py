from typing import List, Dict, Optional, Any
from app.models import SysAlgorithm
from app.extensions import mysql
from app.utils.utils import format_time, result_util
from sqlalchemy import or_, and_
import os
from app.utils.file import get_file_size


class AlgorithmService:
    @staticmethod
    def get_algorithm_list(keywords: str = None) -> List[Dict[str, Any]]:
        """
        获取算法树形表格
        :param keywords: 搜索关键词
        :return: 算法列表
        """
        query = SysAlgorithm.query
        
        if keywords:
            query = query.filter(SysAlgorithm.name.like(f'%{keywords}%'))
            
        algorithms = query.all()
        
        # 构建树形结构
        algorithm_dict = {algorithm.id: {
            'id': algorithm.id,
            'parent_id': algorithm.parent_id,
            'type': algorithm.type,
            'name': algorithm.name,
            'path': algorithm.path,
            'size': algorithm.size,
            'img': algorithm.img,
            'params': algorithm.params,
            'flops': algorithm.flops,
            'import_path': algorithm.import_path,
            'description': algorithm.description,
            'status': algorithm.status,
            'create_time': format_time(algorithm.create_time),
            'update_time': format_time(algorithm.update_time),
            'children': []
        } for algorithm in algorithms}
        
        # 构建父子关系
        root_algorithms = []
        for algorithm in algorithm_dict.values():
            if algorithm['parent_id'] == 0:
                root_algorithms.append(algorithm)
            else:
                parent = algorithm_dict.get(algorithm['parent_id'])
                if parent:
                    parent['children'].append(algorithm)
                    
        return root_algorithms

    @staticmethod
    def get_algorithm_options() -> List[Dict[str, Any]]:
        """
        获取模型下拉选项列表
        :return: 模型下拉选项列表
        """
        algorithms = SysAlgorithm.query.filter(SysAlgorithm.status == 1).all()
        
        # 构建树形选项
        algorithm_dict = {algorithm.id: {
            'value': algorithm.id,
            'label': algorithm.name,
            'children': []
        } for algorithm in algorithms}
        
        # 构建父子关系
        root_options = []
        for algorithm in algorithm_dict.values():
            alg_obj = next((a for a in algorithms if a.id == algorithm['value']), None)
            if alg_obj and alg_obj.parent_id == 0:
                root_options.append(algorithm)
            else:
                if alg_obj:
                    parent = algorithm_dict.get(alg_obj.parent_id)
                    if parent:
                        parent['children'].append(algorithm)
                        
        return root_options

    @staticmethod
    def get_algorithm_by_id(algorithm_id: int) -> Optional[Dict[str, Any]]:
        """
        根据ID获取算法信息
        :param algorithm_id: 算法ID
        :return: 算法信息
        """
        algorithm = SysAlgorithm.query.get(algorithm_id)
        if not algorithm:
            return None
            
        return {
            'id': algorithm.id,
            'parent_id': algorithm.parent_id,
            'type': algorithm.type,
            'name': algorithm.name,
            'path': algorithm.path,
            'size': algorithm.size,
            'img': algorithm.img,
            'params': algorithm.params,
            'flops': algorithm.flops,
            'import_path': algorithm.import_path,
            'description': algorithm.description,
            'status': algorithm.status,
            'create_time': format_time(algorithm.create_time),
            'update_time': format_time(algorithm.update_time)
        }

    @staticmethod
    def create_algorithm(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        新增算法
        :param data: 算法数据
        :return: 创建结果
        """
        algorithm = SysAlgorithm()
        algorithm.parent_id = data.get('parent_id', 0)
        algorithm.type = data.get('type', '')
        algorithm.name = data.get('name', '')
        algorithm.path = data.get('path', '')
        algorithm.import_path = data.get('import_path', '')
        algorithm.description = data.get('description', '')
        algorithm.status = data.get('status', 1)
        
        # 如果路径是有效文件，获取文件大小
        if 'path' in data and os.path.isfile(data['path']):
            algorithm.size = get_file_size(data['path'])
        
        try:
            mysql.session.add(algorithm)
            mysql.session.commit()
            return {'success': True, 'message': '算法创建成功'}
        except Exception as e:
            mysql.session.rollback()
            return {'success': False, 'message': f'算法创建失败: {str(e)}'}

    @staticmethod
    def update_algorithm(algorithm_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        修改算法
        :param algorithm_id: 算法ID
        :param data: 算法数据
        :return: 更新结果
        """
        algorithm = SysAlgorithm.query.get(algorithm_id)
        if not algorithm:
            return {'success': False, 'message': '算法不存在'}
            
        algorithm.parent_id = data.get('parent_id', algorithm.parent_id)
        algorithm.type = data.get('type', algorithm.type)
        algorithm.name = data.get('name', algorithm.name)
        algorithm.path = data.get('path', algorithm.path)
        algorithm.import_path = data.get('import_path', algorithm.import_path)
        algorithm.description = data.get('description', algorithm.description)
        algorithm.status = data.get('status', algorithm.status)
        
        # 如果路径是有效文件，获取文件大小
        if 'path' in data and os.path.isfile(data['path']):
            algorithm.size = get_file_size(data['path'])
        
        try:
            mysql.session.commit()
            return {'success': True, 'message': '算法更新成功'}
        except Exception as e:
            mysql.session.rollback()
            return {'success': False, 'message': f'算法更新失败: {str(e)}'}

    @staticmethod
    def delete_algorithms(algorithm_ids: List[int]) -> Dict[str, Any]:
        """
        删除算法
        :param algorithm_ids: 算法ID列表
        :return: 删除结果
        """
        try:
            # 查找要删除的算法及其子算法
            algorithms = SysAlgorithm.query.filter(
                or_(
                    SysAlgorithm.id.in_(algorithm_ids),
                    SysAlgorithm.parent_id.in_(algorithm_ids)
                )
            ).all()
            
            algorithm_id_list = [alg.id for alg in algorithms]
            
            # 删除算法
            if algorithm_id_list:
                deleted_count = SysAlgorithm.query.filter(SysAlgorithm.id.in_(algorithm_id_list)).delete(synchronize_session=False)
            mysql.session.commit()
            return {'success': True, 'message': '算法删除成功'}
        except Exception as e:
            mysql.session.rollback()
            return {'success': False, 'message': f'算法删除失败: {str(e)}'}