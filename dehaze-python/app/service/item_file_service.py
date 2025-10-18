from app.extensions import mysql
from app.models import SysItemFile, SysFile
from app.service.file import upload_file_from_request, _upload_to_storage
from typing import Dict, Any, List, Optional
from io import BytesIO
from PIL import Image
import os
from app.utils.utils import result_util


class ItemFileService:
    """数据项文件服务类，处理数据项文件相关的业务逻辑"""

    @staticmethod
    def save_item_file(item_id: int, file, type: str, description: str = None) -> Dict[str, Any]:
        """
        保存数据项文件
        
        Args:
            item_id (int): 数据项ID
            file: 文件对象
            type (str): 图片类型
            description (str, optional): 描述
            
        Returns:
            Dict[str, Any]: 保存结果
        """
        try:
            # 上传原文件
            original_file_info = upload_file_from_request(file)
            
            # 生成并上传缩略图
            file_bytes = BytesIO(file.read())
            thumbnail_bytes = ItemFileService._generate_thumbnail(file_bytes)
            thumbnail_file_info = _upload_to_storage(
                filename=f"thumbnail_{original_file_info.name}",
                content_type=file.mimetype,
                file_bytes=thumbnail_bytes,
                file_size=len(thumbnail_bytes.getvalue())
            )
            
            # 检查是否已存在关联关系
            item_file = SysItemFile.query.filter(
                SysItemFile.file_id == original_file_info.id,
                SysItemFile.thumbnail_file_id == thumbnail_file_info.id
            ).first()
            
            if not item_file:
                # 创建数据项与文件关联关系
                item_file = SysItemFile()
                item_file.item_id = item_id
                item_file.file_id = original_file_info.id
                item_file.thumbnail_file_id = thumbnail_file_info.id
                item_file.type = type
                item_file.description = description
                
                mysql.session.add(item_file)
                mysql.session.commit()
            
            return {
                'success': True,
                'data': {
                    'id': item_file.id,
                    'datasetItemId': item_file.item_id,
                    'fileId': item_file.file_id,
                    'type': item_file.type,
                    'description': item_file.description,
                    'url': original_file_info.url
                }
            }
        except Exception as e:
            mysql.session.rollback()
            return {
                'success': False,
                'message': f'保存数据项文件失败: {str(e)}'
            }

    @staticmethod
    def _generate_thumbnail(file_bytes: BytesIO, max_width: int = 400, max_height: int = 400) -> BytesIO:
        """
        生成缩略图
        
        Args:
            file_bytes (BytesIO): 原始文件字节流
            max_width (int): 最大宽度
            max_height (int): 最大高度
            
        Returns:
            BytesIO: 缩略图字节流
        """
        # 重置文件指针
        file_bytes.seek(0)
        
        # 打开图片
        image = Image.open(file_bytes)
        
        # 计算缩略图尺寸
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        
        # 保存为字节流
        thumbnail_bytes = BytesIO()
        image.save(thumbnail_bytes, format=image.format)
        thumbnail_bytes.seek(0)
        
        return thumbnail_bytes

    @staticmethod
    def get_image_urls(item_id: int) -> List[Dict[str, Any]]:
        """
        获取图片URL列表
        
        Args:
            item_id (int): 数据项ID
            
        Returns:
            List[Dict[str, Any]]: 图片URL列表
        """
        item_files = SysItemFile.query.filter(SysItemFile.item_id == item_id).all()
        
        image_urls = []
        for item_file in item_files:
            # 获取原文件信息
            original_file = SysFile.query.get(item_file.file_id)
            if original_file:
                image_urls.append({
                    'id': item_file.id,
                    'type': item_file.type,
                    'url': original_file.url,
                    'description': item_file.description
                })
                
        return image_urls

    @staticmethod
    def delete_item_file(item_file_id: int) -> Dict[str, Any]:
        """
        删除数据项文件
        
        Args:
            item_file_id (int): 数据项文件ID
            
        Returns:
            Dict[str, Any]: 删除结果
        """
        try:
            item_file = SysItemFile.query.get(item_file_id)
            if not item_file:
                return {
                    'success': False,
                    'message': '未查询到对应数据项'
                }
            
            # 删除原文件和缩略图文件记录
            # 注意：这里只删除数据库记录，实际文件存储可能还需要额外处理
            mysql.session.delete(item_file)
            mysql.session.commit()
            
            return {
                'success': True,
                'message': '删除成功'
            }
        except Exception as e:
            mysql.session.rollback()
            return {
                'success': False,
                'message': f'删除数据项文件失败: {str(e)}'
            }