"""
操作日志服务
"""

from datetime import datetime, timedelta, timezone
from typing import List, Dict

from sqlalchemy import func, desc
from app.extensions import mysql
from app.models import SysOperationLog


class OperationLogService:
    """操作日志服务"""

    @staticmethod
    def get_logs_by_user_id(user_id: int, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        根据用户ID查询操作日志

        Args:
            user_id: 用户ID
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            日志列表
        """
        logs = SysOperationLog.query.filter_by(user_id=user_id)\
            .order_by(desc(SysOperationLog.create_time))\
            .limit(limit)\
            .offset(offset)\
            .all()

        return [log.to_dict() for log in logs]

    @staticmethod
    def get_logs_by_date_range(start_date: datetime, end_date: datetime,
                                limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        根据日期范围查询操作日志

        Args:
            start_date: 开始日期
            end_date: 结束日期
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            日志列表
        """
        logs = SysOperationLog.query.filter(
            SysOperationLog.create_time >= start_date,
            SysOperationLog.create_time <= end_date
        ).order_by(desc(SysOperationLog.create_time))\
         .limit(limit)\
         .offset(offset)\
         .all()

        return [log.to_dict() for log in logs]

    @staticmethod
    def get_logs_by_status(status: int, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        根据状态码查询操作日志

        Args:
            status: HTTP状态码
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            日志列表
        """
        logs = SysOperationLog.query.filter_by(status=status)\
            .order_by(desc(SysOperationLog.create_time))\
            .limit(limit)\
            .offset(offset)\
            .all()

        return [log.to_dict() for log in logs]

    @staticmethod
    def get_logs_by_path(path: str, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        根据请求路径查询操作日志

        Args:
            path: 请求路径（支持模糊匹配）
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            日志列表
        """
        logs = SysOperationLog.query.filter(
            SysOperationLog.path.like(f'%{path}%')
        ).order_by(desc(SysOperationLog.create_time))\
         .limit(limit)\
         .offset(offset)\
         .all()

        return [log.to_dict() for log in logs]

    @staticmethod
    def get_error_logs(limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        获取错误日志（状态码 >= 400）

        Args:
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            错误日志列表
        """
        logs = SysOperationLog.query.filter(
            SysOperationLog.status >= 400
        ).order_by(desc(SysOperationLog.create_time))\
         .limit(limit)\
         .offset(offset)\
         .all()

        return [log.to_dict() for log in logs]

    @staticmethod
    def get_request_statistics(days: int = 7) -> Dict:
        """
        获取请求统计数据

        Args:
            days: 统计最近几天的数据

        Returns:
            统计数据字典，包含：
            - total_requests: 总请求数
            - success_requests: 成功请求数（2xx, 3xx）
            - error_requests: 错误请求数（4xx, 5xx）
            - avg_latency: 平均响应时间
            - top_paths: 最常访问的路径
            - top_users: 最活跃的用户
        """
        start_date = datetime.now(timezone.utc) - timedelta(days=days)

        # 总请求数
        total_requests = SysOperationLog.query.filter(
            SysOperationLog.create_time >= start_date
        ).count()

        # 成功请求数
        success_requests = SysOperationLog.query.filter(
            SysOperationLog.create_time >= start_date,
            SysOperationLog.status < 400
        ).count()

        # 错误请求数
        error_requests = total_requests - success_requests

        # 平均响应时间
        avg_latency_result = SysOperationLog.query.filter(
            SysOperationLog.create_time >= start_date
        ).with_entities(func.avg(SysOperationLog.latency)).first()

        avg_latency = int(avg_latency_result[0]) if avg_latency_result and avg_latency_result[0] else 0

        # 最常访问的路径（Top 10）
        top_paths_query = SysOperationLog.query.filter(
            SysOperationLog.create_time >= start_date
        ).with_entities(
            SysOperationLog.path,
            func.count(SysOperationLog.id).label('count')
        ).group_by(
            SysOperationLog.path
        ).order_by(desc('count')).limit(10).all()

        top_paths = [
            {'path': path, 'count': count}
            for path, count in top_paths_query
        ]

        # 最活跃的用户（Top 10）
        top_users_query = SysOperationLog.query.filter(
            SysOperationLog.create_time >= start_date,
            SysOperationLog.user_id.isnot(None)
        ).with_entities(
            SysOperationLog.user_id,
            func.count(SysOperationLog.id).label('count')
        ).group_by(
            SysOperationLog.user_id
        ).order_by(desc('count')).limit(10).all()

        top_users = [
            {'userId': user_id, 'count': count}
            for user_id, count in top_users_query
        ]

        return {
            'totalRequests': total_requests,
            'successRequests': success_requests,
            'errorRequests': error_requests,
            'avgLatency': avg_latency,
            'topPaths': top_paths,
            'topUsers': top_users,
            'dateRange': {
                'start': start_date.isoformat(),
                'end': datetime.now(timezone.utc).isoformat()
            }
        }

    @staticmethod
    def get_slow_requests(threshold: int = 1000, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        获取慢请求列表

        Args:
            threshold: 响应时间阈值（毫秒）
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            慢请求列表
        """
        logs = SysOperationLog.query.filter(
            SysOperationLog.latency > threshold
        ).order_by(desc(SysOperationLog.latency))\
         .limit(limit)\
         .offset(offset)\
         .all()

        return [log.to_dict() for log in logs]

    @staticmethod
    def get_ip_statistics(days: int = 7, limit: int = 20) -> List[Dict]:
        """
        获取IP访问统计

        Args:
            days: 统计最近几天的数据
            limit: 返回数量限制

        Returns:
            IP统计列表
        """
        start_date = datetime.now(timezone.utc) - timedelta(days=days)

        ip_stats = SysOperationLog.query.filter(
            SysOperationLog.create_time >= start_date
        ).with_entities(
            SysOperationLog.ip,
            func.count(SysOperationLog.id).label('count'),
            func.avg(SysOperationLog.latency).label('avg_latency')
        ).group_by(
            SysOperationLog.ip
        ).order_by(desc('count')).limit(limit).all()

        return [
            {
                'ip': ip,
                'count': count,
                'avgLatency': int(avg_latency) if avg_latency else 0
            }
            for ip, count, avg_latency in ip_stats
        ]

    @staticmethod
    def delete_old_logs(days: int = 30) -> int:
        """
        删除指定天数之前的旧日志

        Args:
            days: 保留最近几天的日志

        Returns:
            删除的记录数
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        deleted = SysOperationLog.query.filter(
            SysOperationLog.create_time < cutoff_date
        ).delete()

        mysql.session.commit()

        return deleted

    @staticmethod
    def search_logs(keyword: str, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        搜索日志（在路径、User-Agent中搜索）

        Args:
            keyword: 搜索关键词
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            日志列表
        """
        logs = SysOperationLog.query.filter(
            (SysOperationLog.path.like(f'%{keyword}%')) |
            (SysOperationLog.agent.like(f'%{keyword}%')) |
            (SysOperationLog.error_message.like(f'%{keyword}%'))
        ).order_by(desc(SysOperationLog.create_time))\
         .limit(limit)\
         .offset(offset)\
         .all()

        return [log.to_dict() for log in logs]
