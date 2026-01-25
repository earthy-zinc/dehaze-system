"""
批量操作相关视图对象
"""


class BatchDeleteResult:
    """批量删除结果"""

    def __init__(self, total: int, succeeded: int, failed: int, results: list = None):
        self.total = total
        self.succeeded = succeeded
        self.failed = failed
        self.results = results if results else []

    def to_dict(self):
        return {
            'total': self.total,
            'succeeded': self.succeeded,
            'failed': self.failed,
            'results': self.results
        }


class BatchDeleteResultItem:
    """批量删除结果项"""

    def __init__(self, dataset_id: int, status: str, message: str = None, error_code: str = None):
        self.id = dataset_id
        self.status = status
        self.message = message
        self.error_code = error_code

    def to_dict(self):
        return {
            'id': self.id,
            'status': self.status,
            'message': self.message,
            'errorCode': self.error_code
        }


class BatchDeleteResultVO:
    """批量删除结果视图对象"""

    def __init__(self, success: int, failed: int, errors: list = None):
        self.success = success
        self.failed = failed
        self.errors = errors if errors else []

    def to_dict(self):
        return {
            'success': self.success,
            'failed': self.failed,
            'errors': self.errors
        }


class BatchActionFailureDetailVO:
    """批量操作失败详情视图对象"""

    def __init__(self, identifier: str, reason: str):
        self.identifier = identifier
        self.reason = reason

    def to_dict(self):
        return {
            'identifier': self.identifier,
            'reason': self.reason
        }


class BatchOperationResultVO:
    """批量操作结果视图对象"""

    def __init__(self, success_count: int = 0, failed_count: int = 0,
                 success_ids: list = None, failure_details: list = None, message: str = ''):
        self.success_count = success_count
        self.failed_count = failed_count
        self.success_ids = success_ids if success_ids else []
        self.failure_details = failure_details if failure_details else []
        self.message = message

    def to_dict(self):
        return {
            'successCount': self.success_count,
            'failedCount': self.failed_count,
            'successIds': self.success_ids,
            'message': self.message,
            'failureDetails': [f.to_dict() if hasattr(f, 'to_dict') else f for f in self.failure_details]
        }


class BatchUploadResultVO:
    """批量上传结果视图对象"""

    def __init__(self, total: int, success: int, failed: int,
                 success_items: list = None, failed_items: list = None):
        self.total = total
        self.success = success
        self.failed = failed
        self.success_items = success_items if success_items else []
        self.failed_items = failed_items if failed_items else []

    def to_dict(self):
        return {
            'total': self.total,
            'succeeded': self.success,
            'failed': self.failed,
            'successItems': [item.to_dict() if hasattr(item, 'to_dict') else item for item in self.success_items],
            'failedItems': [item.to_dict() if hasattr(item, 'to_dict') else item for item in self.failed_items]
        }


class BatchUploadSuccessItemVO:
    """批量上传成功项视图对象"""

    def __init__(self, dataset_item_id: int, name: str, file_count: int = 1):
        self.dataset_item_id = dataset_item_id
        self.name = name
        self.file_count = file_count

    def to_dict(self):
        return {
            'id': self.dataset_item_id,
            'name': self.name,
            'fileCount': self.file_count
        }


class BatchUploadFailedItemVO:
    """批量上传失败项视图对象"""

    def __init__(self, filename: str, error_message: str):
        self.filename = filename
        self.error_message = error_message

    def to_dict(self):
        return {
            'filename': self.filename,
            'errorMessage': self.error_message
        }
