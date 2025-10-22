class BusinessException(Exception):
    """业务异常类"""
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)


