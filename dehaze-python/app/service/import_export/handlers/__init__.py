"""
导入导出处理器实现
"""
from app.service.import_export.handlers.user_export import UserExportHandler
from app.service.import_export.handlers.user_import import UserImportHandler
from app.service.import_export.handlers.role_export import RoleExportHandler
from app.service.import_export.handlers.role_import import RoleImportHandler
from app.service.import_export.handlers.dept_export import DeptExportHandler
from app.service.import_export.handlers.dept_import import DeptImportHandler
from app.service.import_export.handlers.menu_export import MenuExportHandler
from app.service.import_export.handlers.menu_import import MenuImportHandler
from app.service.import_export.handlers.dict_export import DictExportHandler
from app.service.import_export.handlers.dict_import import DictImportHandler
from app.service.import_export.handlers.algorithm_export import AlgorithmExportHandler
from app.service.import_export.handlers.algorithm_import import AlgorithmImportHandler
from app.service.import_export.handlers.dataset_export import DatasetExportHandler
from app.service.import_export.registry import (register_export_handler,
                                                register_import_handler)

_user_export = UserExportHandler()
_user_import = UserImportHandler()
_role_export = RoleExportHandler()
_role_import = RoleImportHandler()
_dept_export = DeptExportHandler()
_dept_import = DeptImportHandler()
_menu_export = MenuExportHandler()
_menu_import = MenuImportHandler()
_dict_export = DictExportHandler()
_dict_import = DictImportHandler()
_algorithm_export = AlgorithmExportHandler()
_algorithm_import = AlgorithmImportHandler()
_dataset_export = DatasetExportHandler()

register_export_handler(_user_export)
register_import_handler(_user_import)
register_export_handler(_role_export)
register_import_handler(_role_import)
register_export_handler(_dept_export)
register_import_handler(_dept_import)
register_export_handler(_menu_export)
register_import_handler(_menu_import)
register_export_handler(_dict_export)
register_import_handler(_dict_import)
register_export_handler(_algorithm_export)
register_import_handler(_algorithm_import)
register_export_handler(_dataset_export)
