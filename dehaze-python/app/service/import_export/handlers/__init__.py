"""
导入导出处理器实现
"""

from app.service.import_export.handlers.algorithm_export import AlgorithmExportHandler
from app.service.import_export.handlers.algorithm_import import AlgorithmImportHandler
from app.service.import_export.handlers.dataset_export import DatasetExportHandler
from app.service.import_export.handlers.dept_export import DeptExportHandler
from app.service.import_export.handlers.dept_import import DeptImportHandler
from app.service.import_export.handlers.dict_export import DictExportHandler
from app.service.import_export.handlers.dict_import import DictImportHandler
from app.service.import_export.handlers.menu_export import MenuExportHandler
from app.service.import_export.handlers.menu_import import MenuImportHandler
from app.service.import_export.handlers.role_export import RoleExportHandler
from app.service.import_export.handlers.role_import import RoleImportHandler
from app.service.import_export.handlers.user_export import UserExportHandler
from app.service.import_export.handlers.user_import import UserImportHandler
from app.service.import_export.registry import export_handler_registry, import_handler_registry

export_handler_registry.register(UserExportHandler())
import_handler_registry.register(UserImportHandler())
export_handler_registry.register(RoleExportHandler())
import_handler_registry.register(RoleImportHandler())
export_handler_registry.register(DeptExportHandler())
import_handler_registry.register(DeptImportHandler())
export_handler_registry.register(MenuExportHandler())
import_handler_registry.register(MenuImportHandler())
export_handler_registry.register(DictExportHandler())
import_handler_registry.register(DictImportHandler())
export_handler_registry.register(AlgorithmExportHandler())
import_handler_registry.register(AlgorithmImportHandler())
export_handler_registry.register(DatasetExportHandler())
