// 导出主要类和接口
export { DehazeSDK, Builder } from './DehazeSDK';
export { TokenManager } from './utils/TokenManager';
export { HttpManager } from './utils/HttpManager';

// 导出模型类
export { Result } from './model/Result';
export { PageResult } from './model/PageResult';
export { UserInfo, UserQuery, UserPageVO, UserForm } from './model/UserInfo';
export { LoginData, LoginResult, CaptchaResult } from './model/AuthInfo';
export { DeptForm, DeptQuery, DeptVO } from './model/DeptInfo';
export { RoleForm, RoleQuery, RolePageVO } from './model/RoleInfo';
export { MenuTypeEnum, MenuQuery, MenuVO, MenuForm, RouteVO, Meta } from './model/MenuInfo';
export { DictTypeQuery, DictTypePageVO, DictTypeForm, DictQuery, DictForm, DictPageVO } from './model/DictInfo';
export { PredParam, PredResult, EvalParam, EvalResult, PredImageInfo } from './model/ModelInfo';
export { Algorithm, AlgorithmQuery } from './model/AlgorithmInfo';
export { Dataset, DatasetQuery, DatasetImageFileInfo, ImageItem, ImageItemQuery, ImageUrl } from './model/DatasetInfo';
export { FileInfo, ImageFileInfo } from './model/FileInfo';

// 导出API类
export { UserAPI } from './api/user';
export { AuthAPI } from './api/auth';
export { DeptAPI } from './api/dept';
export { RoleAPI } from './api/role';
export { MenuAPI } from './api/menu';
export { DictAPI } from './api/dict';
export { ModelAPI } from './api/model';
export { AlgorithmAPI } from './api/algorithm';
export { DatasetAPI } from './api/dataset';
export { FileAPI } from './api/file';