// ==================== SDK 核心类 ====================
export { DehazeSDK, Builder } from './DehazeSDK';
export { TokenManager } from './utils/TokenManager';
export { HttpManager, ApiError } from './utils/HttpManager';

// ==================== 模型类 ====================
export { Result, ResultCode, ErrorItem, TOKEN_INVALID_CODES } from './model/Result';
export { PageResult } from './model/PageResult';
export { LoginData, LoginResult, LoginUser, CaptchaResult, AuthUserInfo } from './model/AuthInfo';
export { Algorithm, AlgorithmQuery } from './model/AlgorithmInfo';
export {
  Dataset, DatasetQuery, DatasetItem, DatasetItemQuery,
  ImageUrl, ImageItem, PairedImageUpload, DatasetImageFileInfo
} from './model/DatasetInfo';
export {
  PredParam, PredResult, PredLog, PredLogQuery,
  EvalParam, EvalResult, EvalLogQuery
} from './model/ModelInfo';
export { InputHistory, InputHistoryQuery, CreateInputHistory, HistoryStatus } from './model/InputHistoryInfo';
export { FileInfo } from './model/FileInfo';
export { DeptForm, DeptQuery, DeptVO } from './model/DeptInfo';
export { RoleForm, RoleQuery, RolePageVO } from './model/RoleInfo';
export { MenuTypeEnum, MenuQuery, MenuVO, MenuForm, RouteVO, Meta } from './model/MenuInfo';
export { DictTypeQuery, DictTypePageVO, DictTypeForm, DictQuery, DictForm, DictPageVO } from './model/DictInfo';
export { UserInfo, UserQuery, UserPageVO, UserForm } from './model/UserInfo';

// ==================== API 类 ====================
export { AuthAPI } from './api/auth';
export { UserAPI } from './api/user';
export { AlgorithmAPI } from './api/algorithm';
export { DatasetAPI } from './api/dataset';
export { PredictionAPI } from './api/prediction';
export { EvaluationAPI } from './api/evaluation';
export { InputHistoryAPI } from './api/inputHistory';
export { DeptAPI } from './api/dept';
export { DictAPI } from './api/dict';
export { FileAPI } from './api/file';
export { MenuAPI } from './api/menu';
export { RoleAPI } from './api/role';
