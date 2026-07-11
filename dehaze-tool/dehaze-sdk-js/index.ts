// API 模块导入
import AlgorithmAPI from "./src/api/algorithm";
import AuthAPI from "./src/api/auth";
import DatasetAPI, { DatasetItemAPI, ItemFileAPI, ExportTaskAPI } from "./src/api/dataset";
import DeptAPI from "./src/api/dept";
import DictAPI from "./src/api/dict";
import FileAPI from "./src/api/file";
import ImageInputHistoryAPI from "./src/api/image-input";
import MenuAPI from "./src/api/menu";
import ModelAPI from "./src/api/model";
import RoleAPI from "./src/api/role";
import UserAPI from "./src/api/user";

// API 模型导出
export * from "./src/api/algorithm/model";
export * from "./src/api/auth/model";
export * from "./src/api/dataset/model";
export * from "./src/api/dept/model";
export * from "./src/api/dict/model";
export * from "./src/api/file/model";
export * from "./src/api/image-input/model";
export * from "./src/api/menu/model";
export * from "./src/api/model/model";
export * from "./src/api/role/model";
export * from "./src/api/user/model";
export * from "./src/types";
export * from "./src/enums";

// 配置导出
export { configJavaAxios, configPythonAxios } from "./src/config";

// Axios 实例导出（用于 token 刷新后重发请求）
export { javaService } from "./src/utils/request";
export { pythonService } from "./src/utils/requestPy";

// API 导出
export {
  AlgorithmAPI,
  AuthAPI,
  DatasetAPI,
  DatasetItemAPI,
  ItemFileAPI,
  ExportTaskAPI,
  DeptAPI,
  DictAPI,
  FileAPI,
  ImageInputHistoryAPI,
  MenuAPI,
  ModelAPI,
  RoleAPI,
  UserAPI,
};
