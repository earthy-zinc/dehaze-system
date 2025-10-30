// API 模块导入
import AlgorithmAPI from "@/api/algorithm";
import AuthAPI from "@/api/auth";
import DatasetAPI from "@/api/dataset";
import DeptAPI from "@/api/dept";
import DictAPI from "@/api/dict";
import FileAPI from "@/api/file";
import MenuAPI from "@/api/menu";
import ModelAPI from "@/api/model";
import RoleAPI from "@/api/role";
import UserAPI from "@/api/user";

// API 模型导出
export * from "@/api/algorithm/model";
export * from "@/api/auth/model";
export * from "@/api/dataset/model";
export * from "@/api/dept/model";
export * from "@/api/dict/model";
export * from "@/api/file/model";
export * from "@/api/menu/model";
export * from "@/api/model/model";
export * from "@/api/role/model";
export * from "@/api/user/model";
export * from "@/types";
export * from "@/enums";

// 配置导出
export {configJavaAxios, configPythonAxios} from "@/config";

// API 导出
export {
    AlgorithmAPI,
    AuthAPI,
    DatasetAPI,
    DeptAPI,
    DictAPI,
    FileAPI,
    MenuAPI,
    ModelAPI,
    RoleAPI,
    UserAPI,
};
